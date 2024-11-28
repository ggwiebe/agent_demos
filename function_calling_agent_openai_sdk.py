# Databricks notebook source
# MAGIC %md
# MAGIC # Function Calling Agent w/ Retriever 
# MAGIC
# MAGIC In this notebook, we construct a function-calling Agent with a Retriever tool using MLflow + the OpenAI SDK connected to Databricks Model Serving. This Agent is encapsulated in a MLflow PyFunc class called `FunctionCallingAgent()`.

# COMMAND ----------

# # If running this notebook by itself, uncomment these.
# %pip install --upgrade -qqqq databricks-agents databricks-vectorsearch "git+https://github.com/mlflow/mlflow.git" databricks-sdk[openai] pydantic "git+https://github.com/unitycatalog/unitycatalog.git#subdirectory=ai/core" "git+https://github.com/unitycatalog/unitycatalog.git#subdirectory=ai/integrations/openai"
# dbutils.library.restartPython()

# COMMAND ----------

import json
import os
from typing import Any, Callable, Dict, List, Optional, Union
import mlflow
from dataclasses import asdict, dataclass
import pandas as pd
from mlflow.models import set_model, ModelConfig
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest, Message
from databricks.sdk import WorkspaceClient
import os

import logging
logging.getLogger('mlflow').setLevel(logging.ERROR)

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from pyspark.errors import SparkRuntimeException
from unitycatalog.ai.openai.toolkit import UCFunctionToolkit

class UCTool:
  def __init__(self, config: Dict[str, Any]):
    self.config = config
    self.uc_client = DatabricksFunctionClient()
    self.toolkit = UCFunctionToolkit(function_names=[f"{self.config.get('uc_tool_fqn')}"], client=self.uc_client)
    

  @mlflow.trace(span_type="TOOL", name="uc_tool")
  def __call__(
        self, **kwargs
    ) -> Dict[str, str]: 
    span = mlflow.get_current_active_span()
    span.set_attributes({"uc_tool_name": self.config.get("uc_tool_fqn")})
    traced_exec_function = self.uc_client.execute_function
    args_json = json.loads(json.dumps(kwargs, default=str))
    # if 'code' in args_json:
    #   args_json['code'] = args_json['code'].encode('unicode_escape').decode('utf-8').replace("\\", "\\\\")

    try:
      result = traced_exec_function(function_name=self.config.get("uc_tool_fqn"), parameters=args_json)
      return result.to_json()
    except SparkRuntimeException as e:
      try:
        error = e.getMessageParameters()['error'].replace('File "<string>",', '').strip()
      except Exception as e:
        error = e.getMessageParameters()['error']
      return {
        'status': 'Error in generated code.  Please think step-by-step about the error and try calling this tool again.',
        'error': error
      }
    except Exception as e:
      return {
        'status': 'Error in generated code.  Please think step-by-step about the error and try again calling this tool again.  If you included `\` in your code, consider removing those.',
        'error': str(e)
      }



# COMMAND ----------

# DBTITLE 1,Retriever tool
from mlflow.entities import Document
from databricks.vector_search.client import VectorSearchClient

class VectorSearchRetriever:
    """
    Class using Databricks Vector Search to retrieve relevant documents.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_search_client = VectorSearchClient(disable_notice=True)
        self.vector_search_index = self.vector_search_client.get_index(
            index_name=self.config.get("vector_search_index")
        )

        vector_search_schema = self.config.get("vector_search_schema")
        mlflow.models.set_retriever_schema(
            name=self.config.get("vector_search_index"),
            primary_key=vector_search_schema.get("primary_key"),
            text_column=vector_search_schema.get("chunk_text"),
            doc_uri=vector_search_schema.get("document_uri"),
        )

    @mlflow.trace(span_type="RETRIEVER", name="vector_search")
    def __call__(
        self, query: str, filters: Dict[Any, Any] = None
    ) -> List[Document]:
        """
        Performs vector search to retrieve relevant chunks.

        Args:
            query: Search query.
            filters: Optional filters to apply to the search, must follow the Databricks Vector Search filter spec (https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#use-filters-on-queries)

        Returns:
            List of retrieved Documents.
        """

        traced_search = mlflow.trace(
            self.vector_search_index.similarity_search,
            name="vector_search.similarity_search",
        )

        vector_search_schema = self.config.get("vector_search_schema")
        additional_metadata_columns = (
            vector_search_schema.get("additional_metadata_columns") or []
        )

        columns = [
            vector_search_schema.get("primary_key"),
            vector_search_schema.get("chunk_text"),
            vector_search_schema.get("document_uri"),
        ] + additional_metadata_columns

        # de-duplicate
        columns = list(set(columns))

        # Parse filters into Vector Search compatible format
        vs_filters = self.parse_filters(filters) if filters else None

        results = traced_search(
            query_text=query,
            filters=vs_filters,
            columns=columns,
            **self.config.get("vector_search_parameters"),
        )

        # if filters is None:
        #     results = traced_search(
        #         query_text=query,
        #         columns=columns,
        #         **self.config.get("vector_search_parameters"),
        #     )
        # else:
        #     results = traced_search(
        #         query_text=query,
        #         filters=filters,
        #         columns=columns,
        #         **self.config.get("vector_search_parameters"),
        #     )

        vector_search_threshold = self.config.get("vector_search_threshold")
        documents = self.convert_vector_search_to_documents(
            results, vector_search_threshold
        )

        return [asdict(doc) for doc in documents]

    @mlflow.trace(span_type="PARSER")
    def convert_vector_search_to_documents(
        self, vs_results, vector_search_threshold
    ) -> List[Document]:
        column_names = []
        for column in vs_results["manifest"]["columns"]:
            column_names.append(column)

        docs = []
        result_row_count = vs_results["result"]["row_count"]
        if result_row_count > 0:
            for item in vs_results["result"]["data_array"]:
                metadata = {}
                score = item[-1]
                if score >= vector_search_threshold:
                    metadata["similarity_score"] = score
                    for i, field in enumerate(item[0:-1]):
                        metadata[column_names[i]["name"]] = field
                    # put contents of the chunk into page_content
                    page_content = metadata[
                        self.config.get("vector_search_schema").get("chunk_text")
                    ]
                    del metadata[
                        self.config.get("vector_search_schema").get("chunk_text")
                    ]

                    doc = Document(
                        page_content=page_content, metadata=metadata
                    )
                    docs.append(doc)

        return docs
    
    @mlflow.trace(span_type="PARSER")
    def parse_filters(self, filters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse input filters into Vector Search compatible format.

        Args:
            filters: List of input filters in the new format.

        Returns:
            Filters in Vector Search compatible format.
        """
        vs_filters = {}
        for filter_item in filters:
            key = filter_item['field']
            value = filter_item['filter']
            
            if isinstance(value, list):
                vs_filters[key] = {"OR": value}
            elif isinstance(value, dict):
                operator, operand = next(iter(value.items()))
                if operator in ["<", "<=", ">", ">="]:
                    vs_filters[f"{key} {operator}"] = operand
                elif operator.upper() == "LIKE":
                    vs_filters[f"{key} LIKE"] = operand
                elif operator.upper() == "NOT":
                    vs_filters[f"{key} !="] = operand
            else:
                vs_filters[key] = value
        return vs_filters

# COMMAND ----------

# DBTITLE 1,Agent
class FunctionCallingAgent(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that does function-calling with tools using OpenAI SDK
    """
    # def __init__(self, agent_config: dict = None):
    #     self.__agent_config = agent_config
    #     if self.__agent_config is None:
    #         self.__agent_config = globals().get("__mlflow_model_config__")

    #     print(globals().get("__mlflow_model_config__"))

    def __init__(self, agent_config: dict = None):
        self.__agent_config = agent_config
        if self.__agent_config is not None:
            self.config = mlflow.models.ModelConfig(development_config=self.__agent_config)
        else:
            self.config = mlflow.models.ModelConfig(development_config="config.yml")
            
                
        # vector_search_schema = self.config.get("retriever_config").get("schema")
        # mlflow.models.set_retriever_schema(
        #     primary_key=vector_search_schema.get("primary_key"),
        #     text_column=vector_search_schema.get("chunk_text"),
        #     doc_uri=vector_search_schema.get("doc_uri"),
        # )

        # OpenAI client used to query Databricks Chat Completion endpoint
        # self.model_serving_client = OpenAI(
        #     api_key=os.environ.get("OPENAI_API_KEY"),
        #     base_url=str(os.environ.get("OPENAI_BASE_URL")), #+ "/serving-endpoints",
        # )
        w = WorkspaceClient()
        self.model_serving_client = w.serving_endpoints.get_open_ai_client()

        # self.model_serving_client = get_deploy_client("databricks")

        # Initialize the tools
        self.tool_functions = {}
        self.tool_json_schemas =[]
        for tool in self.config.get("llm_config").get("tools"):
            # print(tool)
            # 1 Instantiate the tool's class w/ by passing the tool's config to it
            # 2 Store the instantiated tool to use later
            self.tool_functions[tool.get("tool_name")] = globals()[tool.get("tool_class_name")](config=tool)
            self.tool_json_schemas.append(tool.get("tool_input_json_schema"))

        # print(self.tool_json_schemas)

        # # Init the retriever for `search_customer_notes_for_topic` tool
        # self.retriever_tool = VectorSearchRetriever(
        #     self.config.get("search_note_tool").get("retriever_config")
        # )

        # self.tool_functions = {
        #     "retrieve_documents": self.retriever_tool,
        # }

        self.chat_history = []

    @mlflow.trace(name="customer_service_transcripts", span_type="AGENT")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> StringResponse:
        ##############################################################################
        # Extract `messages` key from the `model_input`
        messages = self.get_messages_array(model_input)

        ##############################################################################
        # Parse `messages` array into the user's query & the chat history
        with mlflow.start_span(name="parse_input", span_type="PARSER") as span:
            span.set_inputs({"messages": messages})
            user_query = self.extract_user_query_string(messages)
            # Save the history inside the Agent's internal state
            self.chat_history = self.extract_chat_history(messages)
            span.set_outputs(
                {"user_query": user_query, "chat_history": self.chat_history}
            )

        ##############################################################################
        # Call LLM

        # messages to send the model
        # For models with shorter context length, you will need to trim this to ensure it fits within the model's context length
        system_prompt = self.config.get("llm_config").get("llm_system_prompt_template")
        messages = (
            [{"role": "system", "content": system_prompt}]
            + self.chat_history  # append chat history for multi turn
            + [{"role": "user", "content": user_query}]
        )

        # Call the LLM to recursively calls tools and eventually deliver a generation to send back to the user
        (
            model_response,
            messages_log_with_tool_calls,
        ) = self.recursively_call_and_run_tools(messages=messages)

        # If your front end keeps of converastion history and automatically appends the bot's response to the messages history, remove this line.
        messages_log_with_tool_calls.append(model_response.choices[0].message.to_dict()) #OpenAI client
        # messages_log_with_tool_calls.append(model_response.choices[0]["message"]) #Mlflow client

        # remove the system prompt - this should not be exposed to the Agent caller
        messages_log_with_tool_calls = messages_log_with_tool_calls[1:]

        
        return {
            "content": model_response.choices[0].message.content, #openai client
            # "content": model_response.choices[0]["message"]["content"], #mlflow client
            # messages should be returned back to the Review App (or any other front end app) and stored there so it can be passed back to this stateless agent with the next turns of converastion.

            "messages": messages_log_with_tool_calls,
        }

    @mlflow.trace(span_type="AGENT")
    def recursively_call_and_run_tools(self, max_iter=10, **kwargs):
        messages = kwargs["messages"]
        del kwargs["messages"]
        i = 0
        while i < max_iter:
            response = self.chat_completion(messages=messages, tools=True)
            assistant_message = response.choices[0].message #openai client
            # assistant_message = response.choices[0]["message"] #mlflow client
            tool_calls = assistant_message.tool_calls #openai
            # tool_calls = assistant_message.get('tool_calls')#mlflow client
            if tool_calls is None:
                # the tool execution finished, and we have a generation
                return (response, messages)
            tool_messages = []
            for tool_call in tool_calls:  # TODO: should run in parallel
                function = tool_call.function #openai
                # function = tool_call['function'] #mlflow
                # uc_func_name = decode_function_name(function.name)
                args = json.loads(function.arguments) #openai
                # args = json.loads(function['arguments']) #mlflow
                # result = exec_uc_func(uc_func_name, **args)
                result = self.execute_function(function.name, args) #openai
                # result = self.execute_function(function['name'], args) #mlflow

                # format for the LLM, will throw exception if not possible
                try:
                    result_for_llm = json.dumps(result)
                except Exception as e:
                    result_for_llm = str(result)

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_for_llm,
                } #openai
                # tool_message = {
                #     "role": "tool",
                #     "tool_call_id": tool_call['id'],
                #     "content": result,
                # } #mlflow
                tool_messages.append(tool_message)
                # print(tool_message)
            assistant_message_dict = assistant_message.dict().copy() #openai
            # assistant_message_dict = assistant_message.copy() #mlflow
            del assistant_message_dict["content"]
            del assistant_message_dict["function_call"] #openai only
            if "audio" in assistant_message_dict: del assistant_message_dict["audio"] #llama70b hack
            # print(assistant_message_dict)
            messages = (
                messages
                + [
                    assistant_message_dict,
                ]
                + tool_messages
            )
            # print(messages)
            i += 1
        # TODO: Handle more gracefully
        raise "ERROR: max iter reached"

    @mlflow.trace(span_type="FUNCTION")
    def execute_function(self, function_name, args):
        the_function = self.tool_functions.get(function_name)
        result = the_function(**args)
        return result

    def chat_completion(self, messages: List[Dict[str, str]], tools: bool = False):
        endpoint_name = self.config.get("llm_config").get("llm_endpoint_name")
        llm_options = self.config.get("llm_config").get("llm_parameters")

        # # Trace the call to Model Serving - openai versio
        traced_create = mlflow.trace(
            self.model_serving_client.chat.completions.create,
            name="chat_completions_api",
            span_type="CHAT_MODEL",
        )

        # Trace the call to Model Serving - mlflow version 
        # traced_create = mlflow.trace(
        #     self.model_serving_client.predict,
        #     name="chat_completions_api",
        #     span_type="CHAT_MODEL",
        # )

        #mlflow client - start
        # if tools:
        #     # Get all tools
        #     tools = self.tool_json_schemas

        #     inputs = {
        #         "messages": messages,
        #         "tools": tools,
        #         **llm_options,
        #     }
        # else:
        #     inputs = {
        #         "messages": messages,
        #         **llm_options,
        #     }

        # # Use the traced_create to make the prediction
        # return traced_create(
        #     endpoint=endpoint_name,
        #     inputs=inputs,
        # )

        #mlflow client - end
        # Openai - start
        if tools:
            return traced_create(
                model=endpoint_name,
                messages=messages,
                tools=self.tool_json_schemas,
                parallel_tool_calls=False,
                **llm_options,
            )
        else:
            return traced_create(model=endpoint_name, messages=messages, **llm_options)
        # Openai - end

    @mlflow.trace(span_type="PARSER")
    def get_messages_array(
        self, model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame]
    ) -> List[Dict[str, str]]:
        if type(model_input) == ChatCompletionRequest:
            return model_input.messages
        elif type(model_input) == dict:
            return model_input.get("messages")
        elif type(model_input) == pd.DataFrame:
            return model_input.iloc[0].to_dict().get("messages")

    @mlflow.trace(span_type="PARSER")
    def extract_user_query_string(
        self, chat_messages_array: List[Dict[str, str]]
    ) -> str:
        """
        Extracts user query string from the chat messages array.

        Args:
            chat_messages_array: Array of chat messages.

        Returns:
            User query string.
        """

        if isinstance(chat_messages_array, pd.Series):
            chat_messages_array = chat_messages_array.tolist()

        if isinstance(chat_messages_array[-1], dict):
            return chat_messages_array[-1]["content"]
        elif isinstance(chat_messages_array[-1], Message):
            return chat_messages_array[-1].content
        else:
            return chat_messages_array[-1]

    @mlflow.trace(span_type="PARSER")
    def extract_chat_history(
        self, chat_messages_array: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Extracts the chat history from the chat messages array.

        Args:
            chat_messages_array: Array of chat messages.

        Returns:
            The chat history.
        """
        # Convert DataFrame to dict
        if isinstance(chat_messages_array, pd.Series):
            chat_messages_array = chat_messages_array.tolist()

        # Dictionary, return as is
        if isinstance(chat_messages_array[0], dict):
            return chat_messages_array[:-1]  # return all messages except the last one
        # MLflow Message, convert to Dictionary
        elif isinstance(chat_messages_array[0], Message):
            new_array = []
            for message in chat_messages_array[:-1]:
                new_array.append(asdict(message))
            return new_array
        else:
            raise ValueError(
                "chat_messages_array is not an Array of Dictionary, Pandas DataFrame, or array of MLflow Message."
            )

# tell MLflow logging where to find the agent's code
set_model(FunctionCallingAgent())

# COMMAND ----------

# DBTITLE 1,debugging code
debug = False

if debug:
  agent = FunctionCallingAgent(agent_config='config.yml')

  vibe_check_query = {
      "messages": [
          # {"role": "user", "content": f"what is agent evaluation?"},
          # {"role": "user", "content": f"calculate the value of 2+2?"},
          {"role": "user", "content": f"what are recent customer issues?  what words appeared most frequently?"},
      ]
  }

  agent.predict(model_input=vibe_check_query)
