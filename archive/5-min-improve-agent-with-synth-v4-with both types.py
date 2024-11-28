# Databricks notebook source
# MAGIC %md
# MAGIC # 1/ Setup

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow-skinny databricks-sdk[openai]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC ##1.0/ Configure user name to avoid conflicts with other users running the demo

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient

# Get current user's name & email to ensure each user doesn't over-write other user's outputs
w = WorkspaceClient()
user_email = w.current_user.me().user_name
user_name = user_email.split("@")[0].replace(".", "_")

experiment = mlflow.set_experiment(f"/Users/{user_email}/agents-demo-experiment")

UC_MODEL_NAME = f"agents_demo.synthetic_data.db_docs__{user_name}"

print(f"User: {user_name}")
print()
print(f"MLflow Experiment: {experiment.name}")
print()
print(f"Unity Catalog Model: {UC_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1/ Load the docs corpus from the Cookbook repo

# COMMAND ----------

# MAGIC %md
# MAGIC First, we load the documents (Databricks documentation) used by our Agent.

# COMMAND ----------

import pandas as pd

databricks_docs_url = "https://github.com/databricks/genai-cookbook/raw/refs/heads/main/quick_start_demo/chunked_databricks_docs.snappy.parquet"

CHUNKS = (
    pd.read_parquet(databricks_docs_url)
    .rename(columns={"chunked_text": "content", "url": "doc_uri"})[
        ["content", "doc_uri"]
    ]
)

patterns = [
    "/archive/", "/release-notes/", "/resources/", "/rag-studio/", 
    "/sql/language-manual/", "/admin/", "/migration/", "/dev-tools/", "/error-messages/"
]

for pattern in patterns:
    CHUNKS = CHUNKS.loc[~CHUNKS["doc_uri"].str.contains(pattern)]

CHUNKS = CHUNKS[:500].to_dict("records")
display(CHUNKS)

# COMMAND ----------

# Export the documents to save in the MLlfow model
chunks_dict = CHUNKS
jsonl_file_name = "./chunks.jsonl"

with open(jsonl_file_name, 'w') as jsonl_file:
    for chunk in CHUNKS:
        jsonl_file.write(json.dumps(chunk) + '\n')

# COMMAND ----------

import pandas as pd

databricks_docs_url = "https://github.com/databricks/genai-cookbook/raw/refs/heads/main/quick_start_demo/chunked_databricks_docs.snappy.parquet"

CHUNKS = (
    pd.read_parquet(databricks_docs_url)[:500]
    .rename(columns={"chunked_text": "content", "url": "doc_uri"})[
        ["content", "doc_uri"]
    ] # filter lower quality documents
    .[~CHUNKS["doc_uri"].str.contains(pattern) for pattern in [
        "/archive/", "/release-notes/", "/resources/", "/rag-studio/", 
        "/sql/language-manual/", "/admin/", "/migration/", "/dev-tools/", "/error-messages/"
    ]]
    .to_dict("records")
)
display(CHUNKS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2/ Generate synthetic evaluation data

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC **Challenges Addressed**
# MAGIC 1. How to start quality evaluation with diverse, representative data without SMEs spending months labeling?
# MAGIC
# MAGIC **What is happening?**
# MAGIC - We pass the documents to the Synthetic API along with a `num_evals` and prompt-like `guidelines` to tailor the generated questions for our use case. This API uses a proprietary synthetic generation pipeline developed by Mosaic AI Research.
# MAGIC - The API produces `num_evals` questions, each coupled with the source document & a list of facts, generated based on the source document.  Each fact must be present in the Agent's response for it to be considered correct.
# MAGIC
# MAGIC *Why does the the API generates a list of facts, rather than a fully written answer.  This...*
# MAGIC - Makes SME review more efficient: by focusing on facts rather than a full response, they can review/edit more quickly.
# MAGIC - Improves the accuracy of our proprietary LLM judges.

# COMMAND ----------

# Use the synthetic eval generation API to get some evals
from databricks.agents.eval import generate_evals_df

# "Ghost text" for guidelines - feel free to modify as you see fit.
guidelines = f"""
# Task Description
The Agent is a RAG chatbot that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.

# User personas
- A developer who is new to the Databricks platform
- An experienced, highly technical Data Scientist or Data Engineer

# Example questions
- what API lets me parallelize operations over rows of a delta table?
- Which cluster settings will give me the best performance when using Spark?

# Additional Guidelines
- Questions should be succinct, and human-like
"""

num_evals = 25
evals = generate_evals_df(
    docs=pd.DataFrame(CHUNKS), # Pass your docs. Pandas/Spark Dataframes with columns `content STRING, doc_uri STRING` are suitable.
    num_evals=num_evals,
    guidelines=guidelines
)
display(evals)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2/ Write the Agent's code

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1/ Base Llama 3 agent

# COMMAND ----------

# MAGIC %md
# MAGIC **Challenges addressed**
# MAGIC - How do I track different versions of my agent's code/config?
# MAGIC - How do I enable observability, monitoring, and debugging of my Agent’s logic?
# MAGIC
# MAGIC **What is happening?**
# MAGIC
# MAGIC First, we create an Agent. To focus on how Agent Evaluation works, we start with a simple Agent using only an LLM. Later in this notebook, we'll create and evaluate an Agent with a Retriever.
# MAGIC
# MAGIC We write the Agent using OpenAI's SDK and Python, but Mosaic AI supports all popular frameworks such as LangGraph, AutoGen, LlamaIndex, and more.
# MAGIC
# MAGIC *A few things to note about the code:*
# MAGIC 1. The code is written to `llm_only_agent.py` in order to use [MLflow Models from Code](https://www.mlflow.org/blog/models_from_code) for logging, enabling easy tracking of each iteration as we tune the Agent for quality.
# MAGIC 2. The code is parameterized with an [MLflow Model Configuration](https://docs.databricks.com/en/generative-ai/agent-framework/create-agent.html#use-parameters-to-configure-the-agent), enabling easy tuning of these parameters for quality improvement.
# MAGIC 3. The code is wrapped in an MLflow PyFunc model, making the Agent's code deployment-ready so any iteration can be shared with stakeholders for testing.
# MAGIC 4. The code implements [MLflow Tracing](https://docs.databricks.com/en/mlflow/mlflow-tracing.html) for unified observability during development and production. The same trace defined here will be logged for every production request post-deployment. For agent authoring frameworks, you can tracing with one line of code: `mlflow.framework_agent.autolog()`.

# COMMAND ----------

# MAGIC %%writefile llm_only_agent.py
# MAGIC
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from openai import OpenAI
# MAGIC from typing import Any, Union, Dict, List
# MAGIC import mlflow
# MAGIC from mlflow.models.rag_signatures import ChatCompletionResponse, ChatCompletionRequest
# MAGIC import dataclasses
# MAGIC import pandas as pd
# MAGIC
# MAGIC # Define the configuration that will be used by the Agent.
# MAGIC DEFAULT_CONFIG = {
# MAGIC     'endpoint_name': "agents-demo-gpt4o",
# MAGIC     'temperature': 0.01,
# MAGIC     'max_tokens': 1000,
# MAGIC     'system_prompt': "The Agent is a RAG chatbot that answers questions about Databricks. Questions unrelated to Databricks are irrelevant."
# MAGIC }
# MAGIC
# MAGIC class LLMOnlyAgent(mlflow.pyfunc.PythonModel):
# MAGIC     """
# MAGIC     Class representing an Agent that does uses an LLM only.
# MAGIC     """
# MAGIC     def __init__(self):
# MAGIC         # Initialize the OpenAI SDK client connected to Model Serving.
# MAGIC         w = WorkspaceClient()
# MAGIC         self.model_serving_client: OpenAI = w.serving_endpoints.get_open_ai_client()
# MAGIC
# MAGIC         # Load the Agent's configuration from MLflow Model Config.
# MAGIC         self.config = mlflow.models.ModelConfig(development_config=DEFAULT_CONFIG)
# MAGIC
# MAGIC     @mlflow.trace(name="rag_agent", span_type="AGENT")
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         context: Any = None,
# MAGIC         model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
# MAGIC         params: Any = None,
# MAGIC     ) -> ChatCompletionResponse:
# MAGIC         """
# MAGIC         Main function.  Respond to a user's request.
# MAGIC         """
# MAGIC         # Extract the user's request from the `model_input`
# MAGIC         request = self.get_request_dict(model_input)
# MAGIC
# MAGIC         # Add system prompt to messages history
# MAGIC         request = self.prepend_system_prompt(request, self.config.get("system_prompt"))
# MAGIC
# MAGIC         # Generate with the LLM
# MAGIC         return self.chat_completion(
# MAGIC             request=request
# MAGIC         )
# MAGIC     
# MAGIC     ###
# MAGIC     # Helper functions
# MAGIC     ###
# MAGIC     @mlflow.trace(span_type="FUNCTION")
# MAGIC     def prepend_system_prompt(
# MAGIC         self, request: ChatCompletionRequest, system_prompt: str
# MAGIC     ) -> ChatCompletionRequest:
# MAGIC         """
# MAGIC         Add system prompt to the user's request.
# MAGIC         """
# MAGIC         if isinstance(request, ChatCompletionRequest):
# MAGIC             request = dataclasses.asdict(request)
# MAGIC         if request["messages"][0]["role"] != "system":
# MAGIC             return {
# MAGIC                 **request,
# MAGIC                 "messages": [
# MAGIC                     {"role": "system", "content": system_prompt},
# MAGIC                     *request["messages"],
# MAGIC                 ],
# MAGIC             }
# MAGIC         return request
# MAGIC
# MAGIC     
# MAGIC     @mlflow.trace(span_type="PARSER")
# MAGIC     def get_request_dict(self, 
# MAGIC         model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame, List]
# MAGIC     ) -> List[Dict[str, str]]:
# MAGIC         """
# MAGIC         Convert the PyFunc input to a single dictionary.
# MAGIC         """
# MAGIC         if type(model_input) == list:
# MAGIC             # get the first row
# MAGIC             model_input = list[0]
# MAGIC         elif type(model_input) == pd.DataFrame:
# MAGIC             # return the first row, this model doesn't support batch input
# MAGIC             return model_input.to_dict(orient="records")[0]
# MAGIC         
# MAGIC         # now, try to unpack the single item or first row of batch input
# MAGIC         if type(model_input) == ChatCompletionRequest:
# MAGIC             return asdict(model_input)
# MAGIC         elif type(model_input) == dict:
# MAGIC             return model_input
# MAGIC         
# MAGIC     def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
# MAGIC         """
# MAGIC         Call the LLM configured via the ModelConfig using the OpenAI SDK
# MAGIC         """
# MAGIC         traced_chat_completions_create_fn = mlflow.trace(
# MAGIC             self.model_serving_client.chat.completions.create,
# MAGIC             name="chat_completions_api",
# MAGIC             span_type="CHAT_MODEL",
# MAGIC         )
# MAGIC         request = {**request, "temperature": self.config.get("temperature"), "max_tokens": self.config.get("max_tokens")}
# MAGIC         return traced_chat_completions_create_fn(
# MAGIC             model=self.config.get("endpoint_name"), **request
# MAGIC         ).to_dict()
# MAGIC
# MAGIC # tell MLflow logging where to find the agent's code
# MAGIC mlflow.models.set_model(LLMOnlyAgent())

# COMMAND ----------

# MAGIC %md
# MAGIC Empty `__init__.py` to allow the `LLMOnlyAgent()` to be imported.

# COMMAND ----------

# MAGIC %%writefile ___init__.py
# MAGIC
# MAGIC # Empty file
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Let's test the Agent for a sample query to see the MLflow Trace.

# COMMAND ----------

from llm_only_agent import LLMOnlyAgent
import mlflow

agent = LLMOnlyAgent()

response = agent.predict(
    model_input={"messages": [{"role": "user", "content": "What is Databricks?"}]}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2/ Evaluate the Base agent

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Challenges addressed**
# MAGIC - What are the right metrics to evaluate quality?  How do I trust the outputs of these metrics?
# MAGIC - I need to evaluate many ideas - how do I…
# MAGIC     - …run evaluation quickly so the majority of my time isn’t spent waiting?
# MAGIC     - …quickly compare these different versions of my agent on cost/quality/latency?
# MAGIC - How do I quickly identify the root cause of any quality problems?
# MAGIC
# MAGIC **What is happening?**
# MAGIC
# MAGIC Now, we run Agent Evaluation's propietary LLM judges using the synthetic evaluation set to see the quality/cost/latency of the Agent and identify any root causes of quality issues.  Agent Evaluation is tightly integrated with `mlflow.evaluate()`.  
# MAGIC
# MAGIC Mosaic AI Research has invested signficantly in the quality AND speed of the LLM judges, optimizing the judges to agree with human raters.  Read more [details in our blog](https://www.databricks.com/blog/databricks-announces-significant-improvements-built-llm-judges-agent-evaluation) about how our judges outperform the competition.  
# MAGIC
# MAGIC Once evaluation runs, click `View Evaluation Results` to open the MLflow UI for this Run.  This lets you:
# MAGIC - See summary metrics
# MAGIC - See root cause analysis that identifies the most important issues to fix
# MAGIC - Inspect individual responses to gain intuition about how the Agent is performing
# MAGIC - See the judge outputs to understand why the responses were graded as good/bad
# MAGIC - Compare between multiple runs to see how quality changed between experiments
# MAGIC
# MAGIC You can also inspect the other tabs:
# MAGIC - `Overview` lets you see the Agent's config/parameters
# MAGIC - `Artifacts` lets you see the Agent's code
# MAGIC
# MAGIC This UIs, coupled with the speed of evaluation, help you efficiently test your hypotheses to improve quality, letting you reach the production quality bar in less time. 

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import ChatCompletionResponse, ChatCompletionRequest

with mlflow.start_run(run_name="agent_without_retriever"):

    # Log the Agent's code/config to MLflow
    model_info = mlflow.pyfunc.log_model(
        python_model="llm_only_agent.py",
        artifact_path="agent",
        model_config={ # Parameters defined in the Agent's code above
            "endpoint_name": "agents-demo-gpt4o",
            "temperature": 0.01,
            "max_tokens": 1000,
            "system_prompt": "The Agent is a RAG chatbot that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.",
        },
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=ChatCompletionResponse(),
        ),
        pip_requirements=["databricks-sdk[openai]", "mlflow", "databricks-agents"],
    )

    # Run evaluation
    eval_results = mlflow.evaluate(
        data=evals,  # Your evaluation set
        model=model_info.model_uri, # Logged Agent from above
        model_type="databricks-agent",  # activate Mosaic AI Agent Evaluation
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # 3/ RAG agent with keyword-based retriever

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1/ Custom RAG Agent

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's create a RAG-based Agent to compare to the base Agent.   Like before, you can see the MLflow Trace to quickly understand how the Agent works.

# COMMAND ----------



# COMMAND ----------

# MAGIC %%writefile fc_agent.py
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from openai import OpenAI
# MAGIC import pandas as pd
# MAGIC from typing import Any, Union, Dict, List
# MAGIC import mlflow
# MAGIC from mlflow.models.rag_signatures import ChatCompletionResponse, ChatCompletionRequest
# MAGIC import dataclasses
# MAGIC import json
# MAGIC
# MAGIC DEFAULT_CONFIG = {
# MAGIC     'endpoint_name': "agents-demo-gpt4o",
# MAGIC     'temperature': 0.01,
# MAGIC     'max_tokens': 1000,
# MAGIC     'system_prompt': """The Agent is a RAG chatbot that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.
# MAGIC
# MAGIC     You are a helpful assistant that answers questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request.
# MAGIC     """,
# MAGIC     'max_context_chars': 4096 * 4
# MAGIC }
# MAGIC
# MAGIC RETRIEVER_TOOL_SPEC = [{
# MAGIC     "type": "function",
# MAGIC     "function": {
# MAGIC         "name": "search_product_docs",
# MAGIC         "description": "Use this tool to search for Databricks product documentation.",
# MAGIC         "parameters": {
# MAGIC             "type": "object",
# MAGIC             "required": ["query"],
# MAGIC             "additionalProperties": False,
# MAGIC             "properties": {
# MAGIC                 "query": {
# MAGIC                     "description": "query to look up in retriever",
# MAGIC                     "type": "string",
# MAGIC                 }
# MAGIC             },
# MAGIC         },
# MAGIC     },
# MAGIC }]
# MAGIC
# MAGIC DOCS_ARTIFACT_KEY = "docs_data"
# MAGIC
# MAGIC class FCAgent(mlflow.pyfunc.PythonModel):
# MAGIC     """
# MAGIC     Class representing an Agent that does simple RAG using keyword-based search.
# MAGIC     """
# MAGIC
# MAGIC     def load_context(self, context=None):
# MAGIC         """
# MAGIC         Load the retriever's docs.
# MAGIC         """
# MAGIC         # Load the document data - either from the logged MLflow artifact or the web URL
# MAGIC         # if context is not None:
# MAGIC         #     raw_docs_json = context.artifacts[DOCS_ARTIFACT_KEY]
# MAGIC         #     self.docs = pd.read_json(raw_docs_json, lines=True)[:500].to_dict("records")
# MAGIC         # else:
# MAGIC         # raw_docs_parquet = "https://github.com/databricks/genai-cookbook/raw/refs/heads/main/quick_start_demo/chunked_databricks_docs.snappy.parquet"
# MAGIC         # self.docs = pd.read_parquet(raw_docs_parquet)[:500].to_dict("records")
# MAGIC
# MAGIC         
# MAGIC
# MAGIC     def __init__(self):
# MAGIC         """
# MAGIC         Initialize the OpenAI SDK client connected to Model Serving.
# MAGIC         Load the Agent's configuration from MLflow Model Config.
# MAGIC         """
# MAGIC         self.docs = None
# MAGIC
# MAGIC         # Initialize OpenAI SDK connected to Model Serving
# MAGIC         w = WorkspaceClient()
# MAGIC         self.model_serving_client: OpenAI = w.serving_endpoints.get_open_ai_client()
# MAGIC
# MAGIC         # Load config
# MAGIC         self.config = mlflow.models.ModelConfig(development_config=DEFAULT_CONFIG)
# MAGIC
# MAGIC         # Configure playground & review app & agent evaluation to display / see the chunks from the retriever 
# MAGIC         mlflow.models.set_retriever_schema(
# MAGIC             name="db_docs",
# MAGIC             primary_key="chunk_id",
# MAGIC             text_column="chunked_text",
# MAGIC             doc_uri="doc_uri",
# MAGIC         )
# MAGIC
# MAGIC         raw_docs_parquet = "https://github.com/databricks/genai-cookbook/raw/refs/heads/main/quick_start_demo/chunked_databricks_docs.snappy.parquet"
# MAGIC         self.docs = pd.read_parquet(raw_docs_parquet)[:500].to_dict("records")
# MAGIC
# MAGIC         self.tool_functions = {
# MAGIC             'search_product_docs': self.search_product_docs
# MAGIC         }
# MAGIC
# MAGIC     @mlflow.trace(name="rag_agent", span_type="AGENT")
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         context: Any = None,
# MAGIC         model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
# MAGIC         params: Any = None,
# MAGIC     ) -> ChatCompletionResponse:
# MAGIC         ##############################################################################
# MAGIC         # Extract `messages` key from the `model_input`
# MAGIC         request = self.get_request_dict(model_input)
# MAGIC
# MAGIC         # Ask the LLM to call tools & generate the response
# MAGIC         return self.recursively_call_and_run_tools(
# MAGIC             **request
# MAGIC         )
# MAGIC
# MAGIC     @mlflow.trace(span_type="FUNCTION")
# MAGIC     def prepend_system_prompt(
# MAGIC         self, request: ChatCompletionRequest, system_prompt: str
# MAGIC     ) -> ChatCompletionRequest:
# MAGIC         if isinstance(request, ChatCompletionRequest):
# MAGIC             request = dataclasses.asdict(request)
# MAGIC         if request["messages"][0]["role"] != "system":
# MAGIC             return {
# MAGIC                 **request,
# MAGIC                 "messages": [
# MAGIC                     {"role": "system", "content": system_prompt},
# MAGIC                     *request["messages"],
# MAGIC                 ],
# MAGIC             }
# MAGIC         return request
# MAGIC
# MAGIC     def search_product_docs(self, query: str) -> list[dict]:
# MAGIC         keywords = self.extract_keywords(query)
# MAGIC         return self.retrieve_documents(keywords)
# MAGIC     
# MAGIC     # Helper function for keyword-based retriever - would be replaced by the Vector Index
# MAGIC     @mlflow.trace(span_type="PARSER")
# MAGIC     def extract_keywords(self, query: str) -> list[str]:
# MAGIC         prompt = f"""Given a user query, extract the most salient keywords from the user query. These keywords will be used in a search engine to retrieve relevant documents to the query.
# MAGIC         
# MAGIC         Example query: "What is Databricks Delta Live Tables?
# MAGIC         Example keywords: databricks,delta,live,table
# MAGIC
# MAGIC         Query: {query}
# MAGIC
# MAGIC         Respond only with the keywords and nothing else.
# MAGIC         """
# MAGIC         model_response = self.chat_completion(
# MAGIC             {"messages": [{"role": "user", "content": prompt}]}, tools=False
# MAGIC         )
# MAGIC         return [keyword.strip() for keyword in model_response["choices"][0]["message"]["content"].split(",")]
# MAGIC
# MAGIC     # Simple keyword-based retriever - would be replaced with a Vector Index
# MAGIC     @mlflow.trace(span_type="RETRIEVER")
# MAGIC     def retrieve_documents(self, keywords: list[str]) -> list[dict]:
# MAGIC         if len(keywords) == 0:
# MAGIC             return []
# MAGIC         result = []
# MAGIC         for chunk in self.docs:
# MAGIC             score = sum(
# MAGIC                 (keyword.lower() in chunk["chunked_text"].lower())
# MAGIC                 for keyword in keywords
# MAGIC             )
# MAGIC             result.append(
# MAGIC                 {
# MAGIC                     "page_content": chunk["chunked_text"],
# MAGIC                     "metadata": {
# MAGIC                         "doc_uri": chunk["url"],
# MAGIC                         "score": score,
# MAGIC                         "chunk_id": chunk["chunk_id"],
# MAGIC                     },
# MAGIC                 }
# MAGIC             )
# MAGIC         ranked_docs = sorted(result, key=lambda x: x["metadata"]["score"], reverse=True)
# MAGIC         cutoff_docs = []
# MAGIC         context_budget_left = self.config.get("max_context_chars")
# MAGIC         for doc in ranked_docs:
# MAGIC             content = doc["page_content"]
# MAGIC             doc_len = len(content)
# MAGIC             if context_budget_left < doc_len:
# MAGIC                 cutoff_docs.append(
# MAGIC                     {**doc, "page_content": content[:context_budget_left]}
# MAGIC                 )
# MAGIC                 break
# MAGIC             else:
# MAGIC                 cutoff_docs.append(doc)
# MAGIC             context_budget_left -= doc_len
# MAGIC         return cutoff_docs
# MAGIC
# MAGIC     # Create a shared function to call the LLM
# MAGIC     def chat_completion(self, request: ChatCompletionRequest, tools: bool = False, return_dict: bool = True) -> ChatCompletionResponse:
# MAGIC         """
# MAGIC         Call the LLM configured via the ModelConfig using the OpenAI SDK
# MAGIC         """
# MAGIC         traced_chat_completions_create_fn = mlflow.trace(
# MAGIC             self.model_serving_client.chat.completions.create,
# MAGIC             name="chat_completions_api",
# MAGIC             span_type="CHAT_MODEL",
# MAGIC         )
# MAGIC         request = {**request, "temperature": self.config.get("temperature"), "max_tokens": self.config.get("max_tokens")}
# MAGIC         if tools:
# MAGIC             request = {**request, "tools": RETRIEVER_TOOL_SPEC, "parallel_tool_calls":False}
# MAGIC         result = traced_chat_completions_create_fn(
# MAGIC             model=self.config.get("endpoint_name"), **request,
# MAGIC                 
# MAGIC         )
# MAGIC         if return_dict:
# MAGIC             return result.to_dict()
# MAGIC         else:
# MAGIC             return result
# MAGIC
# MAGIC     @mlflow.trace(span_type="CHAIN")
# MAGIC     def recursively_call_and_run_tools(self, max_iter=10, **kwargs):
# MAGIC         messages = kwargs["messages"]
# MAGIC         del kwargs["messages"]
# MAGIC         i = 0
# MAGIC         while i < max_iter:
# MAGIC             with mlflow.start_span(name=f"iteration_{i}", span_type="CHAIN") as span:
# MAGIC                 response = self.chat_completion(request={'messages': messages}, tools=True, return_dict=False)
# MAGIC                 assistant_message = response.choices[0].message  # openai client
# MAGIC                 tool_calls = assistant_message.tool_calls  # openai
# MAGIC                 if tool_calls is None:
# MAGIC                     # the tool execution finished, and we have a generation
# MAGIC                     # messages.append(assistant_message.to_dict())
# MAGIC                     # return messages
# MAGIC                     return response.to_dict()
# MAGIC                 tool_messages = []
# MAGIC                 for tool_call in tool_calls:  # TODO: should run in parallel
# MAGIC                     with mlflow.start_span(
# MAGIC                         name="execute_tool", span_type="TOOL"
# MAGIC                     ) as span:
# MAGIC                         function = tool_call.function  # openai
# MAGIC                         args = json.loads(function.arguments)  # openai
# MAGIC                         span.set_inputs(
# MAGIC                             {
# MAGIC                                 "function_name": function.name,
# MAGIC                                 "function_args_raw": function.arguments,
# MAGIC                                 "function_args_loaded": args,
# MAGIC                             }
# MAGIC                         )
# MAGIC                         result = self.execute_function(
# MAGIC                             self.tool_functions[function.name], args
# MAGIC                         )
# MAGIC                         tool_message = {
# MAGIC                             "role": "tool",
# MAGIC                             "tool_call_id": tool_call.id,
# MAGIC                             "content": result,
# MAGIC                         }  # openai
# MAGIC
# MAGIC                         tool_messages.append(tool_message)
# MAGIC                         span.set_outputs({"new_message": tool_message})
# MAGIC                 assistant_message_dict = assistant_message.dict().copy()  # openai
# MAGIC                 del assistant_message_dict["content"]
# MAGIC                 del assistant_message_dict["function_call"]  # openai only
# MAGIC                 if "audio" in assistant_message_dict:
# MAGIC                     del assistant_message_dict["audio"]  # llama70b hack
# MAGIC                 messages = (
# MAGIC                     messages
# MAGIC                     + [
# MAGIC                         assistant_message_dict,
# MAGIC                     ]
# MAGIC                     + tool_messages
# MAGIC                 )
# MAGIC                 i += 1
# MAGIC         # TODO: Handle more gracefully
# MAGIC         raise "ERROR: max iter reached"
# MAGIC
# MAGIC     def execute_function(self, tool, args):
# MAGIC         result = tool(**args)
# MAGIC         return json.dumps(result)
# MAGIC
# MAGIC     # Helpers
# MAGIC     @mlflow.trace(span_type="PARSER")
# MAGIC     def get_request_dict(self, 
# MAGIC         model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame, List]
# MAGIC     ) -> List[Dict[str, str]]:
# MAGIC         if type(model_input) == list:
# MAGIC             # get the first row
# MAGIC             model_input = list[0]
# MAGIC         elif type(model_input) == pd.DataFrame:
# MAGIC             # return the first row, this model doesn't support batch input
# MAGIC             return model_input.to_dict(orient="records")[0]
# MAGIC         
# MAGIC         # now, try to unpack the single item or first row of batch input
# MAGIC         if type(model_input) == ChatCompletionRequest:
# MAGIC             return asdict(model_input)
# MAGIC         elif type(model_input) == dict:
# MAGIC             return model_input
# MAGIC         
# MAGIC
# MAGIC
# MAGIC # tell MLflow logging where to find the agent's code
# MAGIC mlflow.models.set_model(FCAgent())
# MAGIC
# MAGIC

# COMMAND ----------

from fc_agent import FCAgent
fc_agent = FCAgent()
# rag_agent.load_context() # load the retriever's docs

response = fc_agent.predict(model_input={"messages": [{"role": "user", "content": "What is lakehouse monitoring?"}]})

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import ChatCompletionResponse, ChatCompletionRequest

with mlflow.start_run(run_name="fc_agent"):
    # Log the Agent's code/config to MLflow
    model_info = mlflow.pyfunc.log_model(
        python_model="fc_agent.py",
        artifact_path="agent",
        model_config={
    'endpoint_name': "agents-demo-gpt4o",
    'temperature': 0.01,
    'max_tokens': 1000,
    'system_prompt': """The Agent is a RAG chatbot that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.

    You are a helpful assistant that answers questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request.
    """,
    'max_context_chars': 4096 * 4
},
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=ChatCompletionResponse(),
        ),
        pip_requirements=["databricks-sdk[openai]", "mlflow", "databricks-agents"],
    )

    eval_results = mlflow.evaluate(
        data=evals,
        model=model_info.model_uri,
        model_type="databricks-agent",
    )

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# from fc_agent import FCAgent



# COMMAND ----------

# MAGIC %%writefile rag_agent.py
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from openai import OpenAI
# MAGIC import pandas as pd
# MAGIC from typing import Any, Union, Dict, List
# MAGIC import mlflow
# MAGIC from mlflow.models.rag_signatures import ChatCompletionResponse, ChatCompletionRequest
# MAGIC import dataclasses
# MAGIC
# MAGIC DEFAULT_CONFIG = {
# MAGIC     'endpoint_name': "agents-demo-gpt4o",
# MAGIC     'temperature': 0.01,
# MAGIC     'max_tokens': 1000,
# MAGIC     'system_prompt': """The Agent is a RAG chatbot that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.
# MAGIC
# MAGIC     Use the following context to answer the user's query:
# MAGIC     {context}
# MAGIC     """,
# MAGIC     'max_context_chars': 4096 * 4
# MAGIC }
# MAGIC
# MAGIC DOCS_ARTIFACT_KEY = "docs_data"
# MAGIC
# MAGIC class RAGAgent(mlflow.pyfunc.PythonModel):
# MAGIC     """
# MAGIC     Class representing an Agent that does simple RAG using keyword-based search.
# MAGIC     """
# MAGIC
# MAGIC     def load_context(self, context=None):
# MAGIC         """
# MAGIC         Load the retriever's docs.
# MAGIC         """
# MAGIC         # Load the document data - either from the logged MLflow artifact or the web URL
# MAGIC         # if context is not None:
# MAGIC         #     raw_docs_json = context.artifacts[DOCS_ARTIFACT_KEY]
# MAGIC         #     self.docs = pd.read_json(raw_docs_json, lines=True)[:500].to_dict("records")
# MAGIC         # else:
# MAGIC         # raw_docs_parquet = "https://github.com/databricks/genai-cookbook/raw/refs/heads/main/quick_start_demo/chunked_databricks_docs.snappy.parquet"
# MAGIC         # self.docs = pd.read_parquet(raw_docs_parquet)[:500].to_dict("records")
# MAGIC
# MAGIC         
# MAGIC
# MAGIC     def __init__(self):
# MAGIC         """
# MAGIC         Initialize the OpenAI SDK client connected to Model Serving.
# MAGIC         Load the Agent's configuration from MLflow Model Config.
# MAGIC         """
# MAGIC         self.docs = None
# MAGIC
# MAGIC         # Initialize OpenAI SDK connected to Model Serving
# MAGIC         w = WorkspaceClient()
# MAGIC         self.model_serving_client: OpenAI = w.serving_endpoints.get_open_ai_client()
# MAGIC
# MAGIC         # Load config
# MAGIC         self.config = mlflow.models.ModelConfig(development_config=DEFAULT_CONFIG)
# MAGIC
# MAGIC         # Configure playground & review app & agent evaluation to display / see the chunks from the retriever 
# MAGIC         mlflow.models.set_retriever_schema(
# MAGIC             name="db_docs",
# MAGIC             primary_key="chunk_id",
# MAGIC             text_column="chunked_text",
# MAGIC             doc_uri="doc_uri",
# MAGIC         )
# MAGIC
# MAGIC         raw_docs_parquet = "https://github.com/databricks/genai-cookbook/raw/refs/heads/main/quick_start_demo/chunked_databricks_docs.snappy.parquet"
# MAGIC         self.docs = pd.read_parquet(raw_docs_parquet)[:500].to_dict("records")
# MAGIC
# MAGIC     @mlflow.trace(name="rag_agent", span_type="AGENT")
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         context: Any = None,
# MAGIC         model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
# MAGIC         params: Any = None,
# MAGIC     ) -> ChatCompletionResponse:
# MAGIC         ##############################################################################
# MAGIC         # Extract `messages` key from the `model_input`
# MAGIC         request = self.get_request_dict(model_input)
# MAGIC
# MAGIC         # Do retrieval & augement the prompt
# MAGIC         user_query = request["messages"][-1]["content"]
# MAGIC         keywords = self.extract_keywords(user_query)
# MAGIC         docs = self.retrieve_documents(keywords)
# MAGIC         context = "\n\n".join([doc["page_content"] for doc in docs])
# MAGIC         system_prompt_formatted = self.config.get("system_prompt").format(context=context)
# MAGIC
# MAGIC         # Add system prompt to messages history
# MAGIC         request = self.prepend_system_prompt(request, system_prompt_formatted)
# MAGIC
# MAGIC         # Generate with the LLM
# MAGIC         return self.chat_completion(
# MAGIC             request=request
# MAGIC         )
# MAGIC
# MAGIC     @mlflow.trace(span_type="FUNCTION")
# MAGIC     def prepend_system_prompt(
# MAGIC         self, request: ChatCompletionRequest, system_prompt: str
# MAGIC     ) -> ChatCompletionRequest:
# MAGIC         if isinstance(request, ChatCompletionRequest):
# MAGIC             request = dataclasses.asdict(request)
# MAGIC         if request["messages"][0]["role"] != "system":
# MAGIC             return {
# MAGIC                 **request,
# MAGIC                 "messages": [
# MAGIC                     {"role": "system", "content": system_prompt},
# MAGIC                     *request["messages"],
# MAGIC                 ],
# MAGIC             }
# MAGIC         return request
# MAGIC
# MAGIC     # Helper function for keyword-based retriever - would be replaced by the Vector Index
# MAGIC     @mlflow.trace(span_type="PARSER")
# MAGIC     def extract_keywords(self, query: str) -> list[str]:
# MAGIC         prompt = f"""Given a user query, extract the most salient keywords from the user query. These keywords will be used in a search engine to retrieve relevant documents to the query.
# MAGIC         
# MAGIC         Example query: "What is Databricks Delta Live Tables?
# MAGIC         Example keywords: databricks,delta,live,table
# MAGIC
# MAGIC         Query: {query}
# MAGIC
# MAGIC         Respond only with the keywords and nothing else.
# MAGIC         """
# MAGIC         model_response = self.chat_completion(
# MAGIC             {"messages": [{"role": "user", "content": prompt}]}
# MAGIC         )
# MAGIC         return [keyword.strip() for keyword in model_response["choices"][0]["message"]["content"].split(",")]
# MAGIC
# MAGIC     # Simple keyword-based retriever - would be replaced with a Vector Index
# MAGIC     @mlflow.trace(span_type="RETRIEVER")
# MAGIC     def retrieve_documents(self, keywords: list[str]) -> list[dict]:
# MAGIC         if len(keywords) == 0:
# MAGIC             return []
# MAGIC         result = []
# MAGIC         for chunk in self.docs:
# MAGIC             score = sum(
# MAGIC                 (keyword.lower() in chunk["chunked_text"].lower())
# MAGIC                 for keyword in keywords
# MAGIC             )
# MAGIC             result.append(
# MAGIC                 {
# MAGIC                     "page_content": chunk["chunked_text"],
# MAGIC                     "metadata": {
# MAGIC                         "doc_uri": chunk["url"],
# MAGIC                         "score": score,
# MAGIC                         "chunk_id": chunk["chunk_id"],
# MAGIC                     },
# MAGIC                 }
# MAGIC             )
# MAGIC         ranked_docs = sorted(result, key=lambda x: x["metadata"]["score"], reverse=True)
# MAGIC         cutoff_docs = []
# MAGIC         context_budget_left = self.config.get("max_context_chars")
# MAGIC         for doc in ranked_docs:
# MAGIC             content = doc["page_content"]
# MAGIC             doc_len = len(content)
# MAGIC             if context_budget_left < doc_len:
# MAGIC                 cutoff_docs.append(
# MAGIC                     {**doc, "page_content": content[:context_budget_left]}
# MAGIC                 )
# MAGIC                 break
# MAGIC             else:
# MAGIC                 cutoff_docs.append(doc)
# MAGIC             context_budget_left -= doc_len
# MAGIC         return cutoff_docs
# MAGIC
# MAGIC     # Create a shared function to call the LLM
# MAGIC     def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
# MAGIC         """
# MAGIC         Call the LLM configured via the ModelConfig using the OpenAI SDK
# MAGIC         """
# MAGIC         traced_chat_completions_create_fn = mlflow.trace(
# MAGIC             self.model_serving_client.chat.completions.create,
# MAGIC             name="chat_completions_api",
# MAGIC             span_type="CHAT_MODEL",
# MAGIC         )
# MAGIC         request = {**request, "temperature": self.config.get("temperature"), "max_tokens": self.config.get("max_tokens")}
# MAGIC         return traced_chat_completions_create_fn(
# MAGIC             model=self.config.get("endpoint_name"), **request
# MAGIC         ).to_dict()
# MAGIC
# MAGIC     # Helpers
# MAGIC     @mlflow.trace(span_type="PARSER")
# MAGIC     def get_request_dict(self, 
# MAGIC         model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame, List]
# MAGIC     ) -> List[Dict[str, str]]:
# MAGIC         if type(model_input) == list:
# MAGIC             # get the first row
# MAGIC             model_input = list[0]
# MAGIC         elif type(model_input) == pd.DataFrame:
# MAGIC             # return the first row, this model doesn't support batch input
# MAGIC             return model_input.to_dict(orient="records")[0]
# MAGIC         
# MAGIC         # now, try to unpack the single item or first row of batch input
# MAGIC         if type(model_input) == ChatCompletionRequest:
# MAGIC             return asdict(model_input)
# MAGIC         elif type(model_input) == dict:
# MAGIC             return model_input
# MAGIC         
# MAGIC
# MAGIC
# MAGIC # tell MLflow logging where to find the agent's code
# MAGIC mlflow.models.set_model(RAGAgent())

# COMMAND ----------

from rag_agent import RAGAgent

rag_agent = RAGAgent()
# rag_agent.load_context() # load the retriever's docs

response = rag_agent.predict(model_input={"messages": [{"role": "user", "content": "What is Databricks?"}]})

# COMMAND ----------

# PROMPT = """Given the following context
#   {context}
#   ###############
#   Answer the following query to the best of your knowledge:
#   {user_query}
# """
# CONTEXT_LEN_CHARS = 4096 * 4

# @mlflow.trace(name="chain", span_type="CHAIN")
# def rag_agent(request: ChatCompletionRequest) -> ChatCompletionResponse:
#     # Add system prompt to messages history
#     request = prepend_system_prompt(request, TASK_DESCRIPTION)
#     # Do retrieval & augement the prompt
#     user_query = request["messages"][-1]["content"]
#     keywords = extract_keywords(user_query)
#     docs = retrieve_documents(keywords)
#     context = "\n\n".join([doc["page_content"] for doc in docs])
#     agent_query = PROMPT.format(context=context, user_query=user_query)
#     # Generate with the LLM
#     return chat_completion(request={
#         **request,
#         "messages": [
#             *request["messages"][:-1], # Keep the chat history.
#             {"role": "user", "content": agent_query}
#         ]
#     })

# # Helper function for keyword-based retriever - would be replaced by the Vector Index
# @mlflow.trace(span_type="PARSER")
# def extract_keywords(query: str) -> list[str]:
#     prompt = f"""Given a user query, extract the most salient keywords from the user query. These keywords will be used in a search engine to retrieve relevant documents to the query.
    
#     Example query: "What is Databricks Delta Live Tables?
#     Example keywords: databricks,delta,live,table

#     Query: {query}

#     Respond only with the keywords and nothing else.
#     """
#     model_response = chat_completion({
#         "messages": [{"role": "user", "content": prompt}]
#     })
#     return [keyword.strip() for keyword in model_response["choices"][0]["message"]["content"].split(",")]

# # Simple keyword-based retriever - would be replaced with a Vector Index
# @mlflow.trace(span_type="RETRIEVER")
# def retrieve_documents(keywords: list[str]) -> list[dict]:
#     if len(keywords) == 0:
#         return []
#     result = []
#     for chunk in CHUNKS:
#         score = sum(
#             (keyword.lower() in chunk["chunked_text"].lower()) for keyword in keywords
#         )
#         result.append({
#             "page_content": chunk["chunked_text"],
#             "metadata": {
#                 "doc_uri": chunk["url"],
#                 "score": score,
#                 "chunk_id": chunk["chunk_id"],
#             },
#         })
#     ranked_docs = sorted(result, key=lambda x: x["metadata"]["score"], reverse=True)
#     cutoff_docs = []
#     context_budget_left = CONTEXT_LEN_CHARS
#     for doc in ranked_docs:
#         content = doc["page_content"]
#         doc_len = len(content)
#         if context_budget_left < doc_len:
#             cutoff_docs.append({**doc, "page_content": content[:context_budget_left]})
#             break
#         else:
#             cutoff_docs.append(doc)
#         context_budget_left -= doc_len
#     return cutoff_docs

# response = rag_agent({"messages": [{"role": "user", "content": "What is Databricks?"}]})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2/ Evaluate the RAG agent

# COMMAND ----------

# MAGIC %md
# MAGIC Let's run Agent Evaluation on the RAG Agent. 
# MAGIC
# MAGIC In the MLFLow Evaluations UI, open the `agent_with_retriever` run and use the "Compare to Run" dropdown  to select `agent_without_retriever`.
# MAGIC
# MAGIC This comparison view helps you quickly identify where the Agent got better/worse/stayed the same.

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import ChatCompletionResponse, ChatCompletionRequest

with mlflow.start_run(run_name="agent_with_retriever"):
    # Log the Agent's code/config to MLflow
    model_info = mlflow.pyfunc.log_model(
        python_model="rag_agent.py",
        artifact_path="agent",
        model_config={
            "endpoint_name": "agents-demo-gpt4o",
            "temperature": 0.01,
            "max_tokens": 1000,
            "system_prompt": """The Agent is a RAG chatbot that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.

    Use the following context to answer the user's query:
    {context}
    """,
            "max_context_chars": 4096 * 4,
        },
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=ChatCompletionResponse(),
        ),
        pip_requirements=["databricks-sdk[openai]", "mlflow", "databricks-agents"],
    )

    eval_results = mlflow.evaluate(
        data=evals,
        model=model_info.model_uri,
        model_type="databricks-agent",
    )

# COMMAND ----------

If you return to the MLflow Experiment from the Run, you can use the UI to compare quality/cost/latency metrics between experiements.  This helps you make informed tradeoffs in partnership with your business stakeholders about cost/latency/quality.  Further, you can use this view to provide quantitative updates to your stakeholders so they can follow your progress improving quality!


# COMMAND ----------

# MAGIC %md
# MAGIC # 4/ [Optional] Deploy the RAG Agent

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1/ Create a production-ready version of the Agent as a MLflow PyFunc model
# MAGIC
# MAGIC Here, we turn the prototype Agent into a PyFunc class that can be logged to MLFlow & deployed as a production REST API.  We write this code to a file called `rag_agent.py` so we can use [MLflow Models from Code](https://www.mlflow.org/blog/models_from_code) to log the Agent.

# COMMAND ----------

# MAGIC %%writefile rag_agent.py
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from openai import OpenAI
# MAGIC import pandas as pd
# MAGIC from typing import Any, Union, Dict, List
# MAGIC import mlflow
# MAGIC from mlflow.models.rag_signatures import ChatCompletionResponse, ChatCompletionRequest
# MAGIC import dataclasses
# MAGIC
# MAGIC # Constants
# MAGIC ENDPOINT_NAME = "agents-demo-gpt4o"
# MAGIC TEMPERATURE = 0.01
# MAGIC MAX_TOKENS = 1000
# MAGIC TASK_DESCRIPTION = """The Agent is a RAG chatbot that answers questions about Databricks. Questions unrelated to Databricks are irrelevant."""
# MAGIC CONTEXT_LEN_CHARS = 4096 * 4
# MAGIC PROMPT = """Given the following context
# MAGIC {context}
# MAGIC ###############
# MAGIC Answer the following query to the best of your knowledge:
# MAGIC {user_query}
# MAGIC """
# MAGIC DOCS_ARTIFACT_KEY = "docs_data"
# MAGIC
# MAGIC class RAGAgent(mlflow.pyfunc.PythonModel):
# MAGIC     """
# MAGIC     Class representing an Agent that does simple RAG using keyword-based search.
# MAGIC     """
# MAGIC
# MAGIC     def load_context(self, context=None):
# MAGIC         """
# MAGIC         Load the model from the specified artifacts directory.
# MAGIC         """
# MAGIC         # Initialize the OpenAI SDK client connected to Model Serving
# MAGIC         w = WorkspaceClient()
# MAGIC         self.model_serving_client: OpenAI = w.serving_endpoints.get_open_ai_client()
# MAGIC
# MAGIC         # Load the document data -  ex the MLflow artifacts
# MAGIC         if context is not None:
# MAGIC             raw_docs_json = context.artifacts[DOCS_ARTIFACT_KEY]
# MAGIC             self.docs = pd.read_json(raw_docs_json, lines=True)[:500].to_dict("records")
# MAGIC         else:
# MAGIC             raw_docs_parquet = "https://github.com/databricks/genai-cookbook/raw/refs/heads/main/quick_start_demo/chunked_databricks_docs.snappy.parquet"
# MAGIC             self.docs = pd.read_parquet(raw_docs_parquet)[:500].to_dict("records")
# MAGIC
# MAGIC         # Configure playground & review app & agent evaluation to display / see the chunks from the retriever 
# MAGIC         mlflow.models.set_retriever_schema(
# MAGIC             name="db_docs",
# MAGIC             primary_key="chunk_id",
# MAGIC             text_column="chunked_text",
# MAGIC             doc_uri="doc_uri",
# MAGIC         )
# MAGIC
# MAGIC     def __init__(self):
# MAGIC         self.docs = None
# MAGIC         self.model_serving_client = None
# MAGIC
# MAGIC     @mlflow.trace(name="rag_agent", span_type="AGENT")
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         context: Any = None,
# MAGIC         model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
# MAGIC         params: Any = None,
# MAGIC     ) -> ChatCompletionResponse:
# MAGIC         ##############################################################################
# MAGIC         # Extract `messages` key from the `model_input`
# MAGIC         request = self.get_request_dict(model_input)
# MAGIC
# MAGIC         # Add system prompt to messages history
# MAGIC         request = self.prepend_system_prompt(request, TASK_DESCRIPTION)
# MAGIC
# MAGIC         # Do retrieval & augement the prompt
# MAGIC         user_query = request["messages"][-1]["content"]
# MAGIC         keywords = self.extract_keywords(user_query)
# MAGIC         docs = self.retrieve_documents(keywords)
# MAGIC         context = "\n\n".join([doc["page_content"] for doc in docs])
# MAGIC         agent_query = PROMPT.format(context=context, user_query=user_query)
# MAGIC
# MAGIC         # Generate with the LLM
# MAGIC         return self.chat_completion(
# MAGIC             request={
# MAGIC                 **request,
# MAGIC                 "messages": [
# MAGIC                     *request["messages"][:-1],  # Keep the chat history.
# MAGIC                     {"role": "user", "content": agent_query},
# MAGIC                 ],
# MAGIC             }
# MAGIC         )
# MAGIC
# MAGIC     @mlflow.trace(span_type="FUNCTION")
# MAGIC     def prepend_system_prompt(
# MAGIC         self, request: ChatCompletionRequest, system_prompt: str
# MAGIC     ) -> ChatCompletionRequest:
# MAGIC         if isinstance(request, ChatCompletionRequest):
# MAGIC             request = dataclasses.asdict(request)
# MAGIC         if request["messages"][0]["role"] != "system":
# MAGIC             return {
# MAGIC                 **request,
# MAGIC                 "messages": [
# MAGIC                     {"role": "system", "content": system_prompt},
# MAGIC                     *request["messages"],
# MAGIC                 ],
# MAGIC             }
# MAGIC         return request
# MAGIC
# MAGIC     # Helper function for keyword-based retriever - would be replaced by the Vector Index
# MAGIC     @mlflow.trace(span_type="PARSER")
# MAGIC     def extract_keywords(self, query: str) -> list[str]:
# MAGIC         prompt = f"""Given a user query, extract the most salient keywords from the user query. These keywords will be used in a search engine to retrieve relevant documents to the query.
# MAGIC         
# MAGIC         Example query: "What is Databricks Delta Live Tables?
# MAGIC         Example keywords: databricks,delta,live,table
# MAGIC
# MAGIC         Query: {query}
# MAGIC
# MAGIC         Respond only with the keywords and nothing else.
# MAGIC         """
# MAGIC         model_response = self.chat_completion(
# MAGIC             {"messages": [{"role": "user", "content": prompt}]}
# MAGIC         )
# MAGIC         return [keyword.strip() for keyword in model_response["choices"][0]["message"]["content"].split(",")]
# MAGIC
# MAGIC     # Simple keyword-based retriever - would be replaced with a Vector Index
# MAGIC     @mlflow.trace(span_type="RETRIEVER")
# MAGIC     def retrieve_documents(self, keywords: list[str]) -> list[dict]:
# MAGIC         if len(keywords) == 0:
# MAGIC             return []
# MAGIC         result = []
# MAGIC         for chunk in self.docs:
# MAGIC             score = sum(
# MAGIC                 (keyword.lower() in chunk["chunked_text"].lower())
# MAGIC                 for keyword in keywords
# MAGIC             )
# MAGIC             result.append(
# MAGIC                 {
# MAGIC                     "page_content": chunk["chunked_text"],
# MAGIC                     "metadata": {
# MAGIC                         "doc_uri": chunk["url"],
# MAGIC                         "score": score,
# MAGIC                         "chunk_id": chunk["chunk_id"],
# MAGIC                     },
# MAGIC                 }
# MAGIC             )
# MAGIC         ranked_docs = sorted(result, key=lambda x: x["metadata"]["score"], reverse=True)
# MAGIC         cutoff_docs = []
# MAGIC         context_budget_left = CONTEXT_LEN_CHARS
# MAGIC         for doc in ranked_docs:
# MAGIC             content = doc["page_content"]
# MAGIC             doc_len = len(content)
# MAGIC             if context_budget_left < doc_len:
# MAGIC                 cutoff_docs.append(
# MAGIC                     {**doc, "page_content": content[:context_budget_left]}
# MAGIC                 )
# MAGIC                 break
# MAGIC             else:
# MAGIC                 cutoff_docs.append(doc)
# MAGIC             context_budget_left -= doc_len
# MAGIC         return cutoff_docs
# MAGIC
# MAGIC     # Create a shared function to call the LLM
# MAGIC     def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
# MAGIC         traced_chat_completions_create_fn = mlflow.trace(
# MAGIC             self.model_serving_client.chat.completions.create,
# MAGIC             name="chat_completions_api",
# MAGIC             span_type="CHAT_MODEL",
# MAGIC         )
# MAGIC         request = {**request, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
# MAGIC         return traced_chat_completions_create_fn(
# MAGIC             model=ENDPOINT_NAME, **request
# MAGIC         ).to_dict()
# MAGIC
# MAGIC     # Helpers
# MAGIC     @mlflow.trace(span_type="PARSER")
# MAGIC     def get_request_dict(self, 
# MAGIC         model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame, List]
# MAGIC     ) -> List[Dict[str, str]]:
# MAGIC         if type(model_input) == list:
# MAGIC             # get the first row
# MAGIC             model_input = list[0]
# MAGIC         elif type(model_input) == pd.DataFrame:
# MAGIC             # return the first row, this model doesn't support batch input
# MAGIC             return model_input.to_dict(orient="records")[0]
# MAGIC         
# MAGIC         # now, try to unpack the single item or first row of batch input
# MAGIC         if type(model_input) == ChatCompletionRequest:
# MAGIC             return asdict(model_input)
# MAGIC         elif type(model_input) == dict:
# MAGIC             return model_input
# MAGIC         
# MAGIC
# MAGIC
# MAGIC # tell MLflow logging where to find the agent's code
# MAGIC mlflow.models.set_model(RAGAgent())

# COMMAND ----------

# MAGIC %md
# MAGIC Empty `__init__.py` to allow the RAGAgent to be imported.

# COMMAND ----------

# MAGIC %%writefile __init__.py
# MAGIC
# MAGIC # Empty file

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2/ Log the Agent to MLflow

# COMMAND ----------

# MAGIC %md
# MAGIC First, check that the Agent works.

# COMMAND ----------

# Import from rag_agent.py
from rag_agent import RAGAgent

# Load the Agent
agent = RAGAgent()
agent.load_context(context=None)

output = agent.predict(model_input={"messages": [{"role": "user", "content": "What is Databricks?"}]})

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import ChatCompletionResponse, ChatCompletionRequest
from mlflow.models.resources import DatabricksServingEndpoint
import rag_agent
import mlflow
import json

release_config = {
    'endpoint_name': "agents-demo-gpt4o",
    'temperature': 0.01,
    'max_tokens': 1000,
    'system_prompt': """The Agent is a RAG chatbot that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.

    Use the following context to answer the user's query:
    {context}
    """,
    'max_context_chars': 4096 * 4
}

# Annotate the logged model with the required Databricks resources
resources = [DatabricksServingEndpoint(endpoint_name=release_config['endpoint_name'])]

# Input example provides an example input for the model serving endpoint
# input_example = {"messages": [{"content": "What is Databricks?", "role": "user"}]}

# Export the documents to save in the MLlfow model
# chunks_dict = CHUNKS
# jsonl_file_name = "./chunks.jsonl"

# with open(jsonl_file_name, 'w') as jsonl_file:
#     for chunk in CHUNKS:
#         jsonl_file.write(json.dumps(chunk) + '\n')

with mlflow.start_run(run_name="release_candidate_1"):
    # Log the Agent
    model_info = mlflow.pyfunc.log_model(
        python_model="rag_agent.py",
        artifact_path="agent",
        model_config=release_config,
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=ChatCompletionResponse(),
        ),
        resources=resources,
        # artifacts={rag_agent.DOCS_ARTIFACT_KEY: jsonl_file_name},
        # input_example=input_example,
        pip_requirements=["databricks-sdk[openai]", "mlflow", "databricks-agents"]
    )

    # Store an evaluation of the logged Agent
    # eval_results = mlflow.evaluate(
    #     data=evals,
    #     model=model_info.model_uri,
    #     model_type="databricks-agent",
    # )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2/ Deploy to Model Serving

# COMMAND ----------

from databricks import agents
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Grant the necessary privileges to the user
mlflow.set_registry_uri("databricks-uc")

# Register the chain to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=model_info.model_uri, name=UC_MODEL_NAME
)
# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(
    model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version
)

# Wait for the Review App to be ready
print("Wait for endpoint to deploy.  This can take 15 - 20 minutes.")
print(f"Endpoint name: {deployment_info.endpoint_name}")
