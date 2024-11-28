# Databricks notebook source
# MAGIC %md
# MAGIC # 1/ Setup

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow-skinny databricks-sdk[openai]
# MAGIC dbutils.library.restartPython()

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

import pandas as pd
databricks_docs_url = "https://github.com/databricks/genai-cookbook/raw/refs/heads/main/quick_start_demo/chunked_databricks_docs.snappy.parquet"
CHUNKS = pd.read_parquet(databricks_docs_url)[:500].to_dict('records')
display(CHUNKS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2/ Generate synthetic data

# COMMAND ----------

# Use the synthetic eval generation API to get some evals
from databricks.agents.eval import generate_evals_df

TASK_DESCRIPTION= """
  The Agent is a RAG chatbot that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.
"""

# YOUR DOCS GO HERE. Pandas/spark Dataframes with columns `content STRING, doc_uri STRING` are suitable.
chunks_spark_df = spark.createDataFrame(CHUNKS)
docs = chunks_spark_df.select(chunks_spark_df.chunked_text.alias("content"), chunks_spark_df.chunk_id.alias("doc_uri"))

# "Ghost text" for guidelines - feel free to modify as you see fit.
guidelines = f"""
# Task Description
{TASK_DESCRIPTION}

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
    docs,
    num_evals=num_evals,
    guidelines=guidelines
)
display(evals)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2/ Base agent (no retriever)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1/ Base Llama 3 agent

# COMMAND ----------

import mlflow
from mlflow.models.rag_signatures import ChatCompletionResponse, ChatCompletionRequest
import dataclasses
from databricks.sdk import WorkspaceClient

# ENDPOINT_NAME = "databricks-meta-llama-3-1-70b-instruct"
ENDPOINT_NAME = "agents-demo-gpt4o"
TEMPERATURE = 0.01
MAX_TOKENS = 1000

# Get OpenAI SDK connected to Databricks Model Serving / AI Gateway
w = WorkspaceClient()
model_serving_client = w.serving_endpoints.get_open_ai_client()


# Create a shared function to call the LLM
def chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    traced_chat_completions_create_fn = mlflow.trace(
        model_serving_client.chat.completions.create,
        name="chat_completions_api",
        span_type="CHAT_MODEL",
    )
    request = {**request, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
    return traced_chat_completions_create_fn(model=ENDPOINT_NAME, **request).to_dict()


@mlflow.trace(span_type="FUNCTION")
def prepend_system_prompt(
    request: ChatCompletionRequest, system_prompt: str
) -> ChatCompletionRequest:
    if isinstance(request, ChatCompletionRequest):
        request = dataclasses.asdict(request)
    if request["messages"][0]["role"] != "system":
        return {
            **request,
            "messages": [
                {"role": "system", "content": system_prompt},
                *request["messages"],
            ],
        }
    return request


# Define the agent as a function that calls the model serving endpoint for the Llama 3.1 model.
@mlflow.trace(name="chain", span_type="CHAIN")
def llm_only_agent(request: ChatCompletionRequest) -> ChatCompletionResponse:
    request = prepend_system_prompt(request, TASK_DESCRIPTION)
    return chat_completion(request)


response = llm_only_agent(
    {"messages": [{"role": "user", "content": "What is Databricks?"}]}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2/ Evaluate the Base agent

# COMMAND ----------

with mlflow.start_run(run_name="agent_without_retriever"):
    eval_results = mlflow.evaluate(
        data=evals, # Your evaluation set
        model=llm_only_agent,
        model_type="databricks-agent", # activate Mosaic AI Agent Evaluation
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # 3/ RAG agent with keyword-based retriever

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1/ Custom RAG Agent

# COMMAND ----------

# MAGIC %md
# MAGIC Helper functions for calling the LLM

# COMMAND ----------

import mlflow
from mlflow.models.rag_signatures import ChatCompletionResponse, ChatCompletionRequest
import dataclasses
from databricks.sdk import WorkspaceClient

# Get OpenAI SDK connected to Databricks Model Serving / AI Gateway
w = WorkspaceClient()
model_serving_client = w.serving_endpoints.get_open_ai_client()

TEMPERATURE = 0.01
MAX_TOKENS = 1000

# Create a shared function to call the LLM
def chat_completion(request: ChatCompletionRequest, endpoint_name: str) -> ChatCompletionResponse:
    traced_chat_completions_create_fn = mlflow.trace(
        model_serving_client.chat.completions.create,
        name="chat_completions_api",
        span_type="CHAT_MODEL",
    )
    request = {**request, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
    return traced_chat_completions_create_fn(model=ENDPOINT_NAME, **request).to_dict()


@mlflow.trace(span_type="FUNCTION")
def prepend_system_prompt(
    request: ChatCompletionRequest, system_prompt: str
) -> ChatCompletionRequest:
    if isinstance(request, ChatCompletionRequest):
        request = dataclasses.asdict(request)
    if request["messages"][0]["role"] != "system":
        return {
            **request,
            "messages": [
                {"role": "system", "content": system_prompt},
                *request["messages"],
            ],
        }
    return request



# COMMAND ----------

PROMPT = """Given the following context
  {context}
  ###############
  Answer the following query to the best of your knowledge:
  {user_query}
"""
CONTEXT_LEN_CHARS = 4096 * 4

@mlflow.trace(name="chain", span_type="CHAIN")
def rag_agent(request: ChatCompletionRequest) -> ChatCompletionResponse:
    # Add system prompt to messages history
    request = prepend_system_prompt(request, TASK_DESCRIPTION)
    # Do retrieval & augement the prompt
    user_query = request["messages"][-1]["content"]
    keywords = extract_keywords(user_query)
    docs = retrieve_documents(keywords)
    context = "\n\n".join([doc["page_content"] for doc in docs])
    agent_query = PROMPT.format(context=context, user_query=user_query)
    # Generate with the LLM
    return chat_completion(request={
        **request,
        "messages": [
            *request["messages"][:-1], # Keep the chat history.
            {"role": "user", "content": agent_query}
        ]
    })

# Helper function for keyword-based retriever - would be replaced by the Vector Index
@mlflow.trace(span_type="PARSER")
def extract_keywords(query: str) -> list[str]:
    prompt = f"""Given a user query, extract the most salient keywords from the user query. These keywords will be used in a search engine to retrieve relevant documents to the query.
    
    Example query: "What is Databricks Delta Live Tables?
    Example keywords: databricks,delta,live,table

    Query: {query}

    Respond only with the keywords and nothing else.
    """
    model_response = chat_completion({
        "messages": [{"role": "user", "content": prompt}]
    })
    return [keyword.strip() for keyword in model_response["choices"][0]["message"]["content"].split(",")]

# Simple keyword-based retriever - would be replaced with a Vector Index
@mlflow.trace(span_type="RETRIEVER")
def retrieve_documents(keywords: list[str]) -> list[dict]:
    if len(keywords) == 0:
        return []
    result = []
    for chunk in CHUNKS:
        score = sum(
            (keyword.lower() in chunk["chunked_text"].lower()) for keyword in keywords
        )
        result.append({
            "page_content": chunk["chunked_text"],
            "metadata": {
                "doc_uri": chunk["url"],
                "score": score,
                "chunk_id": chunk["chunk_id"],
            },
        })
    ranked_docs = sorted(result, key=lambda x: x["metadata"]["score"], reverse=True)
    cutoff_docs = []
    context_budget_left = CONTEXT_LEN_CHARS
    for doc in ranked_docs:
        content = doc["page_content"]
        doc_len = len(content)
        if context_budget_left < doc_len:
            cutoff_docs.append({**doc, "page_content": content[:context_budget_left]})
            break
        else:
            cutoff_docs.append(doc)
        context_budget_left -= doc_len
    return cutoff_docs



response = rag_agent({"messages": [{"role": "user", "content": "What is Databricks?"}]})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2/ Evaluate the RAG agent

# COMMAND ----------

with mlflow.start_run(run_name="agent_with_retriever"):
    eval_results = mlflow.evaluate(
        data=evals,
        model=rag_agent,
        model_type="databricks-agent",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3/ Compare the two runs
# MAGIC In the MLFLow Evaluations UI, open the `agent_with_retriever` run and use the "Compare to Run" dropdown  to select `agent_without_retriever`.

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

# Annotate the logged model with the required Databricks resources
resources = [DatabricksServingEndpoint(endpoint_name=rag_agent.ENDPOINT_NAME)]

# Input example provides an example input for the model serving endpoint
input_example = {"messages": [{"content": "What is Databricks?", "role": "user"}]}

# Export the documents to save in the MLlfow model
chunks_dict = CHUNKS
jsonl_file_name = "./chunks.jsonl"

with open(jsonl_file_name, 'w') as jsonl_file:
    for chunk in CHUNKS:
        jsonl_file.write(json.dumps(chunk) + '\n')

with mlflow.start_run(run_name="release_candidate_1"):
    # Log the Agent
    model_info = mlflow.pyfunc.log_model(
        python_model="rag_agent.py",
        artifact_path="agent",
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=ChatCompletionResponse(),
        ),
        resources=resources,
        artifacts={rag_agent.DOCS_ARTIFACT_KEY: jsonl_file_name},
        input_example=input_example,
        pip_requirements=["databricks-sdk[openai]", "mlflow", "databricks-agents"]
    )

    # Store an evaluation of the logged Agent
    eval_results = mlflow.evaluate(
        data=evals,
        model=model_info.model_uri,
        model_type="databricks-agent",
    )

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
