# Databricks notebook source
# MAGIC %md
# MAGIC # 1/ Setup

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow-skinny databricks-sdk databricks-agents
# MAGIC dbutils.library.restartPython()

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
from mlflow.deployments import get_deploy_client
import dataclasses

ENDPOINT_NAME="databricks-meta-llama-3-1-70b-instruct"
TEMPERATURE=0.01
MAX_TOKENS=1000
deploy_client = get_deploy_client("databricks")

def prepend_system_prompt(request: ChatCompletionRequest, system_prompt: str) -> ChatCompletionRequest:
  if isinstance(request, ChatCompletionRequest):
    request = dataclasses.asdict(request)
  if request["messages"][0]["role"] != "system":
    return {
      **request,
      "messages": [
        {"role": "system", "content": system_prompt},
        *request["messages"]
      ]
    }
  return request

@mlflow.trace(name="chat_completion", span_type="CHAT_MODEL")
def chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
  request = {**request, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
  return deploy_client.predict(endpoint=ENDPOINT_NAME, inputs=request)

# Define the agent as a function that calls the model serving endpoint for the Llama 3.1 model.
@mlflow.trace(name="chain", span_type="CHAIN")
def llama3_agent(request: ChatCompletionRequest) -> ChatCompletionResponse:
  request = prepend_system_prompt(request, TASK_DESCRIPTION)
  return chat_completion(request)

response = llama3_agent({"messages": [{"role": "user", "content": "What is Databricks?"}]})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2/ Evaluate the Base agent

# COMMAND ----------

with mlflow.start_run(run_name="agent_without_retriever"):
    eval_results = mlflow.evaluate(
        data=evals, # Your evaluation set
        model=llama3_agent,
        model_type="databricks-agent", # activate Mosaic AI Agent Evaluation
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # 3/ RAG agent with keyword-based retriever

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1/ Custom RAG Agent

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
    request = prepend_system_prompt(request, TASK_DESCRIPTION)
    user_query = request["messages"][-1]["content"]
    keywords = extract_keywords(user_query)
    docs = retrieve_documents(keywords)
    context = "\n\n".join([doc["page_content"] for doc in docs])
    agent_query = PROMPT.format(context=context, user_query=user_query)
    return chat_completion({
        **request,
        "messages": [
            *request["messages"][:-1], # Keep the chat history.
            {"role": "user", "content": agent_query}
        ]
    })

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
    return model_response.choices[0]["message"]["content"].split(",")

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
# MAGIC ## 4.1/ Log the Agent to MLFlow

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.models.resources import DatabricksServingEndpoint

resources = [DatabricksServingEndpoint(endpoint_name=ENDPOINT_NAME)]
input_example = {"messages": [{"content": "What is Databricks?", "role": "user"}]}

with mlflow.start_run(run_name="agent_with_retriever"):
    model_info = mlflow.pyfunc.log_model(
        python_model=rag_agent,
        artifact_path="agent",
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=ChatCompletionResponse(),
        ),
        resources=resources,
        input_example=input_example
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2/ Deploy to Model Serving

# COMMAND ----------

from databricks import agents
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks.sdk import WorkspaceClient

UC_MODEL_NAME = "mosaic_catalog.lilac_schema.nsthorat_agent_rag"

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

# COMMAND ----------

# MAGIC %md
# MAGIC # Share the Review UI with an SME

# COMMAND ----------

# Note: APIs are not yet published to databricks.agents
from databricks.rag_eval.datasets import managed_evals

EVALS_TABLE_NAME = f"mosaic_catalog.managed_evals_pupr_bugbash.nikhil_improve_agent_with_synth_evals_v3"
managed_evals.create_evals_table(evals_table_name=EVALS_TABLE_NAME)

# COMMAND ----------

# Configure the SME UI
managed_evals.update_eval_config(
  evals_table_name=EVALS_TABLE_NAME,  
  agent_name='Databricks Q/A',
  # Optional[str], Model serving endpoint name for your agent (allows SME to interact with agent in SME UI.)
  model_serving_endpoint_name="databricks-meta-llama-3-1-70b-instruct",#'agents_mosaic_catalog-lilac_schema-smilkov_agent_rag', #deployment_info.endpoint_name, #deployment_info.endpoint_name, 
)

# COMMAND ----------

# Import the dataset into managed evals
managed_evals.add_evals(evals_table_name=EVALS_TABLE_NAME, evals=evals)

# COMMAND ----------

# Go to managed evals SME review UI
from mlflow.utils import databricks_utils
host_creds = databricks_utils.get_databricks_host_creds()
api_url = host_creds.host
DEV_URL = f"{api_url}/ml/evals/{EVALS_TABLE_NAME}/dashboard"
SME_URL = f"{api_url}/ml/evals/{EVALS_TABLE_NAME}/review"
displayHTML(
  f'<a href="{DEV_URL}" target="_blank"><button style="color: white; background-color: #0073e6; padding: 10px 24px; cursor: pointer; border: none; border-radius: 4px;">Visit Managed Evals DEV UI</button></a>'
)
displayHTML(
  f'<a href="{SME_URL}" target="_blank"><button style="color: white; background-color: #0073e6; padding: 10px 24px; cursor: pointer; border: none; border-radius: 4px;">Visit Managed Evals SME UI</button></a>'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate with the SME-reviewed evals

# COMMAND ----------

evals = spark.read.table(EVALS_TABLE_NAME).toPandas()
display(evals)

# COMMAND ----------

with mlflow.start_run(run_name="agent_with_retriever_sme"):
    eval_results = mlflow.evaluate(
        data=evals,
        model=rag_agent,
        model_type="databricks-agent",
    )
