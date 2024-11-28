# Databricks notebook source
# MAGIC %pip install databricks-agents pandas mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks.agents.eval import generate_evals_df
import pandas as pd

# These documents can be a Pandas DataFrame or a Spark DataFrame with two columns: 'content' and 'doc_uri'.
docs = pd.DataFrame.from_records(
    [
      {
        'content': f"""
            Apache Spark is a unified analytics engine for large-scale data processing. It provides high-level APIs in Java,
            Scala, Python and R, and an optimized engine that supports general execution graphs. It also supports a rich set
            of higher-level tools including Spark SQL for SQL and structured data processing, pandas API on Spark for pandas
            workloads, MLlib for machine learning, GraphX for graph processing, and Structured Streaming for incremental
            computation and stream processing.
        """,
        'doc_uri': 'https://spark.apache.org/docs/3.5.2/'
      },
      {
        'content': f"""
            Spark’s primary abstraction is a distributed collection of items called a Dataset. Datasets can be created from Hadoop InputFormats (such as HDFS files) or by transforming other Datasets. Due to Python’s dynamic nature, we don’t need the Dataset to be strongly-typed in Python. As a result, all Datasets in Python are Dataset[Row], and we call it DataFrame to be consistent with the data frame concept in Pandas and R.""",
        'doc_uri': 'https://spark.apache.org/docs/3.5.2/quick-start.html'
      }
    ]
)

# NOTE: The guidelines you provide are a free-form string. The markdown string below is the suggested formatting for the set of guidelines, however you are free
# to add your sections here. Note that this will be prompt-engineering an LLM that generates the synthetic data, so you may have to iterate on these guidelines before
# you get the results you desire.
guidelines = """
# Task Description
The Agent is a RAG chatbot that answers questions about using Spark on Databricks. The Agent has access to a corpus of Databricks documents, and its task is to answer the user's questions by retrieving the relevant docs from the corpus and synthesizing a helpful, accurate response. The corpus covers a lot of info, but the Agent is specifically designed to interact with Databricks users who have questions about Spark. So questions outside of this scope are considered irrelevant.

# User personas
- A developer who is new to the Databricks platform
- An experienced, highly technical Data Scientist or Data Engineer

# Example questions
- what API lets me parallelize operations over rows of a delta table?
- Which cluster settings will give me the best performance when using Spark?

# Additional Guidelines
- Questions should be succinct, and human-like
"""

evals = generate_evals_df(
    docs,
    # The number of evaluations to generate for each doc.
    num_questions_per_doc=2,
    # A set of guidelines that help guide the synthetic generation. This is a free-form string that will be used to prompt the generation.
    guidelines=guidelines
)

display(evals)

# Define the agent as a function that calls the model serving endpoint for the Llama 3.1 model.
def llama3_agent(input):
  client = mlflow.deployments.get_deploy_client("databricks")
  input.pop("databricks_options", None)
  return client.predict(endpoint="databricks-meta-llama-3-1-405b-instruct", inputs=input)

# Evaluate the model using the newly generated evaluation set. After the function call completes, click the UI link to see the results. You can use this as a baseline for your agent.
results = mlflow.evaluate(
  model=llama3_agent,
  data=evals,
  model_type="databricks-agent"
)

