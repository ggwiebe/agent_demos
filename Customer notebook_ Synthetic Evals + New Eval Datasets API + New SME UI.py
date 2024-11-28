# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluation Datasets, Synthetic Evaluations, and SME UI Private Preview
# MAGIC
# MAGIC This notebook will walk you through our latest recommended workflow for improving agent quality with synthetic evaluations and SME feedback.
# MAGIC
# MAGIC 1. Create a new evaluation dataset and (optionally) migrate data from existing datasets.
# MAGIC 1. Generate synthetic evaluation questions from a set of documents.
# MAGIC 1. Add the synthetic evaluations to the dataset.
# MAGIC 1. Grant SME permissions to review the evaluations.
# MAGIC 1. Review the synthetic evals in a developer-facing UI and send synthetic evaluations for SME to review in an SME-facing UI.
# MAGIC 1. Export and run `mlflow.evaluate()` on the evals.

# COMMAND ----------

# MAGIC %pip install -U -qqq mlflow mlflow[databricks] "databricks-agents>=0.11.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.agents.evals import datasets as eval_datasets

# TODO: Specify where you want your evaluation dataset to live in Unity Catalog.
# If you have an existing eval dataset, you should still create a new dataset with a new name - we will migrate your data later in the notebook.
EVAL_TABLE = "cat.sch.tbl"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create and manage dataset instances
# MAGIC
# MAGIC #### Create
# MAGIC To create a new evaluation dataset, use `databricks.agents.evals.datasets.create_evals_table`. Only `evals_table_name` is required, other params are optional.
# MAGIC ```python
# MAGIC def create_evals_table(
# MAGIC     evals_table_name: str,
# MAGIC     *,
# MAGIC     agent_name: Optional[str],  # The human-readable name of the agent to display in the UI.
# MAGIC     agent_serving_endpoint: Optional[str],  # The name of the model serving endpoint that serves the agent.
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC #### Update config
# MAGIC To update the configs, use `databricks.agents.evals.datasets.update_eval_labeling_config`. Params that are not specified will not be updated. Set them to `None` to clear them.
# MAGIC ```python
# MAGIC def update_eval_labeling_config(
# MAGIC     evals_table_name: str,
# MAGIC     *,
# MAGIC     agent_name: Optional[str],
# MAGIC     agent_serving_endpoint: Optional[str],
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC #### Get config
# MAGIC To get the configs of an evaluation dataset, use `databricks.agents.evals.datasets.get_eval_labeling_config`.
# MAGIC ```python
# MAGIC def get_eval_labeling_config(
# MAGIC     evals_table_name: str,
# MAGIC ) -> Dict[str, Any]
# MAGIC ```
# MAGIC
# MAGIC #### Delete
# MAGIC To delete a evaluation dataset, use `databricks.agents.evals.datasets.delete_evals_table`. Caution, deleting an evaluation dataset will delete all the data. The operation is not reversible.
# MAGIC ```python
# MAGIC def delete_evals_table(
# MAGIC     evals_table_name: str,
# MAGIC )
# MAGIC ```

# COMMAND ----------

eval_datasets.create_evals_table(
  evals_table_name=EVAL_TABLE,
  # Below params are optional, you can set or update them later with `eval_datasets.update_eval_labeling_config`
  agent_name="My agent name",  # TODO: replace it with your agent name
  agent_serving_endpoint="databricks-meta-llama-3-1-70b-instruct",  # TODO: replace it with your agent serving endpoint
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (Optional) Migrate from an evals dataset created from an older version of this package
# MAGIC
# MAGIC Skip this section if you don't have existing data to migrate from.
# MAGIC
# MAGIC We have made several breaking changes to the dataset format, and the new API is not guaranteed to work on older datasets. If you have an existing evaluation dataset, this script will copy configuration and evals from the existing dataset into the newly created instance. Data in the existing dataset will remain unchanged.
# MAGIC
# MAGIC
# MAGIC
# MAGIC Notes:
# MAGIC - Agent name and serving endpoint config will be migrated.
# MAGIC - Tags and evaluations will be migrated.
# MAGIC - Grading notes will be added to `expected_facts`.
# MAGIC - Document table will **NOT** be migrated.

# COMMAND ----------

# DBTITLE 1,Define the migration script
from databricks.agents.evals import datasets as eval_datasets
from typing import Literal
from mlflow.utils.rest_utils import http_request
from mlflow.utils import databricks_utils
from mlflow.environment_variables import MLFLOW_HTTP_REQUEST_TIMEOUT
from mlflow.deployments.constants import MLFLOW_DEPLOYMENT_CLIENT_REQUEST_RETRY_CODES
from concurrent.futures import ThreadPoolExecutor

# Note that the synth prefix also contains '(NOTE: It is sufficient for the answer to mention the above **implicitly** (explicit agreement is not required!):' but we are relaxing the prefix for conversion.
_OLD_GRADING_NOTES_SYNTH_PREFIX='The answer should mention the following facts either explicitly or implicitly'

def migrate_evals_table(old_eval_dataset, new_eval_dataset):
    print(f"Migrating '{old_eval_dataset}' to '{new_eval_dataset}'...")
    # Validate new eval dataset exists
    assert (
        spark.catalog.tableExists(new_eval_dataset)
    ), f"New eval dataset {new_eval_dataset} does not exist. Please create it first using `create_evals_table`."

    # Validate old eval dataset
    old_eval_dataset_is_not_view = set(spark.table(old_eval_dataset).columns) == set(
        ["key", "value"]
    )
    if old_eval_dataset_is_not_view:
        old_evals_view_name = old_eval_dataset + "_view"
        old_metadata_table_name = old_eval_dataset
    else:
        old_evals_view_name = old_eval_dataset
        old_metadata_table_name = old_eval_dataset + "_metadata"

    # Get metadata from the old dataset
    old_metadata = {}
    for row in (
        spark.table(old_metadata_table_name).toPandas().to_dict(orient="records")
    ):
        old_metadata[row["key"]] = row["value"]

    # Get tags from the old dataset
    old_eval_dataset_tags_table = f'{old_eval_dataset}_tags'
    old_tag_names = []
    if spark.catalog.tableExists(new_eval_dataset):
        # Read the unique `tag_name` column off the tags table.
        old_tag_names_rows = spark.table(old_eval_dataset_tags_table).select(
            "tag_name"
            ).distinct().collect()
        old_tag_names = [row.tag_name for row in old_tag_names_rows]

    # Get evals from the old dataset
    evals_df = spark.table(old_evals_view_name).toPandas()
    evals = []
    expected_responses_to_convert = []
    for i, e in enumerate(evals_df.to_dict(orient="records")):
        expected_facts = list(e.get("expected_facts", []))
        expected_response = e.get("expected_response") or e.get("ground_truth")

        old_grading_notes = e.get("grading_notes")
        # Extract the facts from the old grading notes.
        if old_grading_notes:
            if old_grading_notes.lstrip("- ").strip().startswith(_OLD_GRADING_NOTES_SYNTH_PREFIX):
                # Remove the first line of the grading notes, which is the prefix.
                old_grading_notes = '\n'.join(old_grading_notes.split("\n")[1:])
            
            # Remove whitespace + dash prefixes on each line.
            old_grading_notes_lines = [
                line.lstrip("- ").strip() for line in old_grading_notes.split('\n')
            ]
            # Remove any empty lines.
            old_grading_notes_lines = [line for line in old_grading_notes_lines if line]
            expected_facts.extend(old_grading_notes_lines)
        elif expected_response:
            # Convert the expected_response to expected_facts.
            expected_responses_to_convert.append((i, e.get("request"), expected_response))

        old_source_type = e.get("source_type") or e.get("request_source") or ""
        old_source_type = old_source_type.lower()
        if 'human' in old_source_type:
            source_type = "HUMAN_SOURCE"
        elif 'synthetic' in old_source_type:
            source_type = "SYNTHETIC_FROM_DOC"
        else:
            source_type = None
        source_id = e.get("source_id")
        if not source_id:
            source_id = (
                e.get("synthetic_source_doc_id")
                if source_type == "SYNTHETIC_FROM_DOC"
                else e.get("created_by")
            )

        tags = e.get("tags", [])
        tag_names = [
            tag if isinstance(tag, str) else tag.get("tag_name") for tag in tags
        ]

        review_status = e.get("review_status") or e.get("status")
        evals.append(
            {
                "request_id": e.get("request_id"),
                "request": e.get("request"),
                "expected_response": None,
                "expected_facts": expected_facts,
                "expected_retrieved_context": e.get("expected_retrieved_context"),
                "source_type": source_type,
                "source_id": source_id,
                "tags": tag_names,
                "review_status": review_status,
            }
        )
    # Convert expected responses in a threadpool.
    thread_pool = ThreadPoolExecutor(max_workers=5)
    if expected_responses_to_convert:
        print('Converting `expected_response`s to `expected_facts`... This may take a few minutes.')

    expected_facts_conversions = list(
        thread_pool.map(
            lambda eval: (eval[0], _convert_expected_response_to_expected_facts(request=eval[1], expected_response=eval[2])),
            expected_responses_to_convert
        )
    )
    # Map the expected_facts_conversions back.
    for i, expected_facts in expected_facts_conversions:
        evals[i]["expected_facts"] = expected_facts

    # Add tags
    if old_tag_names:
        eval_datasets.add_tags(new_eval_dataset, tag_names=old_tag_names)

    # Add evals
    eval_datasets.add_evals(new_eval_dataset, evals=evals)

    print(f"Successfully migrated {len(evals)} evals from '{old_eval_dataset}' to '{new_eval_dataset}'")

def _convert_expected_response_to_expected_facts(
    request: str, 
    expected_response: str
) -> list[str]:
    try:
        body = {
            "question": request,
            "context": expected_response,
            "answer_types": ["MINIMAL_FACTS"]
        }
        endpoint = "managed-evals/generate-answer"

        _MAX_RETRIES = 5
        _BACKOFF_FACTOR = 2
        _BACKOFF_JITTER = 1
        response = http_request(
            host_creds=databricks_utils.get_databricks_host_creds(),
            endpoint=f"/api/2.0/{endpoint}",
            method="POST" if body else "GET",
            timeout=MLFLOW_HTTP_REQUEST_TIMEOUT,
            max_retries=5,
            backoff_factor=2,
            backoff_jitter=1,
            raise_on_status=False,
            retry_codes=MLFLOW_DEPLOYMENT_CLIENT_REQUEST_RETRY_CODES,
            json=body,
            params={},
        )
        expected_facts_response = response.json()
        return expected_facts_response.get("synthetic_minimal_facts", [])
    except:
        return []

# COMMAND ----------

# NOTE: Only run this cell if you want to migrate old data. You must run the cell above to define the migration script (it is hidden for brevity).

# TODO: Replace this with your older evals dataset name.
OLD_EVAL_DATASET_NAME = "cat.sch.old_eval_dataset"

# Run the migration. It will copy the data from OLD_EVAL_DATASET_NAME into EVAL_TABLE.
# Synthesized grading notes will get converted to expected_facts.
# `expected_response`s will get converted to expected_facts.
migrate_evals_table(OLD_EVAL_DATASET_NAME, EVAL_TABLE)

display(spark.read.table(EVAL_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Synthesize evaluations from documents
# MAGIC
# MAGIC This section demonstrate a Python API to synthesize evaluations from a dataframe of documents. The generated evaluations can be imported into evaluation datasets, reviewed and annotated by SME, and used with mlflow.evaluate().
# MAGIC
# MAGIC For more details, see the [documentation](https://docs.databricks.com/generative-ai/agent-evaluation/synthesize-evaluation-set.html).
# MAGIC
# MAGIC #### API
# MAGIC ```python
# MAGIC def generate_evals_df(
# MAGIC     docs: Union[pd.DataFrame, "pyspark.sql.DataFrame"],
# MAGIC     *,
# MAGIC     num_evals: int,
# MAGIC     guidelines: Optional[str] = None,
# MAGIC ) -> pd.DataFrame:
# MAGIC     """
# MAGIC     Generate an evaluation dataset with questions and expected answers.
# MAGIC     Generated evaluation set can be used with Databricks Agent Evaluation
# MAGIC     (https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluate-agent.html).
# MAGIC
# MAGIC     :param docs: A pandas/Spark DataFrame with a text column `content` and a `doc_uri` column.
# MAGIC     :param num_evals: The number of questions (and corresponding answers) to generate in total.
# MAGIC     :param guidelines: Optional guidelines to help guide the synthetic generation. This is a free-form string that will
# MAGIC         be used to prompt the generation. The string can be formatted in markdown and may include sections like:
# MAGIC         - Task Description: Overview of the agent's purpose and scope
# MAGIC         - User Personas: Types of users the agent should support
# MAGIC         - Example Questions: Sample questions to guide generation
# MAGIC         - Additional Guidelines: Extra rules or requirements
# MAGIC     """
# MAGIC ```

# COMMAND ----------

# Use the synthetic eval generation API to get some evals
from databricks.agents.evals import generate_evals_df
import pandas as pd

# These documents can be a Pandas DataFrame or a Spark DataFrame. It must have two columns: 'content' and 'doc_uri'.
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
        'doc_uri': 'https://spark.apache.org/docs/3.5.2/index.html'
      },
      {
        'content': f"""
            Spark’s primary abstraction is a distributed collection of items called a Dataset. Datasets can be created from Hadoop InputFormats (such as HDFS files) or by transforming other Datasets. Due to Python’s dynamic nature, we don’t need the Dataset to be strongly-typed in Python. As a result, all Datasets in Python are Dataset[Row], and we call it DataFrame to be consistent with the data frame concept in Pandas and R.""",
        'doc_uri': 'https://spark.apache.org/docs/3.5.2/quick-start.html'
      }
    ]
)

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

num_evals = 10  # Generating 10 questions

synthetic_evals = generate_evals_df(
    docs,
    # The total number of evals to generate. We will attempt to generate evals that have full coverage over the documents
    # provided. If this number is lower than the number of evals that can be generated to get full coverage over the docs,
    # we will generate this many evals from the first set of documents in order and return the `source_doc_ids` that we
    # generated from. This can be used to join back to your original dataframe if you want to generate more evals, and
    # ignore doc_ids that you've already synthesized from.
    num_evals=num_evals,
    # A set of guidelines that help guide the synthetic generation. This is a free-form string that will be used to prompt the generation.
    guidelines=guidelines
)

display(synthetic_evals)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add synthetic evals to the evaluation dataset
# MAGIC
# MAGIC #### API
# MAGIC
# MAGIC ```python
# MAGIC def add_evals(
# MAGIC     evals_table_name: str,
# MAGIC     *,
# MAGIC     evals: Union[List[Dict], pd.DataFrame, pyspark.sql.DataFrame],
# MAGIC ) -> List[str]:
# MAGIC     """
# MAGIC     Add evals to the evals table.
# MAGIC
# MAGIC     Args:
# MAGIC         evals_table_name: The name of the evals table.
# MAGIC         evals: The evals to add. This can be a list of dictionaries, a pandas DataFrame, or a Spark DataFrame.
# MAGIC
# MAGIC     Returns:
# MAGIC         A list of the IDs of the added evals.
# MAGIC     """
# MAGIC ```
# MAGIC
# MAGIC Input `evals` should be in [Agent Evaluation input schema](https://docs.databricks.com/generative-ai/agent-evaluation/evaluation-schema.html#evaluation-input-schema).

# COMMAND ----------

eval_datasets.add_evals(evals_table_name=EVAL_TABLE, evals=synthetic_evals)
display(spark.table(EVAL_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grant SME permissions to review evals
# MAGIC
# MAGIC This section demonstrate how to grant SME access to review and edit the evals, as well as invoke the agent (if you have one configured).
# MAGIC
# MAGIC
# MAGIC #### API
# MAGIC
# MAGIC ```python
# MAGIC def grant_access(
# MAGIC     evals_table_name: str,
# MAGIC     *,
# MAGIC     user_emails: List[str],
# MAGIC ) -> None:
# MAGIC     """Grant access to read and modify the evals table (via Spark and UI) to the specified users.
# MAGIC
# MAGIC     If an agent is configured for the labeling UI, this function also grants QUERY access to the model.
# MAGIC
# MAGIC     Args:
# MAGIC         evals_table_name: The name of the evals table.
# MAGIC         user_emails: The emails of the users to grant access to.
# MAGIC     """
# MAGIC
# MAGIC def revoke_access(
# MAGIC     evals_table_name: str,
# MAGIC     *,
# MAGIC     user_emails: List[str],
# MAGIC ) -> None:
# MAGIC     """Revoke access to read and modify the evals table (via Spark and UI) to the specified users.
# MAGIC
# MAGIC     If an agent is configured for the labeling UI, this function also revokes QUERY access to the model.
# MAGIC
# MAGIC     Args:
# MAGIC         evals_table_name: The name of the evals table.
# MAGIC         user_emails: The emails of the users to revoke access from.
# MAGIC     """
# MAGIC ```

# COMMAND ----------

user_emails = ["person@example.com"]
eval_datasets.grant_access(EVAL_TABLE, user_emails=user_emails)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review the evals in the UI
# MAGIC
# MAGIC The developer and SME UIs allow users to view and edit requests and expected facts.
# MAGIC
# MAGIC > Expected facts is an effective contract for communicating expectations to an LLM judge for automated reviewing, allowing the SME to review the expected facts just once, and the developer to then rely on LLM judges to improve agent quality.
# MAGIC
# MAGIC The SME also has the opportunity to mark requests as REJECTED if they feel that the question is not useful or relevant.
# MAGIC
# MAGIC Finally, the developer can see and accept the SME's recommendations in the Developer Dashboard.
# MAGIC

# COMMAND ----------

# Display the links to the UI
eval_datasets.get_ui_links(EVAL_TABLE)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Run mlflow.evaluate() on the reviewed evaluation dataset
# MAGIC
# MAGIC Reviewed evaluations can be retrieved by querying the UC Table at `EVAL_TABLE`.

# COMMAND ----------

import mlflow
from pyspark.sql.functions import array_contains

# Read the evals table.
evals = spark.read.table(EVAL_TABLE)

# Optionally filter the data by review status or tags.
# evals = evals.filter('review_status == "REVIEWED"') # Other values are: "ACCEPTED", "DRAFT", "REVIEWED", null.
# evals = evals.filter(array_contains(evals.tags, "my_tag"))

# Evaluate the model using the newly generated evaluation set. After the function call completes, click the UI link to see the results. You can use this as a baseline for your agent.
results = mlflow.evaluate(
    model="endpoints:/databricks-meta-llama-3-1-70b-instruct", # Replace with your Agent endpoint
    data=evals.toPandas(),
    model_type="databricks-agent"
)

display(results.tables["eval_results"])
