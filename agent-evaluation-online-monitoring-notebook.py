# Databricks notebook source
# MAGIC %md
# MAGIC # Monitor agent quality in production
# MAGIC This notebook runs Agent Evaluation on a sample of the requests served by an agent endpoint. 
# MAGIC - To run the notebook once, fill in the required parameters up top and click **Run all**. 
# MAGIC - To continuously monitor your production traffic, click **Schedule** to create a job to run the notebook periodically. For endpoints with a large number of requests, we recommend setting an hourly schedule.
# MAGIC
# MAGIC The notebook creates a few artifacts:
# MAGIC - A table that records a sample of the requests received by an agent endpoint along with the metrics calculated by Agent Evaluation on those requests. 
# MAGIC - A dashboard that visualizes the evaluation results.
# MAGIC - An MLFlow experiment to track runs of `mlflow.evaluate`
# MAGIC  
# MAGIC The derived table has the name `<inference_table>_request_logs_eval`, where `<inference_table>` is the inference table associated with the agent endpoint. The dashboard is created automatically and is linked in the final cells of the notebook. You can use the table of contents at the left of the notebook to go directly to this cell. 
# MAGIC
# MAGIC **Note:** You should not need to edit this notebook, other than filling in the widgets at the top. This notebook requires either Serverless compute or a cluster with Databricks Runtime 15.2 or above.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -qqq databricks-agents mlflow[databricks] databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Initialize widgets
dbutils.widgets.text(name="endpoint_name", defaultValue="", label="1. Agent's Model serving endpoint name")

dbutils.widgets.text(name="topics", defaultValue="\"other\"", label="""3. List of topics in which to categorize the input requests. Format must be comma-separated strings in double quotes, e.g., "finance", "healthcare", "other". We recommend having an "other" category as a catch-all.""")

dbutils.widgets.text(name="sample_rate", defaultValue="0.3", label="2. Sampling rate between 0.0 and 1.0: on what % of requests should the LLM judge quality analysis run?")

dbutils.widgets.text(name="workspace_folder", defaultValue="", label="4. (Optional) Workspace folder for storing artifacts like the experiment and dashboard (e.g., `/Users/{your_username}`). Default value is home folder.")

# COMMAND ----------

# DBTITLE 1,Helper methods
import pandas as pd
from typing import List
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace
from IPython.display import display_markdown
from mlflow.utils import databricks_utils as du

wc = WorkspaceClient()

def is_valid_endpoint(name: str) -> bool:
  try:
    wc.serving_endpoints.get(endpoint_name)
    return True
  except:
    return False

def get_payload_table_name(endpoint_info) -> str:
  catalog_name = endpoint_info.config.auto_capture_config.catalog_name
  schema_name = endpoint_info.config.auto_capture_config.schema_name
  payload_table_name = endpoint_info.config.auto_capture_config.state.payload_table.name
  return f"{catalog_name}.{schema_name}.{payload_table_name}"

def get_model_name(endpoint_info) -> str:
  served_models = [
    model for model in endpoint_info.config.served_models if not model.model_name.endswith(".feedback")
  ]
  return served_models[0].model_name

# Helper function for display Delta Table URLs
def get_table_url(table_fqdn: str) -> str:
    table_fqdn = table_fqdn.replace("`", "")
    split = table_fqdn.split(".")
    browser_url = du.get_browser_hostname()
    url = f"https://{browser_url}/explore/data/{split[0]}/{split[1]}/{split[2]}"
    return url

# Helper function to split the evaluation dataframe by hour
def split_df_by_hour(df: pd.DataFrame, max_samples_per_hours: int) -> List[pd.DataFrame]:
    df['hour'] = pd.to_datetime(df["timestamp"]).dt.floor('H')
    dfs_by_hour = [
        group.sample(min(len(group), max_samples_per_hours), replace=False)
        for _, group in df.groupby('hour')
    ]
    return dfs_by_hour

# COMMAND ----------

# DBTITLE 1,Read parameters
endpoint_name = dbutils.widgets.get("endpoint_name")
assert(is_valid_endpoint(endpoint_name)), 'Please specify a valid serving endpoint name.'

sample_rate = float(dbutils.widgets.get("sample_rate"))
# Verify that sample_rate is a number in [0,1]
assert(0.0 <= sample_rate and sample_rate <= 1.0), 'Please specify a sample rate between 0.0 and 1.0'

sample_rate_display = f"{sample_rate:0.0%}"

workspace_folder = dbutils.widgets.get("workspace_folder").replace("/Workspace", "")
if workspace_folder is None or workspace_folder == "":
  username = spark.sql("select current_user() as username").collect()[0]["username"]
  workspace_folder=f"/Users/{username}"

folder_info = wc.workspace.get_status(workspace_folder)
assert (folder_info is not None and folder_info.object_type == workspace.ObjectType.DIRECTORY), f"Please specify a valid workspace folder. The specified folder {workspace_folder} is invalid."

topics = dbutils.widgets.get("topics")

# Print debugging information
display_markdown("## Monitoring notebook configuration", raw=True)
display_markdown(f"- **Agent Model Serving endpoint name:** `{endpoint_name}`", raw=True)
display_markdown(f"- **% of requests that will be run through LLM judge quality analysis:** `{sample_rate_display}`", raw=True)
display_markdown(f"- **Storing output artifacts in:** `{workspace_folder}`", raw=True)
display_markdown(f"- **Topics to detect:** `{topics}`", raw=True)

# COMMAND ----------

# DBTITLE 1,Set up table-name variables
def escape_table_name(table_name: str) -> str:
  return ".".join(list(map(lambda x: f"`{x}`", table_name.split("."))))

# Deployed agents create multiple inference tables that can be used for further processing such as evaluation. See the documentation:
# AWS documentation: https://docs.databricks.com/en/generative-ai/deploy-agent.html#agent-enhanced-inference-tables
# Azure documentation: https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent#agent-enhanced-inference-tables
endpoint_info = wc.serving_endpoints.get(endpoint_name)
inference_table_name = get_payload_table_name(endpoint_info)
fully_qualified_uc_model_name = get_model_name(endpoint_info)

requests_log_table_name = f"{inference_table_name}_request_logs"
eval_requests_log_table_name = escape_table_name(f"{requests_log_table_name}_eval")
assessment_log_table_name = escape_table_name(f"{inference_table_name}_assessment_logs")

eval_requests_log_checkpoint = f"{requests_log_table_name}_eval_checkpoint"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {escape_table_name(eval_requests_log_checkpoint)}")
eval_requests_log_checkpoint_path = f"/Volumes/{eval_requests_log_checkpoint.replace('.', '/')}"

# Print debugging information
display_markdown("## Input tables", raw=True)
display_markdown(
  f"- **Inference table:** [{inference_table_name}]({get_table_url(inference_table_name)})",
  raw=True
)
display_markdown(
  f"- **Request logs table:** [{requests_log_table_name}]({get_table_url(requests_log_table_name)})",
  raw=True
)
display_markdown(
  f'- **Human feedback logs table:** [{assessment_log_table_name.replace("`", "")}]({get_table_url(assessment_log_table_name.replace("`", ""))})',
  raw=True
)
display_markdown("## Output tables/volumes", raw=True)
display_markdown(
  f'- **LLM judge results table:** [{eval_requests_log_table_name.replace("`", "")}]({get_table_url(eval_requests_log_table_name.replace("`", ""))})',
  raw=True
)
display_markdown(f"- **Streaming checkpoints volume:** `{eval_requests_log_checkpoint_path}`", raw=True)

# COMMAND ----------

# DBTITLE 1,Iniitialize mlflow experiment
import mlflow
mlflow_client = mlflow.MlflowClient()

# Create a single experiment to store all monitoring runs for this endpoint
experiment_name = f"{workspace_folder}/{inference_table_name}_experiment"
experiment = mlflow_client.get_experiment_by_name(experiment_name)
experiment_id = mlflow_client.create_experiment(experiment_name) if experiment is None else experiment.experiment_id

mlflow.set_experiment(experiment_name=experiment_name)

# COMMAND ----------

# DBTITLE 1,Update the table with unprocessed requests
import pyspark.sql.functions as F

# Streams any unprocessed rows from the requests log table into the evaluation requests log table.
# Unprocessed requests have an empty run_id.
# Processed requests have one of the following values: "skipped", "to_process", or a valid run_id.
(
  spark.readStream.format("delta")
  .table(escape_table_name(requests_log_table_name))
  .withColumn("run_id", F.lit(None).cast("string"))
  .withColumn(
    "retrieval/llm_judged/chunk_relevance/precision", F.lit(None).cast("double")
  )
  .writeStream.option("checkpointLocation", eval_requests_log_checkpoint_path)
  .option("mergeSchema", "true")
  .format("delta")
  .outputMode("append")
  .trigger(availableNow=True)
  .toTable(eval_requests_log_table_name)
  .awaitTermination()
)

# COMMAND ----------

# DBTITLE 1,Mark rows for processing, mark the rest as "skipped"
spark.sql(f"""
          WITH sampled_requests AS (
            -- Sample unprocessed requests
            SELECT *
            FROM (
              SELECT databricks_request_id
              FROM {eval_requests_log_table_name}
              WHERE 
                run_id IS NULL
                AND `timestamp` >= CURRENT_TIMESTAMP() - INTERVAL 30 DAY
            ) TABLESAMPLE ({sample_rate*100} PERCENT)
          ), requests_with_user_feedback AS (
            -- Get unprocessed requests with user feedback
            SELECT assessments.request_id
            FROM {assessment_log_table_name} assessments
            INNER JOIN {eval_requests_log_table_name} requests
            ON assessments.request_id = requests.databricks_request_id
            WHERE 
              requests.run_id is NULL
              AND assessments.`timestamp` >= CURRENT_TIMESTAMP() - INTERVAL 30 DAY
          )

          -- Mark the selected rows as `to_process`
          UPDATE {eval_requests_log_table_name} 
          SET run_id="to_process"
          WHERE databricks_request_id IN (
            SELECT * FROM sampled_requests
            UNION 
            SELECT * FROM requests_with_user_feedback
          )
          """)

###############
# CONFIG: Add custom logic here to select more rows. Update the run_id of selected rows to the value "to_process".
###############

spark.sql(f"""
          UPDATE {eval_requests_log_table_name}
          SET run_id="skipped"
          WHERE run_id IS NULL
          """)

# COMMAND ----------

# DBTITLE 1,Run evaluate on unprocessed rows
from databricks.rag_eval import env_vars

eval_df = spark.sql(f"""
                    SELECT 
                      `timestamp`,
                      databricks_request_id as request_id, 
                      from_json(request_raw, 'STRUCT<messages ARRAY<STRUCT<role STRING, content STRING>>>') AS request,
                      trace
                    FROM {eval_requests_log_table_name} 
                    WHERE run_id="to_process"                  
                    """)

eval_pdf = eval_df.toPandas().drop_duplicates(subset=["request_id"])

if eval_pdf.empty:
  print("[Warning] No new rows to process.")
else:
  with mlflow.start_run() as run:
    ###############
    # CONFIG: Adjust mlflow.evaluate(...) to change which Databricks LLM judges are run. By default, judges that do not require ground truths
    # are run, including groundedness, safety, chunk relevance, and relevance to query. For more details, see the documentation:
    # AWS documentation: https://docs.databricks.com/en/generative-ai/agent-evaluation/advanced-agent-eval.html#evaluate-agents-using-a-subset-of-llm-judges
    # Azure documentation: https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/advanced-agent-eval#evaluate-agents-using-a-subset-of-llm-judges
    ###############
    for eval_pdf_batch in split_df_by_hour(eval_pdf, max_samples_per_hours=env_vars.RAG_EVAL_MAX_INPUT_ROWS.get()):
      eval_results = mlflow.evaluate(data=eval_pdf_batch, model_type="databricks-agent")

      results_df = (
        spark
        .createDataFrame(eval_results.tables['eval_results'])
        .withColumn("databricks_request_id", F.col("request_id"))
        .withColumn("run_id", F.lit(run.info.run_id).cast("string"))
        .withColumn("experiment_id", F.lit(experiment_id).cast("string"))
        .withColumn("topic", F.lit(None).cast("string"))
        .drop("request_id")
        .drop("request")
        .drop("response")
        .drop("trace")
      )

      results_df.createOrReplaceTempView("updates")

      spark.sql(f"""
                merge with schema evolution into {eval_requests_log_table_name} evals 
                using updates ON evals.databricks_request_id=updates.databricks_request_id 
                WHEN MATCHED THEN UPDATE SET *
                """)

# COMMAND ----------

# DBTITLE 1,Perform topic detection
# Perform topic detection using the `ai_classify` function. For more details, see the documentation:
# AWS documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_classify.html
# Azure documentation: https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/ai_classify
if not eval_pdf.empty:   
  if not len(topics.strip()) or topics == "\"other\"":
    spark.sql(f"""
              merge with schema evolution into {eval_requests_log_table_name} evals 
              using updates ON evals.databricks_request_id=updates.databricks_request_id 
              WHEN MATCHED THEN UPDATE SET topic="other"
              """)
  else:          
    spark.sql(f"""
              merge with schema evolution into {eval_requests_log_table_name} evals 
              using updates ON evals.databricks_request_id=updates.databricks_request_id 
              WHEN MATCHED THEN UPDATE SET topic=ai_classify(request, ARRAY({topics}))
              """)

# COMMAND ----------

# DBTITLE 1,Load the dashboard template
import requests

dashboard_template_url = 'https://raw.githubusercontent.com/databricks/genai-cookbook/main/rag_app_sample_code/resources/agent_quality_online_monitoring_dashboard_template.json'
response = requests.get(dashboard_template_url)

if response.status_code == 200:
  dashboard_template = str(response.text)
else:
  raise Exception("Failed to get the dashboard template. Please try again or download the template directly from the URL.")    

# COMMAND ----------

# DBTITLE 1,Create or get the dashboard
from databricks.sdk import WorkspaceClient
from databricks.sdk import errors

wc = WorkspaceClient()
object_info = None

num_evaluated_requests = spark.sql(f"select count(*) as num_rows from {eval_requests_log_table_name}").collect()[0].num_rows
if not num_evaluated_requests:
  print("There are no evaluated requests! Skipping dashboard creation.")
else:
  dashboard_name = f"Monitoring dashboard for {fully_qualified_uc_model_name}"
  try:
    dashboard_content = (
      dashboard_template
        .replace("{{inference_table_name}}", inference_table_name)
        .replace("{{eval_requests_log_table_name}}", eval_requests_log_table_name)
        .replace("{{assessment_log_table_name}}", assessment_log_table_name)
        .replace("{{fully_qualified_uc_model_name}}", fully_qualified_uc_model_name)
        .replace("{{sample_rate}}", sample_rate_display)
    )
    object_info = wc.workspace.get_status(path=f"{workspace_folder}/{dashboard_name}.lvdash.json")
    dashboard_id = object_info.resource_id
  except errors.platform.ResourceDoesNotExist as e:
    dashboard_info = wc.lakeview.create(display_name=dashboard_name, serialized_dashboard=dashboard_content, parent_path=workspace_folder)
    dashboard_id = dashboard_info.dashboard_id
  
  displayHTML(f"""<a href="sql/dashboardsv3/{dashboard_id}">Dashboard</a>""")

# COMMAND ----------

# DBTITLE 1,Query pass rate
# This is a sample query that tracks the average pass rate over a rolling window of one day. You can set a Databricks SQL alert on the 
# result so that you can get notified when the pass rate drops below a threshold. It is possible to write similar queries for more 
# fine-grained judgements (such as specifically on groundedness or safety) or with different slicing (for example, for a rolling window of one 
# week, or sliced by the version of the agent), and to use these queries with Databricks SQL alerts for the purpose of alerting.
# To learn more about Databricks SQL alerts, see the documentation:
# AWS documentation: https://docs.databricks.com/en/sql/user/alerts/index.html
# Azure documentation: https://learn.microsoft.com/en-us/azure/databricks/sql/user/alerts/

pass_rate_alert_query = f"""
  SELECT 
    `date`,
    AVG(pass_indicator) as avg_pass_rate
  FROM (
    SELECT 
      *,
      CASE
        WHEN `response/overall_assessment/rating` = 'yes' THEN 1
        WHEN `response/overall_assessment/rating` = 'no' THEN 0
        ELSE NULL
      END AS pass_indicator
    FROM {eval_requests_log_table_name}
    WHERE `date` >= CURRENT_TIMESTAMP() - INTERVAL 1 DAY
  )
  GROUP BY ALL
"""
spark.sql(pass_rate_alert_query).display()

# COMMAND ----------

# DBTITLE 1,Clean up artifacts
# # WARNING: This cell will delete all artifacts generated by this notebook. PLEASE be extra careful when running this code.
# wc.lakeview.trash(dashboard_id)
# # Deleting the volume will mean you need to process the whole requests log table again!
# spark.sql(f"DROP VOLUME IF EXISTS {escape_table_name(eval_requests_log_checkpoint)}")
# spark.sql(f"DROP TABLE IF EXISTS {eval_requests_log_table_name}")
# # Deleting the experiment will delete all information from `mlflow.evaluate`!
# mlflow.delete_experiment(experiment_id)
