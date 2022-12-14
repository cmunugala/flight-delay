# Databricks notebook source
# import libraries
import pyspark.sql.functions as F
from pyspark.sql.types import *
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pyspark.sql import functions as f
from pyspark.sql import SQLContext
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import isnan, when, count, col, isnull, percent_rank, avg, mean
from pyspark.sql.functions import min
from pyspark.sql.functions import col, max
from pyspark.sql.functions import format_string
from pyspark.sql.functions import substring
from pyspark.sql.functions import concat_ws
from pyspark.sql.functions import concat
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import lit
from pyspark.sql.functions import to_utc_timestamp
from pyspark.sql.functions import expr
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import instr
from pyspark.sql.functions import row_number
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType

from pyspark.ml.linalg import DenseVector, SparseVector, Vectors
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer,OneHotEncoder
from pyspark.ml.classification import MultilayerPerceptronClassifier

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import GBTClassifier

from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# COMMAND ----------

#Initializes blob storage credentials/location
blob_container = "w261-sec4-group2" # The name of your container created in https://portal.azure.com
storage_account = "kdevery" # The name of your Storage account created in https://portal.azure.com
secret_scope = "sec4-group2" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

#Points to SAS token
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

display(dbutils.fs.ls(f"{blob_url}"))

# COMMAND ----------

#Read processed folds from the blob

lr_df = spark.read.parquet(f"{blob_url}/predictions_LR")
lr_under_df = spark.read.parquet(f"{blob_url}/predictions_LR_undersampled")
rf_df = spark.read.parquet(f"{blob_url}/predictions_RF")
rf_under_df = spark.read.parquet(f"{blob_url}/predictions_RF_undersampled")
gbt_df = spark.read.parquet(f"{blob_url}/predictions_GBT")
gbt_under_df = spark.read.parquet(f"{blob_url}/predictions_GBT_undersampled")
mlpc_df = spark.read.parquet(f"{blob_url}/MLPC_predictions_df_no_undersampling")
mlpc_under_df = spark.read.parquet(f"{blob_url}/MLPC_predictions_df")

# COMMAND ----------

lr_df = lr_df.withColumnRenamed('prediction', 'lr_prediction').select('label', 'index', 'lr_prediction')
lr_under_df = lr_under_df.withColumnRenamed('prediction', 'lr_under_prediction').select('label', 'index', 'lr_under_prediction')
rf_df = rf_df.withColumnRenamed('prediction', 'rf_prediction').select('label', 'index', 'rf_prediction')
rf_under_df = rf_under_df.withColumnRenamed('prediction', 'rf_under_prediction').select('label', 'index', 'rf_under_prediction')
gbt_df = gbt_df.withColumnRenamed('prediction', 'gbt_prediction').select('label', 'index', 'gbt_prediction')
gbt_under_df = gbt_under_df.withColumnRenamed('prediction', 'gbt_under_prediction').select('label', 'index', 'gbt_under_prediction')
mlpc_df = mlpc_df.withColumnRenamed('prediction', 'mlpc_prediction').select('label', 'index', 'mlpc_prediction')
mlpc_under_df = mlpc_under_df.withColumnRenamed('prediction', 'mlpc_under_prediction').select('label', 'index', 'mlpc_under_prediction')

# COMMAND ----------

temp_1df = lr_df.join(lr_under_df, ['index', 'label'])
temp_2df = temp_1df.join(rf_df, ['index', 'label'])
temp_3df = temp_2df.join(rf_under_df, ['index', 'label'])
temp_4df = temp_3df.join(gbt_df, ['index', 'label'])
temp_5df = temp_4df.join(gbt_under_df, ['index', 'label'])
temp_6df = temp_5df.join(mlpc_df, ['index', 'label'])
compiled_df = temp_6df.join(mlpc_under_df, ['index', 'label']).cache()

display(compiled_df)
compiled_df.count()

# COMMAND ----------

compiled_df.columns

# COMMAND ----------

pred_cols = ['lr_prediction', 'lr_under_prediction', 'rf_prediction', 'rf_under_prediction', 'gbt_prediction', 'gbt_under_prediction', 'mlpc_prediction', 'mlpc_under_prediction']
pred_cols_no_under = ['lr_prediction', 'rf_prediction', 'gbt_prediction', 'mlpc_prediction']
compiled_df = compiled_df.withColumn('sum_predictions', sum(compiled_df[col] for col in pred_cols)) \
                         .withColumn('sum_predictions_no_under', sum(compiled_df[col] for col in pred_cols_no_under))


display(compiled_df)

# COMMAND ----------

compiled_df = compiled_df.withColumn('majority_vote_pred', (compiled_df.sum_predictions >= 4).cast('double')) \
                         .withColumn('any_pred', (compiled_df.sum_predictions >= 1).cast('double')) \
                         .withColumn('majority_vote_pred_no_under', (compiled_df.sum_predictions_no_under >= 2).cast('double')) \
                         .withColumn('any_pred_no_under', (compiled_df.sum_predictions_no_under >= 1).cast('double'))
display(compiled_df)

# COMMAND ----------

majority_vote_df = compiled_df.select('label', 'majority_vote_pred').withColumnRenamed('majority_vote_pred', 'prediction')
any_vote_df = compiled_df.select('label', 'any_pred').withColumnRenamed('any_pred', 'prediction')
majority_vote_no_under_df = compiled_df.select('label', 'majority_vote_pred_no_under').withColumnRenamed('majority_vote_pred_no_under', 'prediction')
any_vote_no_under_df = compiled_df.select('label', 'any_pred_no_under').withColumnRenamed('any_pred_no_under', 'prediction')

# COMMAND ----------

# Example using F0.5 score for evaluator
evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=0.5, metricLabel=1)

# COMMAND ----------

evaluator.evaluate(majority_vote_df)

# COMMAND ----------

evaluator.evaluate(any_vote_df)

# COMMAND ----------

evaluator.evaluate(majority_vote_no_under_df)

# COMMAND ----------

evaluator.evaluate(any_vote_no_under_df)

# COMMAND ----------

