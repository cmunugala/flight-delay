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

d_processed = {}

d_processed['df1'] = spark.read.parquet(f"{blob_url}/processed_fold_1")
d_processed['df2'] = spark.read.parquet(f"{blob_url}/processed_fold_2")
d_processed['df3'] = spark.read.parquet(f"{blob_url}/processed_fold_3")
d_processed['df4'] = spark.read.parquet(f"{blob_url}/processed_fold_4")
d_processed['df5'] = spark.read.parquet(f"{blob_url}/processed_fold_5")

# COMMAND ----------

display(d_processed['df1'])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Cross Validation and GridSearch for Random Forest

# COMMAND ----------

# MAGIC %run "/Shared/w261_Section4_Group2/Phase 3/custom_cv_module"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Random Forest

# COMMAND ----------

# set up grid search: estimator, set of params, and evaluator
rf_model = RandomForestClassifier(labelCol="label", featuresCol="scaled_feature_vector")
grid = ParamGridBuilder()\
            .addGrid(rf_model.maxDepth, [5,10])\
            .addGrid(rf_model.numTrees, [32,64,128])\
            .build()

# Example using F0.5 score for evaluator
evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=0.5,metricLabel=1)

# COMMAND ----------

cv_rf = CustomCrossValidator(estimator=rf_model, estimatorParamMaps=grid, evaluator=evaluator,splitWord =('train','val'), cvCol = 'cv',parallelism=10)

# COMMAND ----------

cvModel_rf1 = cv_rf.fit(d_processed)

# COMMAND ----------

# make predictions
predictions_rf = cvModel_rf1.transform(d_processed['df1'])

display(predictions_rf.groupby('label', 'prediction').count())

# COMMAND ----------

fbeta = cvModel_rf1.avgMetrics[0]
print(f"Random Forest F0.5 Score: {fbeta}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Full Train and Test Sets

# COMMAND ----------

# Read in processed training and test data

processed_train_df = spark.read.parquet(f"{blob_url}/processed_train")
processed_test_df = spark.read.parquet(f"{blob_url}/processed_test")

# COMMAND ----------

processed_train_df.count()

# COMMAND ----------

processed_test_df.count()

# COMMAND ----------

display(processed_test_df)

# COMMAND ----------

final_RF = RandomForestClassifier(labelCol="label", featuresCol="scaled_feature_vector", maxDepth=20, numTrees=64)
evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=0.5, metricLabel=1)
RFModel = final_RF.fit(processed_train_df)

# COMMAND ----------

predictions = RFModel.transform(processed_test_df)

# COMMAND ----------

display(predictions)

# COMMAND ----------

display(predictions.groupby('label', 'prediction').count())

# COMMAND ----------

evaluator.evaluate(predictions)

# COMMAND ----------

# predictions.write.mode("overwrite").parquet(f"{blob_url}/predictions_RF")