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

def undersample(data, verbose = False):
    """
    Under samples the majority class
    """

    delay_count = data.filter(f.col('label') == 1 ).count()
    non_delay_count = data.filter(f.col('label') == 0 ).count()
    #   print(f' total count : {data.count()}')
    #   print(f' delayed count : {delay_count}')
    #   print(f' non delayed count : {non_delay_count}')

    fraction_undersample = delay_count / non_delay_count
    #   print(f' delayed / non delayed: {fraction_undersample}')

    train_delayed = data.filter(f.col('label') == 1)
    #   print(f' non delayed count df : {train_delayed.count()}')

    train_non_delay_undersample = data.filter(f.col('label') == 0).sample(withReplacement=False, fraction=fraction_undersample, seed = 42)
    #   print(f' oversampled delayed count : {train_non_delay_undersample.count()}')

    data_undersampled = train_delayed.union(train_non_delay_undersample)
    #   print(f' train_df Oversampled : {train_undersampled.count()}')

    return data_undersampled

# COMMAND ----------

d_processed_undersampled = {}

d_processed_undersampled['df1'] = undersample(d_processed['df1'])
d_processed_undersampled['df2'] = undersample(d_processed['df2'])
d_processed_undersampled['df3'] = undersample(d_processed['df3'])
d_processed_undersampled['df4'] = undersample(d_processed['df4'])
d_processed_undersampled['df5'] = undersample(d_processed['df5'])

# COMMAND ----------

d_processed['df1'].groupBy('label').count().show()

# COMMAND ----------

d_processed_undersampled['df1'].groupBy('label').count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Model Building

# COMMAND ----------

# MAGIC %run "/Shared/w261_Section4_Group2/Phase 3/custom_cv_module"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Gradient Boosting Classifier - Cross Validation

# COMMAND ----------

gbt_model = GBTClassifier(labelCol="label", featuresCol="scaled_feature_vector")

grid = ParamGridBuilder()\
            .addGrid(gbt_model.maxDepth, [5,10])\
            .addGrid(gbt_model.minInfoGain,[0.0,0.2,0.4])\
            .addGrid(gbt_model.maxBins, [32,64])\
            .build()
            
# Example using F0.5 score for evaluator
evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=0.5, metricLabel=1)

# COMMAND ----------

# run cross validation & return the crossvalidation F0.5 score for 'test' set
cv = CustomCrossValidator(estimator=gbt_model, estimatorParamMaps=grid, evaluator=evaluator,splitWord =('train','val'), cvCol = 'cv',parallelism=10)

# COMMAND ----------

cvModel = cv.fit(d_processed_undersampled)

# COMMAND ----------

# make predictions
predictions = cvModel.transform(d_processed_undersampled['df1'])

display(predictions.groupby('label', 'prediction').count())

# COMMAND ----------

fbeta = cvModel.avgMetrics[0]
print(f"Gradient Boosting F0.5 Score: {fbeta}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Full Train and Test

# COMMAND ----------

# Read in processed training and test data

processed_train_df = spark.read.parquet(f"{blob_url}/processed_train")
processed_test_df = spark.read.parquet(f"{blob_url}/processed_test")

# COMMAND ----------

processed_train_undersampled_df = undersample(processed_train_df)

# COMMAND ----------

final_gbt_model = GBTClassifier(labelCol="label", featuresCol="scaled_feature_vector",maxDepth=5, minInfoGain=0, maxBins=64)
evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=0.5, metricLabel=1)
gbtModel = final_gbt_model.fit(processed_train_undersampled_df)

# COMMAND ----------

predictions = gbtModel.transform(processed_test_df)

# COMMAND ----------

display(predictions.groupby('label', 'prediction').count())

# COMMAND ----------

evaluator.evaluate(predictions)

# COMMAND ----------

display(predictions)

# COMMAND ----------

# predictions.write.mode("overwrite").parquet(f"{blob_url}/predictions_GBT_undersampled")

# COMMAND ----------

