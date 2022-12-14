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

import seaborn as sns

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

cvModel = cv.fit(d_processed)

# COMMAND ----------

# make predictions
predictions = cvModel.transform(d_processed['df1'])

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

final_gbt_model = GBTClassifier(labelCol="label", featuresCol="scaled_feature_vector",maxDepth=5, minInfoGain=0, maxBins=64)
evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=0.5, metricLabel=1)
gbtModel = final_gbt_model.fit(processed_train_df)

# COMMAND ----------

feature_importance_df = pd.DataFrame(gbtModel.featureImportances.toArray())
feature_importance_df.columns = ['Importance_Score']

# COMMAND ----------

feature_cols= ['MONTH','DAY_OF_MONTH','DAY_OF_WEEK','DISTANCE','HourlyWindSpeed','Rain','Blowing','Snow','Thunder','CloudySkyCondition','carrier_vec','holiday_period','mean_carrier_delay','Pagerank_Score','PREV_FLIGHT_DELAYED','origin_percent_delayed','dest_percent_delayed','ORIGIN_Prophet_trend','ORIGIN_Prophet_pred','DEST_Prophet_trend','DEST_Prophet_pred']

len(feature_cols)

# COMMAND ----------

plot_df = feature_importance_df[feature_importance_df['Importance_Score'] > 0.01]
plot_df.index = ['DAY_OF_WEEK', 'DISTANCE', 'HourlyWindSpeed', 'Blowing', 'Snow', 'Thunder', 'mean_carrier_delay', 'Pagerank_Score', 'PREV_FLIGHT_DELAYED', 'ORIGIN_Prophet_trend', 'ORIGIN_Prophet_pred', 'DEST_Prophet_trend']

plot = plot_df.plot(kind='bar',figsize = (15,6),title='Feature Importance',fontsize=15)
plot.axes.set_title("Feature Importance",fontsize=30)

# COMMAND ----------

predictions = gbtModel.transform(processed_test_df)

# COMMAND ----------

display(predictions.groupby('label', 'prediction').count())

# COMMAND ----------

evaluator.evaluate(predictions)

# COMMAND ----------

display(predictions)

# COMMAND ----------

# predictions.write.mode("overwrite").parquet(f"{blob_url}/predictions_GBT")

# COMMAND ----------

# MAGIC %md
# MAGIC #Shapely Values

# COMMAND ----------

#get your imports out of the way! 
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import GBTClassificationModel
import pyspark.sql.functions as  F
from pyspark.sql.types import *

#convert the sparse feature vector that is passed to the MLlib GBT model into a pandas dataframe. 
#This 'pandas_df' will be passed to the Shap TreeExplainer.
rows_list = []
for row in processed_train_df.rdd.collect(): 
    dict1 = {}
    dict1.update({k:v for k,v in zip(processed_train_df.cols,row.scaled_feature_vector)})
    rows_list.append(dict1)

pandas_df = pd.DataFrame(rows_list)

#Load the GBT model from the path you have saved it
gbt = gbtModel
#make sure the application where your notebook runs has access to the storage path!

explainer = shap.TreeExplainer(gbt)
#check_additivity requires predictions to be run that is not supported by spark [yet], so it needs to be set to False as it is ignored anyway.
shap_values = explainer(pandas_df, check_additivity = False)
shap_pandas_df = pd.DataFrame(shap_values.values, cols = pandas_df.columns)

spark = SparkSession.builder.config(conf=SparkConf().set("spark.master", "local[*]")).getOrCreate()
spark_shapdf = spark.createDataFrame(shap_pandas_df)


def shap_udf(row): #work on a single spark dataframe row, for all rows. This work is distributed among all the worker nodes of your Apache Spark cluster.
    dict = {}
    pos_features = []
    neg_features = []

    for feature in row.columns:
        dict[feature] = row[feature]

    dict_importance = {key: value for key, value in sorted(dict.items(), key=lambda item: __builtin__.abs(item[1]), reverse = True)}

    for k,v in dict_importance.items():
        if __builtin__.abs(v) >= 0.1:
            if v > 0:
                pos_features.append((k,v))
            else:
                neg_features.append((k,v))
    features = []
    #taking top 5 features from pos and neg features. We can increase this number.
    features.append(pos_features[:5])
    features.append(neg_features[:5])

    return features


udf_obj = F.udf(shap_udf, ArrayType(ArrayType(StructType([
  StructField('Feature', StringType()),
  StructField('Shap_Value', FloatType()),
]))))

new_sparkdf = spark_df.withColumn('Shap_Importance', udf_obj(F.struct([spark_shapdf[x] for x in spark_shapdf.columns])))
final_sparkdf = new_sparkdf.withColumn('Positive_Shap', final_sparkdf.Shap_Importance[0]).withColumn('Negative_Shap', new_sparkdf.Shap_Importance[1])

# COMMAND ----------

display(final_sparkdf)

# COMMAND ----------

