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

# Read in training and test data

train_df = spark.read.parquet(f"{blob_url}/feature_engineered_data")
test_df = spark.read.parquet(f"{blob_url}/feature_engineered_data_test")

# COMMAND ----------

practice_df = train_df.limit(200)

display(practice_df)

# COMMAND ----------

#splitting training dataframe into five folds contained in dictionary "d"

d = {}
folds = ['df1','df2','df3','df4','df5']

each_len = practice_df.count()/5
start = 1
val_size = each_len/5
stop = each_len
precision_list = []

for fold in folds:
    d[fold] = practice_df.filter(col('Index').between(start,stop))\
                                  .withColumn('cv', F.when(col('Index').between(start,(stop-val_size)), 'train')
                                         .otherwise('val'))
    start += each_len
    stop += each_len
                                  

# COMMAND ----------

display(d['df2'])

# COMMAND ----------

def process_fold_df(fold_df):
    
    
    #imputation
    fold_df.createOrReplaceTempView("fold_view")
    
    imputation_columns = ['CRS_ELAPSED_TIME','HourlyAltimeterSetting','HourlyDewPointTemperature',
             'HourlyDryBulbTemperature','HourlyRelativeHumidity','HourlySeaLevelPressure',
             'HourlyStationPressure','HourlyVisibility','HourlyWetBulbTemperature',
             'HourlyWindDirection']

    means = {}

    for impute_col in imputation_columns:
        mean = spark.sql(f"SELECT AVG({impute_col}) FROM fold_view").collect()[0][0]
        means[impute_col] = mean
    
    print(means)
    
    #fill Nones and Nans - Seems to error sometimes?
    fold_df = fold_df.fillna(0,["HourlyWindGustSpeed"]) \
         .fillna(means["CRS_ELAPSED_TIME"],["CRS_ELAPSED_TIME"]) \
         .fillna(means["HourlyAltimeterSetting"],["HourlyAltimeterSetting"]) \
         .fillna(means["HourlyDewPointTemperature"],["HourlyDewPointTemperature"]) \
         .fillna(means["HourlyDryBulbTemperature"],["HourlyDryBulbTemperature"]) \
         .fillna(0,["HourlyPrecipitation"]) \
         .fillna(means["HourlyRelativeHumidity"],["HourlyRelativeHumidity"]) \
         .fillna(means["HourlySeaLevelPressure"],["HourlySeaLevelPressure"]) \
         .fillna(means["HourlyStationPressure"],["HourlyStationPressure"]) \
         .fillna(means["HourlyVisibility"],["HourlyVisibility"]) \
         .fillna(means["HourlyWetBulbTemperature"],["HourlyWetBulbTemperature"]) \
         .fillna(means["HourlyWindDirection"],["HourlyWindDirection"]) \
         .fillna(0,["HourlyWindSpeed"]) \
         .fillna("",["TAIL_NUM"])
         
    
    #string indexing of carrier
    carrier_indexer = StringIndexer(inputCol="OP_CARRIER", outputCol="OP_CARRIER_Index")
    fold_df = carrier_indexer.fit(fold_df).transform(fold_df)
    
    #one hot encoding
    onehotencoder_carrier_vector = OneHotEncoder(inputCol="OP_CARRIER_Index", outputCol="carrier_vec")
    fold_df = onehotencoder_carrier_vector.fit(fold_df).transform(fold_df)
    
    #vector assembler
    feature_cols = ['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK','Snow','carrier_vec']
    #assemble = VectorAssembler(inputCols=feature_cols, outputCol='features')
    #outputCol = "features"
    df_va = VectorAssembler(inputCols = feature_cols, outputCol = 'feature_vector')
    model_input = df_va.transform(fold_df)
    
    #rename delay flag to label
    model_input = model_input.withColumnRenamed("DEP_DEL15","label")
    #model_input = assemble.transform(fold_df) \
    #               .withColumnRenamed('DEP_DEL15', 'label')
    
    #scaling
    scaler=StandardScaler().setInputCol("feature_vector").setOutputCol("scaled_feature_vector")
    model_input = scaler.fit(model_input).transform(model_input)
    #model_input = model_input.select('label', 'scaled_feature_vector','cv')
    
    
    return model_input
display(process_fold_df(d['df2']))

# COMMAND ----------

d_processed = {}
for key in d.keys():
    print(key)
    d_processed[key] = process_fold_df(d[key])

# COMMAND ----------

# set up grid search: estimator, set of params, and evaluator
rf = RandomForestClassifier(labelCol="label", featuresCol="Scaled_features")
grid = ParamGridBuilder()\
            .addGrid(rf.maxDepth, [5, 10])\
            .addGrid(rf.numTrees, [10, 15])\
            .build()

# Example using F0.5 score for evaluator
evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=0.5)
#evaluator = MulticlassClassificationEvaluator(metricName='precisionByLabel')

# COMMAND ----------

#Logistic Regression

#train models with logistic regression
def fit_log_model(df):
    model = LogisticRegression(labelCol="label", featuresCol="scaled_feature_vector")
    lrn = model.fit(df)
    return lrn

#return model results of each fold
logistic_models = {}
for key in d_processed.keys():
    print(key)
    logistic_models[key] = fit_log_model(d_processed[key])
    lrn_summary = logistic_models[key].summary
    display(lrn_summary.predictions)

#evaluator = MulticlassClassificationEvaluator(metricName='precisionByLabel')
#evaluator = BinaryClassificationEvaluator(metricName='Precision')


#grid = ParamGridBuilder()\
#        .addGrid(model.threshold,[0.5,0.8])\
#        .build()

# COMMAND ----------

#Random Forest

#train models with random forest
def fit_forest_model(df, numTrees=10):
    rf = RandomForestClassifier(labelCol="label", featuresCol="feature_vector", numTrees=numTrees)
    lrn = rf.fit(df)
    return lrn

#return model results of each fold
forest_models = {}
for key in d_processed.keys():
    print(key)
    forest_models[key] = fit_forest_model(d_processed[key])
    lrn_summary = forest_models[key].summary
    display(lrn_summary.predictions)

# COMMAND ----------

#XGBoost - Needs to use larger dataset to work

#train models with XGBoost
def fit_xgboost_model(df, maxIter=10):
    xg = GBTClassifier(labelCol="label", featuresCol="feature_vector", maxIter=maxIter)
    lrn = xg.fit(df)
    return lrn

#return model results of each fold
xgboost_models = {}
for key in d_processed.keys():
    print(key)
    xgboost_models[key] = fit_xgboost_model(d_processed[key])
    lrn_summary = xgboost_models[key].summary
    display(lrn_summary.predictions)

# COMMAND ----------

#Neural Network (MLPC) - still troubleshooting

#train models with Multi Layer Neural Perceptron
def fit_MLPC_model(df, blockSize=128, seed=1234, layers = [4, 5, 4, 3], maxIter = 10):
    MLPC = MultilayerPerceptronClassifier(labelCol="label", featuresCol="scaled_feature_vector", maxIter=maxIter, layers=layers, blockSize=blockSize, seed=seed)
    lrn = MLPC.fit(df)
    return lrn

#return model results of each fold
MLPC_models = {}
for key in d_processed.keys():
    print(key)
    MLPC_models[key] = fit_MLPC_model(d_processed[key])
    result = MLPC_models[key].transform(d_processed[key])
    result.show(10)

# COMMAND ----------

# MAGIC %run "/Shared/w261_Section4_Group2/Phase 3/custom_cv_module"

# COMMAND ----------

display(d_processed['df1'])

# COMMAND ----------

# run cross validation & return the crossvalidation F0.5 score for 'test' set
cv = CustomCrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator,splitWord =('train','val'), cvCol = 'cv',parallelism=4)

# COMMAND ----------

cvModel = cv.fit(d_processed)

# COMMAND ----------

cvModel.bestModel

# COMMAND ----------

#precision by label 

cvModel.avgMetrics

# COMMAND ----------

# make predictions
predictions = cvModel.transform(d_processed['df1'])

display(predictions.groupby('label', 'prediction').count())

# COMMAND ----------

display(predictions)

# COMMAND ----------

predictions.show()

# COMMAND ----------

