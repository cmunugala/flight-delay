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

# Read in training and test data

train_df = spark.read.parquet(f"{blob_url}/train_data_with_adv_features").cache()
test_df = spark.read.parquet(f"{blob_url}/test_data_with_adv_features")

# COMMAND ----------

#feature processing of dfs

train_df=train_df.select("*", f.row_number().over(Window.partitionBy().orderBy("Date_Time_sched_dep_utc")).alias("Index"))
train_df = train_df.withColumn("holiday_period", train_df["holiday_period"].cast(IntegerType()))
test_df = test_df.withColumn("holiday_period", test_df["holiday_period"].cast(IntegerType()))

# COMMAND ----------

#string indexing of carrier
carrier_indexer = StringIndexer(inputCol="OP_CARRIER", outputCol="OP_CARRIER_Index")
train_df = carrier_indexer.fit(train_df).transform(train_df)


#one hot encoding
onehotencoder_carrier_vector = OneHotEncoder(inputCol="OP_CARRIER_Index", outputCol="carrier_vec")
train_df = onehotencoder_carrier_vector.fit(train_df).transform(train_df)

# COMMAND ----------

#splitting training dataframe into five folds contained in dictionary "d"

d = {}
folds = ['df1','df2','df3','df4','df5']

each_len = train_df.count()/5
start = 1
val_size = each_len/5
stop = each_len
precision_list = []

for fold in folds:
    d[fold] = train_df.filter(col('Index').between(start,stop))\
                                  .withColumn('cv', F.when(col('Index').between(start,(stop-val_size)), 'train')
                                         .otherwise('val'))
    start += each_len
    stop += each_len

                                  

# COMMAND ----------

d['df1'].count()

# COMMAND ----------

train_df.createOrReplaceTempView('train_view')

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC SELECT holiday_period,mean_carrier_delay,Pagerank_Score,
# MAGIC  PREV_FLIGHT_DELAYED,origin_percent_delayed,
# MAGIC  dest_percent_delayed,
# MAGIC  ORIGIN_Prophet_trend,
# MAGIC  ORIGIN_Prophet_pred,
# MAGIC  DEST_Prophet_trend,
# MAGIC  DEST_Prophet_pred
# MAGIC  FROM train_view
# MAGIC  LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*)
# MAGIC FROM train_view
# MAGIC WHERE ORIGIN_Prophet_trend IS NULL

# COMMAND ----------

display(train_df)

# COMMAND ----------

train_df.columns

# COMMAND ----------

def process_fold_df(fold_df):
    
    
    #imputation
    fold_df.createOrReplaceTempView("fold_view")
    
    imputation_columns = ['CRS_ELAPSED_TIME','HourlyAltimeterSetting','HourlyDewPointTemperature',
             'HourlyDryBulbTemperature','HourlyRelativeHumidity','HourlySeaLevelPressure',
             'HourlyStationPressure','HourlyVisibility','HourlyWetBulbTemperature',
             'HourlyWindDirection','mean_carrier_delay','ORIGIN_Prophet_trend',
             'ORIGIN_Prophet_pred','DEST_Prophet_trend','DEST_Prophet_pred',]

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
         .fillna("",["TAIL_NUM"])\
         .fillna(0,['holiday_period'])\
         .fillna(means['mean_carrier_delay'],['mean_carrier_delay'])\
         .fillna(0,['PREV_FLIGHT_DELAYED'])\
         .fillna(0,['origin_percent_delayed'])\
         .fillna(0,['dest_percent_delayed'])\
         .fillna(means['ORIGIN_Prophet_trend'],['ORIGIN_Prophet_trend'])\
         .fillna(means['ORIGIN_Prophet_pred'],['ORIGIN_Prophet_pred'])\
         .fillna(means['DEST_Prophet_trend'],['DEST_Prophet_trend'])\
         .fillna(means['DEST_Prophet_pred'],['DEST_Prophet_pred'])
         

    
    #vector assembler
    feature_cols = ['MONTH','DAY_OF_MONTH','DAY_OF_WEEK','DISTANCE','HourlyWindSpeed','Rain','Blowing','Snow','Thunder','CloudySkyCondition','carrier_vec',         'holiday_period','mean_carrier_delay','Pagerank_Score','PREV_FLIGHT_DELAYED','origin_percent_delayed','dest_percent_delayed','ORIGIN_Prophet_trend','ORIGIN_Prophet_pred','DEST_Prophet_trend','DEST_Prophet_pred']
    #assemble = VectorAssembler(inputCols=feature_cols, outputCol='features')
    #outputCol = "features"
    df_va = VectorAssembler(inputCols = feature_cols, outputCol = 'feature_vector')
    model_input = df_va.transform(fold_df)
    
    #rename delay flag to label
    model_input = model_input.withColumnRenamed("DEP_DEL15","label")
    #model_input = assemble.transform(fold_df) \
    #               .withColumnRenamed('DEP_DEL15', 'label')
    
    #scaling
    #scaler=StandardScaler().setInputCol("feature_vector").setOutputCol("scaled_feature_vector")
    #model_input = scaler.fit(model_input).transform(model_input)
    #model_input = model_input.select('label', 'scaled_feature_vector','cv')
    
    
    return model_input

# COMMAND ----------

d_processed = {}
for key in d.keys():
    print(key)
    d_processed[key] = process_fold_df(d[key])

# COMMAND ----------

display(d_processed['df1'])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Model Building

# COMMAND ----------

# MAGIC %run "/Shared/w261_Section4_Group2/Phase 3/custom_cv_module"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Logistic Regression

# COMMAND ----------

# set up grid search: estimator, set of params, and evaluator
logistic_model = LogisticRegression(labelCol="label", featuresCol="scaled_feature_vector")
grid = ParamGridBuilder()\
            .addGrid(logistic_model.threshold, [0.3,0.5,0.8])\
            .build() 

# Example using F0.5 score for evaluator
evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=0.5, metricLabel=1)

# COMMAND ----------

# run cross validation & return the crossvalidation F0.5 score for 'test' set
cv = CustomCrossValidator(estimator=logistic_model, estimatorParamMaps=grid, evaluator=evaluator,splitWord =('train','val'), cvCol = 'cv',parallelism=10)

# COMMAND ----------

cvModel = cv.fit(d_processed)

# COMMAND ----------

#for individual testing

#test_train = d_processed['df1'].filter(col('cv')=='train')
#test_val = d_processed['df1'].filter(col('cv')=='val')

#test_logistic_model = LogisticRegression(labelCol="label", featuresCol="scaled_feature_vector")
#evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
#lrModel = test_logistic_model.fit(test_train)
#predictions = lrModel.transform(test_val)
#evaluator.evaluate(predictions)

# COMMAND ----------

# make predictions
predictions = cvModel.transform(d_processed['df1'])

display(predictions.groupby('label', 'prediction').count())

# COMMAND ----------

fbeta = cvModel.avgMetrics[0]
print(f"Logistic Regression F0.5 Score: {fbeta}")

# COMMAND ----------

# Read in training and test data

train_df = spark.read.parquet(f"{blob_url}/train_data_with_adv_features").cache()
test_df = spark.read.parquet(f"{blob_url}/test_data_with_adv_features")

# COMMAND ----------

#string indexing of carrier for train
carrier_indexer = StringIndexer(inputCol="OP_CARRIER", outputCol="OP_CARRIER_Index")
indexer_transformer = carrier_indexer.setHandleInvalid("keep").fit(train_df)
train_df = indexer_transformer.transform(train_df)

#one hot encoding for train
onehotencoder_carrier_vector = OneHotEncoder(inputCol="OP_CARRIER_Index", outputCol="carrier_vec")
onehotencoder_transformer = onehotencoder_carrier_vector.fit(train_df)
train_df = onehotencoder_transformer.transform(train_df)

# COMMAND ----------

display(train_df)

# COMMAND ----------

#string indexing of carrier for test
#one hot encoding for test
test_df = indexer_transformer.transform(test_df)
test_df = onehotencoder_transformer.transform(test_df)

# COMMAND ----------

#cast holiday to integer
train_df = train_df.withColumn("holiday_period", train_df["holiday_period"].cast(IntegerType()))
test_df = test_df.withColumn("holiday_period", test_df["holiday_period"].cast(IntegerType()))

# COMMAND ----------

processed_train_df = process_fold_df(train_df)

#scale to train on train set
scaler=StandardScaler().setInputCol("feature_vector").setOutputCol("scaled_feature_vector")
scaler_transformer = scaler.fit(processed_train_df)
processed_train_df = scaler_transformer.transform(processed_train_df)

processed_test_df = process_fold_df(test_df)
#scale to train on test set
processed_test_df = scaler_transformer.transform(processed_test_df)

# COMMAND ----------

final_logistic_model = LogisticRegression(labelCol="label", featuresCol="scaled_feature_vector")
evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=0.5, metricLabel=1)
lrModel = final_logistic_model.fit(processed_train_df)

# COMMAND ----------

predictions = lrModel.transform(processed_test_df)

# COMMAND ----------

display(predictions.groupby('label', 'prediction').count())

# COMMAND ----------

evaluator.evaluate(predictions)

# COMMAND ----------

feature_cols = ['MONTH','DAY_OF_MONTH','DAY_OF_WEEK','DISTANCE','HourlyWindSpeed','Rain','Blowing','Snow','Thunder','CloudySkyCondition','carrier_vec',         'holiday_period','mean_carrier_delay','Pagerank_Score','PREV_FLIGHT_DELAYED','origin_percent_delayed','dest_percent_delayed','ORIGIN_Prophet_trend','ORIGIN_Prophet_pred','DEST_Prophet_trend','DEST_Prophet_pred']

# COMMAND ----------

lrModel.coefficients

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
# MAGIC ### Advanced Modeling - May Not Use Whats Below This!

# COMMAND ----------

print(processed_train_df)

# COMMAND ----------

#Multi Layer Perceptron Grid Search Hyperparameter selection

# Read in training and test data

train_df = spark.read.parquet(f"{blob_url}/train_data_with_adv_features").cache()
test_df = spark.read.parquet(f"{blob_url}/test_data_with_adv_features")

# set up grid search: estimator, set of params, and evaluator
MLPC_model = MultilayerPerceptronClassifier(labelCol="label", featuresCol="scaled_feature_vector")
grid = ParamGridBuilder()\
            .addGrid(MLPC_model.maxIter, [50,100,200])\
            .addGrid(MLPC_model.layers, [[38,26,2],[38,26,26,2]])\
            .addGrid(MLPC_model.blockSize, [32, 64])\
            .addGrid(MLPC_model.solver, ['gd', 'l-bfgs'] )
            .build() 

# Example using F0.5 score for evaluator
evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=0.5, metricLabel=1)

# run cross validation & return the crossvalidation F0.5 score for 'validation' set
cv = CustomCrossValidator(estimator=logistic_model, estimatorParamMaps=grid, evaluator=evaluator,splitWord =('train','val'), cvCol = 'cv',parallelism=10)

#run to select best model
cvModel = cv.fit(d_processed)


# COMMAND ----------

#Neural Network (MLPC) -

#train models with Multi Layer Neural Perceptron
def fit_MLPC_model(df, blockSize=128, seed=1234, layers = [4, 5, 4, 2], maxIter = 10):
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