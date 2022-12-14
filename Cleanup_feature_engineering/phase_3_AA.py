# Databricks notebook source
# MAGIC %md
# MAGIC #Loading all packages and datasets 

# COMMAND ----------

from pyspark.sql.functions import col,isnan,when,count
from pyspark.sql import functions as f
from pyspark.sql import SQLContext
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import isnan, when, count, col, isnull, percent_rank
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
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.linalg import Vectors
from pyspark.mllib.evaluation import MulticlassMetrics
import random as rnd
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import LogisticRegression


from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# COMMAND ----------

# data_BASE_DIR = "dbfs:/mnt/mids-w261/"
# display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# # Inspect the Mount's Final Project folder 
# # Please IGNORE dbutils.fs.cp("/mnt/mids-w261/datasets_final_project/stations_data/", "/mnt/mids-w261/datasets_final_project_2022/stations_data/", recurse=True)
# data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
# display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Init Script and SAS Token

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

# MAGIC %md
# MAGIC ### Loading Processed Datasets

# COMMAND ----------

display(dbutils.fs.ls(f"{blob_url}"))

# COMMAND ----------

# Read in training and test data

train_df = spark.read.parquet(f"{blob_url}/feature_engineered_data")
test_df = spark.read.parquet(f"{blob_url}/feature_engineered_data_test")

# COMMAND ----------

display(train_df)
display(train_df.count())

# COMMAND ----------

display(test_df)
display(test_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Advanced Feature Engineering

# COMMAND ----------

# Page Rank Scores
pagerank_df = spark.read.parquet(f"{blob_url}/pagerank_scores")
display(pagerank_df)

# COMMAND ----------

#TEST CELL TO DEMONSTRATE INDEXING -> OH ENCODING -> VECTOR ASSEMBLING -> SCALING -> FITTING
#Apply string indexer to carrier
ohe_df = spark.read.parquet(f"{blob_url}/feature_engineered_data")
carrier_indexer = StringIndexer(inputCol="OP_CARRIER", outputCol="OP_CARRIER_Index")
ohe_df = carrier_indexer.fit(ohe_df).transform(ohe_df)

#Apply string indexer to year
year_indexer = StringIndexer(inputCol="YEAR", outputCol="YEAR_Index")
ohe_df = year_indexer.fit(ohe_df).transform(ohe_df)

#Apply one-hot encode to carrier
onehotencoder_carrier_vector = OneHotEncoder(inputCol="OP_CARRIER_Index", outputCol="carrier_vec")
ohe_df = onehotencoder_carrier_vector.fit(ohe_df).transform(ohe_df)

#Apply one-hote encode to year
onehotencoder_year_vector = OneHotEncoder(inputCol="YEAR_Index", outputCol="YEAR_vec")
ohe_df = onehotencoder_year_vector.fit(ohe_df).transform(ohe_df)


#Convert all relevent columns to vector assembler
inputCols = ['YEAR_vec',
 'QUARTER',
 'MONTH',
 'DAY_OF_MONTH',
 'DAY_OF_WEEK',
 'carrier_vec',
 'DISTANCE',
 'DISTANCE_GROUP',
 'ELEVATION',
 'HourlyAltimeterSetting',
 'HourlyDewPointTemperature',
 'HourlyDryBulbTemperature',
 'HourlyPrecipitation',
 'HourlyRelativeHumidity',
 'HourlySeaLevelPressure',
 'HourlyStationPressure',
 'HourlyVisibility',
 'HourlyWetBulbTemperature',
 'HourlyWindDirection',
 'HourlyWindSpeed',
 'HourlyWindGustSpeed',
 'Rain',
 'Snow',
 'Thunder',
 'Fog',
 'Mist',
 'Freezing',
 'Blowing',
 'Smoke',
 'Drizzle',
 'Overcast',
 'Broken',
 'Scattered',
 'CloudySkyCondition']

outputCol = "features"
df_va = VectorAssembler(inputCols = inputCols, outputCol = outputCol)
ohe_df = df_va.transform(ohe_df)

#scale feature vector
Scalerizer=StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
ohe_df = Scalerizer.fit(ohe_df).transform(ohe_df)
ohe_df = ohe_df.select('DEP_DEL15', 'Scaled_features')
display(ohe_df)

#train logistic regression
lr = LogisticRegression(labelCol="DEP_DEL15")
lrn = lr.fit(ohe_df)

#display results
lrn_summary = lrn.summary
display(lrn_summary.predictions)
#display(ohe_df)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Broken Window Cross Validation

# COMMAND ----------

train_df.columns

# COMMAND ----------

display(working_df)

# COMMAND ----------

#Computes baseline model performance
def baseline_model(df, k_folds=5):
  
  # Calculate count of each dataframe rows
  each_len = df.count() // k_folds
  
  #initialize variables
  start = 1 
  stop = each_len
  i=0
  precision_list = []
  recall_list = []
  F1_list = []
  
  while i < k_folds:
    
    #take the range of rows for each fold to make train/validation split
    val_size = each_len/5
    this_fold_train = df.filter(col('Index').between(start,stop-val_size))
    this_fold_val = df.filter(col('Index').between(stop-val_size,stop))
    
    # convert dataframe to RDD
    val_RDD = this_fold_val.rdd.cache()
    
    #convert RDD to libsvm
    val_libsvm = val_RDD.map(lambda line: LabeledPoint(line[0],[line[1]]))
    
    # compute raw scores on the test set
    predictionAndLabels = val_libsvm.map(lambda lp: (float(rnd.randint(0, 1)), lp.label))

    # Instantiate metrics object
    metrics = MulticlassMetrics(predictionAndLabels)
    
    #calculate precision
    precision = metrics.precision(float(1))
    #print('Fold ' + str(i) + ' Precision: ' + str(precision))
    precision_list.append(precision)
    
    #calculate recall
    recall = metrics.recall(float(1))
    #print('Fold ' + str(i) + ' Recall: ' + str(recall))
    recall_list.append(recall)
    
    #caclculate F1
    F1 = metrics.fMeasure(float(1), beta=1.0)
    #print('Fold ' + str(i) + ' F1: '+ str(F1))
    F1_list.append(F1)
    
    #iterate through each fold window
    start += each_len
    stop += each_len
    i+=1
  
  #calculates validation performance across all folds
  avg_prec = sum(precision_list)/k_folds
  avg_recall = sum(recall_list)/k_folds
  avg_F1 = sum(F1_list)/k_folds
  
  #prints validation performance across all folds
  print('Average Baseline Precision: ', avg_prec)
  print('Average Baselin Recall: ', avg_recall)
  print('Average Baseline F1: ', avg_F1)
  
  return avg_prec, avg_F1
    
baseline_model(working_df)

# COMMAND ----------

#Computes baseline model performance
def baseline_model_test(df):
    
  # convert dataframe to RDD
  RDD = df.rdd.cache()

  #convert RDD to libsvm
  test_libsvm = RDD.map(lambda line: LabeledPoint(line[0],[line[1]]))

  # compute raw scores on the test set
  predictionAndLabels = test_libsvm.map(lambda lp: (float(rnd.randint(0, 1)), lp.label))

  # Instantiate metrics object
  metrics = MulticlassMetrics(predictionAndLabels)

  #calculate precision
  precision = metrics.precision(float(1))

  #calculate recall
  recall = metrics.recall(float(1))

  #caclculate F1
  F1 = metrics.fMeasure(float(1), beta=1.0)

  #prints validation performance across all folds
  print('Average Baseline Precision: ', precision)
  print('Average Baselin Recall: ', recall)
  print('Average Baseline F1: ', F1)
  
  return precision, F1
    
baseline_model_test(test_data)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Baseline Results (50/50 Guess)

# COMMAND ----------

# MAGIC %md
# MAGIC       
# MAGIC #####Performance metrics
# MAGIC ######Precision = \\(\frac{TP}{TP+FP}\\)
# MAGIC 
# MAGIC ######F1 = \\(\frac{2PrecisionRecall}{Precision+Recall} = \frac{2TP}{2TP+FP+FN}\\)

# COMMAND ----------

val_prec, val_F1 = baseline_model(working_df)
test_prec, test_F1 = baseline_model_test(test_data)

#results_pd = pd.DataFrame(columns=['Baseline Experiment', 'Input Feature Families', 'Valid Prec', 'Valid F1', 'Test Prec', 'Test F1'])

dict = {'Baseline Experiment':['Rule Based 50/50 Guess'],
        'Input Feature Families':['Categorical, Quantitative, Binary'],
        'Valid Prec':[val_prec],
        'Valid F1': [val_F1],
        'Test Prec': [test_prec],
        'Test F1': [test_F1]}
results_pd = pd.DataFrame(dict)
results_pd


#-- Experiment table with the following details per experiment:

#----- Baseline experiment

#---- The families of input features used

#----- For train/valid/test record the following in a Pandas DataFrame:

#---- List Metrics used along with: their Latex equations

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Feature Engineering and class imbalance

# COMMAND ----------

display(train_df)
display(train_df.count())

# COMMAND ----------

# null_values = train_df.select([count(when(col(c).isNull(), c)).alias(c) for c in train_df.columns]).toPandas()
# null_values
 

# COMMAND ----------

# null_values_test = test_df.select([count(when(col(c).isNull(), c)).alias(c) for c in test_df.columns]).toPandas()
# null_values_test

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## class imbalance 

# COMMAND ----------

#addressing class imbalance

def add_class_weight(data, verbose = False):
  """
  Adds a class weight feature to the data
  """
  
  delayed = data.filter(col("DEP_DEL15") == 1)
  
  ratio = delayed.count()/ data.count() 
  delayed_weight = 1-ratio
  non_delayed_weight = ratio
  
  class_balance_df = data.withColumn("DEP_DEL15_weighted", when(data.DEP_DEL15 == 1, delayed_weight)
                              .otherwise(non_delayed_weight))
  
  return class_balance_df


def oversample(data,verbose = False):
  """
  Over samples the minority class
  """
  
  delay_count = data.filter(f.col('DEP_DEL15') == 1 ).count()
  non_delay_count = data.filter(f.col('DEP_DEL15') == 0 ).count()
  #   print(f' total count : {data.count()}')
  #   print(f' delayed count : {delay_count}')
  #   print(f' non delayed count : {non_delay_count}')
  
  fraction_oversample = non_delay_count / delay_count
  #   print(f' non delayed / delayed: {fraction_oversample}')

  train_non_delay = data.filter(f.col('DEP_DEL15') == 0)
  #   print(f' non delayed count df : {train_non_delay.count()}')
  
  train_delay_oversampled = data.filter(f.col('DEP_DEL15') == 1).sample(withReplacement=True, fraction=fraction_oversample, seed = 261)
  #   print(f' oversampled delayed count : {train_delay_oversampled.count()}')
  
  data_oversampled = train_delay_oversampled.union(train_non_delay)
  #   print(f' train_df Oversampled : {train_oversampled.count()}')
  
  return data_oversampled


def undersample(data, verbose = False):
  """
  Under samples the majority class
  """
  
  delay_count = data.filter(f.col('DEP_DEL15') == 1 ).count()
  non_delay_count = data.filter(f.col('DEP_DEL15') == 0 ).count()
  #   print(f' total count : {data.count()}')
  #   print(f' delayed count : {delay_count}')
  #   print(f' non delayed count : {non_delay_count}')
  
  fraction_undersample = delay_count / non_delay_count
  #   print(f' delayed / non delayed: {fraction_undersample}')
  
  train_delayed = data.filter(f.col('DEP_DEL15') == 1)
  #   print(f' non delayed count df : {train_delayed.count()}')
  
  train_non_delay_undersample = data.filter(f.col('DEP_DEL15') == 0).sample(withReplacement=True, fraction=fraction_undersample, seed = 261)
  #   print(f' oversampled delayed count : {train_non_delay_undersample.count()}')
  
  data_undersampled = train_delayed.union(train_non_delay_undersample)
  #   print(f' train_df Oversampled : {train_undersampled.count()}')
  
  return data_undersampled
  


# COMMAND ----------

# aaa = add_class_weight(train_df)
# display(aaa)
# aaa.count()

# bbb = oversample(train_df)
# display(bbb)
# bbb.count()

# ccc = undersample(train_df)
# display(ccc)
# ccc.count()

  

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Creating Holiday Indicator feature

# COMMAND ----------

# Function to create Holiday range Indicator for the train and test sets.
# National holidays were taken from https://www.officeholidays.com/
# For thanksgiving and Christmas, +/- 1 days were also included as a holiday period due to travel.
# For New years one day after was also included as a holiday period due to travel. 

def holiday_indicator(data):
  """
  Adds a holiday range indicator.
  """
  
  if data == train_df:

    train_df1 = train_df.withColumn('FL_Date', concat(col('YEAR'), lit('-'), col('MONTH'), lit('-'), col('DAY_OF_MONTH')))
    train_df2 = train_df1.withColumn('holiday_period', expr("""CASE WHEN FL_Date in ('2015-1-1','2015-1-2','2015-1-19', '2015-5-25','2015-7-3',
                                        '2015-9-7','2015-10-12','2015-11-25','2015-11-26','2015-11-27','2015-12-24','2015-12-25','2015-12-26',
                                        
                                        '2016-1-1', '2016-1-2','2016-1-18','2016-5-30','2016-7-4',
                                        '2016-9-5','2016-11-24','2016-11-23','2016-11-24','2016-11-25',
                                        '2016-12-25','2016-12-26','2016-12-27',
                                        
                                        '2017-1-2','2017-1-3','2017-1-16','2017-5-29','2017-7-4',
                                        '2017-9-4','2017-11-22','2017-11-23','2017-11-24',
                                        '2017-12-24','2017-12-25','2017-12-26',
                                        
                                        '2018-1-1','2018-1-2','2018-1-15','2018-5-28','2018-7-4',
                                        '2018-9-3','2018-11-21','2018-11-22','2018-11-23',
                                        '2018-12-24','2018-12-25','2018-12-26',
                                        
                                        '2019-1-1','2019-1-2','2019-1-21','2019-5-27','2019-7-4',
                                        '2019-9-2','2019-11-27','2019-11-28','2019-11-29',
                                        '2019-12-24','2019-12-25','2019-12-26',
                                        
                                        '2020-1-1','2020-1-2','2020-1-20','2020-5-25','2020-7-3',
                                        '2020-9-7','2020-11-25','2020-11-26','2020-11-27',
                                        '2020-12-24''2020-12-25','2020-12-26') THEN '1' """ 
                                          "ELSE '0' END"))
    return train_df2
    
  if data == test_df:
    test_df1 = test_df.withColumn('FL_Date', concat(col('YEAR'), lit('-'), col('MONTH'), lit('-'), col('DAY_OF_MONTH')))

    test_df2 = test_df1.withColumn('holiday_period', expr("""CASE WHEN FL_Date in ('2021-1-1','2021-1-2','2021-1-18','2021-5-31',
                                                                        '2021-7-5','2021-9-6','2021-11-24','2021-11-25','2021-11-26',
                                                                        '2021-12-24','2021-12-25','2021-12-26') THEN '1' """ 
                                          "ELSE '0' END"))
    return test_df2
      
    
# display(holiday_indicator(train_df))
# display(holiday_indicator(test_df))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # number of flights per prior_day for origin and destination 

# COMMAND ----------

#reading from blob. 
airline_data_df= spark.read.parquet(f"{blob_url}/premerge_airline_data")

#getting rid of duplicates
# print(f' count before dropping duplicates: {airline_data_df.count()}') 
airline_data_df = airline_data_df.dropDuplicates()
# print(f' count after dropping duplicates: {airline_data_df.count()}')

#creating the new feature for total number of flights per prior_day for Origin and Destination Airports. 
airline_data_df = airline_data_df.withColumn("id", f.monotonically_increasing_id())


from pyspark.sql.functions import to_timestamp

#for each origin
origin_per_day = airline_data_df.select(f.col("FL_DATE").alias("FL_DATE_origin"), 
                                             f.col("ORIGIN").alias('ORIGIN2'), 'id')\
                                              .groupby('FL_DATE_origin', "ORIGIN2")\
                                              .agg(f.count("id").alias("origin_flight_per_day"))
 

origin_per_day = airline_data_df.select(f.col("FL_DATE").alias("FL_DATE_origin"), 
                                             f.col("ORIGIN").alias('ORIGIN2'), 'id')\
                                              .groupby('FL_DATE_origin', "ORIGIN2")\
                                              .agg(f.count("id").alias("origin_flight_per_day"))
 
origin_per_day1 = origin_per_day.select('ORIGIN2','origin_flight_per_day',col("FL_DATE_origin"), 
                                        f.to_date(col("FL_DATE_origin"), "yyy-MM-dd").alias("date_origin"))
 
origin_per_day2 = origin_per_day1.select('ORIGIN2','origin_flight_per_day','FL_DATE_origin',
                                         col("date_origin"),f.date_add(col("date_origin"),1).alias("origin_date_plus_1"))
 
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
origin_per_day3 = origin_per_day2.filter(origin_per_day2.origin_date_plus_1 != '2021-01-01')



#for each destination
destination_per_day = airline_data_df.select(f.col("FL_DATE").alias("FL_DATE_Dest"),
                                              f.col("DEST").alias('DEST2'), 'id')\
                                      .groupby('FL_DATE_Dest', "DEST2")\
                                      .agg(f.count("id").alias("dest_flight_per_day"))

destination_per_day1 = destination_per_day.select('DEST2','dest_flight_per_day',col('FL_DATE_Dest'), 
                                        f.to_date(col("FL_DATE_Dest"), "yyy-MM-dd").alias("date_dest"))
 
destination_per_day2 = destination_per_day1.select('DEST2','dest_flight_per_day','FL_DATE_Dest',
                                         col("date_dest"),f.date_add(col("date_dest"),1).alias("dest_date_plus_1"))
 
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
destination_per_day3 = destination_per_day2.filter(destination_per_day2.dest_date_plus_1 != '2021-01-01')


# join back to the original data. 
flight_origin1 =airline_data_df.join(origin_per_day3,
                                        (airline_data_df.FL_DATE == origin_per_day3.origin_date_plus_1) & 
                                        (airline_data_df.ORIGIN == origin_per_day3.ORIGIN2), how='left') \
                                        .drop('ORIGIN2', 'origin_date_plus_1','FL_DATE_origin','date_origin','id')
 
airline_df_plus_1day = flight_origin1.join(destination_per_day3, 
                                        (flight_origin1.FL_DATE == destination_per_day3.dest_date_plus_1) & 
                                        (flight_origin1.DEST == destination_per_day3.DEST2), how='left') \
                                .drop('DEST2', 'dest_date_plus_1','FL_DATE_Dest','date_dest')
 
display(airline_df_plus_1day)

# COMMAND ----------

# print(f'original before: {airline_data_df.count()}')
# print(f'no shifting one day: {airline_df_no_plus_1day.count()}')
# print(f'shifting one day: {airline_df_plus_1day.count()}')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 26 hours before scheduled flight average delay per carrier

# COMMAND ----------

airline_data_df_carrier = airline_df_plus_1day.drop('id')
partition = Window.partitionBy('OP_CARRIER')
 
parition1 = partition.orderBy(f.unix_timestamp('two_hrs_pre_flight_utc')).rangeBetween(-86400, -1)
 
airline_data_df_carrier1 = airline_data_df_carrier.withColumn('mean_carrier_delay', f.avg('DEP_DEL15').over(parition1))
display(airline_data_df_carrier1)

# COMMAND ----------

null_values = airline_data_df_carrier1.select([count(when(col(c).isNull(), c)).alias(c) for c in airline_data_df_carrier1.columns]).toPandas()
null_values