# Databricks notebook source
# MAGIC %md
# MAGIC #Feature Engineering
# MAGIC ### Number of Previous Flights
# MAGIC ### Percent Previous Delayed
# MAGIC ### Prophet Percent Delayed

# COMMAND ----------

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

# MAGIC %md
# MAGIC 
# MAGIC # number of flights per prior_day for origin and destination 

# COMMAND ----------

# Join sum of number of flights and delays for a given airport to the date 2 days in the future

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
                                             f.col("ORIGIN").alias('ORIGIN2'), 'id', 'DEP_DEL15')\
                                              .groupby('FL_DATE_origin', "ORIGIN2")\
                                              .agg(f.count("id").alias("origin_flight_per_day"), f.sum('DEP_DEL15').alias('origin_delays_per_day'))
 
origin_per_day1 = origin_per_day.select('ORIGIN2','origin_flight_per_day', 'origin_delays_per_day', col("FL_DATE_origin"), 
                                        f.to_date(col("FL_DATE_origin"), "yyy-MM-dd").alias("date_origin"))
 
origin_per_day2 = origin_per_day1.select('ORIGIN2','origin_flight_per_day','origin_delays_per_day','FL_DATE_origin',
                                         col("date_origin"),f.date_add(col("date_origin"),2).alias("origin_date_plus_2"))
 
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
origin_per_day3 = origin_per_day2.filter(origin_per_day2.origin_date_plus_2 != '2021-01-01')



#for each destination
destination_per_day = airline_data_df.select(f.col("FL_DATE").alias("FL_DATE_Dest"),
                                              f.col("DEST").alias('DEST2'), 'id', 'DEP_DEL15')\
                                      .groupby('FL_DATE_Dest', "DEST2")\
                                      .agg(f.count("id").alias("dest_flight_per_day"), f.sum('DEP_DEL15').alias('dest_delays_per_day'))

destination_per_day1 = destination_per_day.select('DEST2','dest_flight_per_day','dest_delays_per_day', col('FL_DATE_Dest'), 
                                        f.to_date(col("FL_DATE_Dest"), "yyy-MM-dd").alias("date_dest"))
 
destination_per_day2 = destination_per_day1.select('DEST2','dest_flight_per_day','dest_delays_per_day','FL_DATE_Dest',
                                         col("date_dest"),f.date_add(col("date_dest"),2).alias("dest_date_plus_2"))
 
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
destination_per_day3 = destination_per_day2.filter(destination_per_day2.dest_date_plus_2 != '2021-01-01')


# join back to the original data. 
flight_origin1 =airline_data_df.join(origin_per_day3,
                                        (airline_data_df.FL_DATE == origin_per_day3.origin_date_plus_2) & 
                                        (airline_data_df.ORIGIN == origin_per_day3.ORIGIN2), how='left') \
                                        .drop('ORIGIN2', 'origin_date_plus_2','FL_DATE_origin','date_origin','id')
 
airline_df_plus_2day = flight_origin1.join(destination_per_day3, 
                                        (flight_origin1.FL_DATE == destination_per_day3.dest_date_plus_2) & 
                                        (flight_origin1.DEST == destination_per_day3.DEST2), how='left') \
                                .drop('DEST2', 'dest_date_plus_2','FL_DATE_Dest','date_dest')
 
display(airline_df_plus_2day)

# COMMAND ----------

# airline_df_plus_1day = airline_df_plus_1day.withColumn('origin_flight_per_day', airline_df_plus_1day.origin_flight_per_day.cast('double')) \
#                                           .withColumn('origin_delays_per_day', airline_df_plus_1day.origin_delays_per_day.cast('double')) \
#                                           .withColumn('dest_flight_per_day', airline_df_plus_1day.dest_flight_per_day.cast('double')) \
#                                           .withColumn('dest_delays_per_day', airline_df_plus_1day.dest_delays_per_day.cast('double'))

#airline_df_plus_1day.printSchema()

airline_df_plus_2day_delr = airline_df_plus_2day.withColumn('origin_percent_delayed', f.col('origin_delays_per_day') / f.col('origin_flight_per_day')) \
                                                .withColumn('dest_percent_delayed', f.col('dest_delays_per_day') / f.col('dest_flight_per_day')).cache()

# COMMAND ----------

display(airline_df_plus_2day_delr)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Prophet on Percent Delayed Flights

# COMMAND ----------

# create list of airports
airports = airline_df_plus_2day_delr.select('ORIGIN').distinct().toPandas()
airports_list = airports['ORIGIN'].tolist()
len(airports_list)

# COMMAND ----------

airports_list.remove('ENV')
airports_list.remove('EFD')
airports_list.remove('TKI')
airports_list.remove('YNG')
airports_list.remove('CDB')
airports_list.remove('ADK')
airports_list.remove('OWB')
airports_list.remove('BFM')
# airports_list.remove('FOD')
# airports_list.remove('DDC')
# airports_list.remove('VCT')
len(airports_list)

# COMMAND ----------

airline_prophet_df = airline_df_plus_2day_delr.filter(airline_df_plus_2day_delr['YEAR'] < 2022) \
                                              .select('ORIGIN', 'FL_DATE', 'origin_percent_delayed') \
                                              .withColumnRenamed('FL_DATE', 'ds') \
                                              .withColumnRenamed('origin_percent_delayed', 'y') \
                                              .dropDuplicates().sort('ORIGIN', 'ds').cache()

airline_prophet_df = airline_prophet_df.repartition('ORIGIN')

display(airline_prophet_df)

# COMMAND ----------

null_df = airline_prophet_df.groupBy('ORIGIN').agg(f.count(f.col('y').isNotNull().alias('valid_points'))).sort('ORIGIN')
display(null_df)

# COMMAND ----------

#Custom Holidays

c_holidays = pd.DataFrame({
  'holiday': 'travel',
  'ds': pd.to_datetime(['2015-1-1','2015-1-2','2015-1-19', '2015-5-25','2015-7-3', '2015-9-7','2015-10-12','2015-11-25','2015-11-26','2015-11-27','2015-12-24','2015-12-25','2015-12-26','2016-1-1', '2016-1-2','2016-1-18','2016-5-30','2016-7-4','2016-9-5','2016-11-24','2016-11-23','2016-11-24','2016-11-25',
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
                                        '2020-12-24','2020-12-25','2020-12-26']),
  'lower_window': 0,
  'upper_window': 1,
})

c_holidays['ds_upper']  = c_holidays['ds']

c_holidays

# COMMAND ----------

#Covid Lockdowns

lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
    {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
    {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
    {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'},
])
for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
lockdowns

# COMMAND ----------

holidays = pd.concat([c_holidays, lockdowns])
holidays

# COMMAND ----------

from prophet import Prophet

prophet_df = pd.DataFrame()

for air in airports_list:
  p_set = airline_prophet_df.filter(airline_prophet_df.ORIGIN == air)

  p_set_df = p_set.toPandas()
  
  model = Prophet(holidays = holidays, 
                  weekly_seasonality = True,
                  yearly_seasonality = True,
                  growth='linear')

  print(air)
  model.fit(p_set_df)

  #set periods to a large number to see window of uncertainty grow
  future_pd = model.make_future_dataframe(
      periods=200,
      include_history=True)

  # predict over the dataset
  forecast_pd = model.predict(future_pd)
  
  forecast_pd['IATA'] = air
  
  prophet_df = pd.concat([prophet_df, forecast_pd])


# COMMAND ----------

prophet_df

# COMMAND ----------

prophet_spark_df=spark.createDataFrame(prophet_df) 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Save to Blob

# COMMAND ----------

# #Checkpoint Airlines Dataset before join
prophet_spark_df.write.mode("overwrite").parquet(f"{blob_url}/prophet_delay_rate")

# COMMAND ----------

airline_df_plus_2day_delr.write.mode("overwrite").parquet(f"{blob_url}/number_flights_and_delay_rate")# airline_df_plus_1day_delr.write.mode("overwrite").parquet(f"{blob_url}/number_flights_and_delay_rate")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Prophet Example for Report Viz

# COMMAND ----------

# for report visualization

from prophet import Prophet

p_set = airline_prophet_df.filter(airline_prophet_df.ORIGIN == 'DFW')

p_set_df = p_set.toPandas()
  
model = Prophet(holidays = holidays,
                weekly_seasonality = True,
                yearly_seasonality = True,
                growth='linear')

model.fit(p_set_df)

  #set periods to a large number to see window of uncertainty grow
future_pd = model.make_future_dataframe(
      periods=100,
      include_history=True)

  # predict over the dataset
forecast_pd = model.predict(future_pd)

# COMMAND ----------

pd.plotting.register_matplotlib_converters()

predict_fig = model.plot(forecast_pd, xlabel='date', ylabel='percent_delayed')
display(predict_fig)

# COMMAND ----------

display(figures = model.plot_components(forecast_pd))

# COMMAND ----------

