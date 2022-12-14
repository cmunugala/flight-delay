# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The first of this notebook will have pre-join exploratory data analysis of the given data.

# COMMAND ----------

# MAGIC %md
# MAGIC #Loading all packages and datasets 

# COMMAND ----------

from pyspark.sql.functions import col,isnan,when,count
from pyspark.sql import functions as f
from pyspark.sql import SQLContext
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import isnan, when, count, col, isnull, percent_rank, avg
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
from pyspark.sql.functions import mean as _mean

from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 1- Prejoin EDA 

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
# display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# Inspect the Mount's Final Project folder 
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Init Script and SAS Token

# COMMAND ----------

# ignoring this for now. 
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

# displaying dataset in our storage blob
display(dbutils.fs.ls(f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading datasets

# COMMAND ----------

df_stations = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")
# display(df_stations)

airline_final_df = spark.read.parquet(f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net/premerge_airline_data")

weather_final_df = spark.read.parquet(f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net/premerge_weather_station_data")
#display(df_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Stations Dataset - EDA

# COMMAND ----------

# we have 2229 unique airport codes in the df_stations dataset. 
print(f" Total number of aiport codes :{df_stations.select('neighbor_call').distinct().count()}")

#import dataset
airport_codes_with_time_zones = pd.read_csv('https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat', header=None)
column_namess = {0: 'AirportID', 1: 'Name', 2: 'City', 3: 'Country', 4: 'IATA', 5: 'ICAO', 6: 'airport_latitude', 
             7: 'airport_longitude', 8: 'airport_elevation', 9: 'Timezone', 10: 'Daylight_savings_time', 11: 'TZ_Timezone', 12: 'Type', 13: 'Source'}
 
#add column names 
airport_codes_with_time_zones.rename(columns=column_namess, inplace=True)
#selecting desired columns 
codes = airport_codes_with_time_zones[['Country','IATA','ICAO','Timezone', 'TZ_Timezone']]
 
# found one airport in the final dataset wiht invalid timezone
# filling the invalid timezone with the correct timezone
codes.loc[codes['IATA'] == 'BIH', 'TZ_Timezone'] = 'America/Los_Angeles' 
 
#converting to PySpark Dataframe
airport_codes = spark.createDataFrame(codes)
 
#filtering stations data set with airport_codes dataset
stations_data_filtered = df_stations.join(airport_codes).where(df_stations["neighbor_call"] == airport_codes["ICAO"])

# COMMAND ----------

#filtering stations data set with airport_codes dataset and joining
stations_data_filtered = df_stations.join(airport_codes).where(df_stations["neighbor_call"] == airport_codes["ICAO"])

#investigating unique countries.
display(stations_data_filtered.groupBy('Country').count())

# COMMAND ----------

#selecting US, Puerto Rico, and Virgin Islands.
countries =['United States','Puerto Rico','Virgin Islands']
stations_data_filtered_US = stations_data_filtered.filter(stations_data_filtered.Country.isin(countries))
 
#selecting desired columns
cols_to_keeep = ['station_id', 'neighbor_name', 'neighbor_state','neighbor_call','IATA',
                 'distance_to_neighbor','Country', 'Timezone', 'TZ_timezone']
 
stations_data_us = stations_data_filtered_US.select(cols_to_keeep)
 
#selecting weather stations that are the closest to each airport.
minimum_distance = stations_data_us.groupby('neighbor_call').agg(min('distance_to_neighbor'))
f_airport_stations = stations_data_us.join(minimum_distance, ['neighbor_call'])
 
#final station data by joing. 
station_final_df = f_airport_stations.filter(f_airport_stations['distance_to_neighbor'] == f_airport_stations['min(distance_to_neighbor)'])
display(station_final_df)
station_final_df.count()

# COMMAND ----------

display(station_final_df.groupBy('IATA','TZ_timezone', 'neighbor_call' ).count())

# COMMAND ----------

# MAGIC %md
# MAGIC We have some values values as missig in IATA and TZ_timezone. We will revisit this once we perform a filter with airline data set. One joins have been completed, these null values will drop

# COMMAND ----------

# Aside from IATA and TZ_timezone having "\N" values, checking for nulls. 
df_Columns_s = station_final_df
station_final_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_Columns_s.columns]
   ).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC # Flight Dataset - EDA

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For data cleanup for the airline dataset, please refer to the Data_cleanup_and_join databricks notebook.

# COMMAND ----------

display(airline_final_df)
print(f' Airline dataset: number of rows = {airline_final_df.count()}, number of columns =  {len(airline_final_df.columns)}')

# COMMAND ----------

# There are some null values that will be dropped after joining
display(airline_final_df.groupBy('dest_IATA','dest_Timezone','dest_TZ','Date_Time_sched_arrival_utc').count())

# COMMAND ----------

airline_final_df.printSchema()

# COMMAND ----------

#The number of flights delayed. 
total = airline_final_df.count()
display(airline_final_df.groupBy('DEP_DEL15').count().withColumnRenamed('count', 'total_count').withColumn('Percentage', (f.col('total_count') / total) * 100))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC There is a class imbalance in our output variable. We have about %17 of flights that are delayed and over %82 of flights that are non delayed.

# COMMAND ----------

# The number of delays vs non-delays over the years
total = airline_final_df.count()
display(airline_final_df.groupBy('YEAR','DEP_DEL15').count().withColumnRenamed('count', 'total_count').withColumn('Percentage', (f.col('total_count') / total) * 100))

# COMMAND ----------

# MAGIC %md
# MAGIC It appears that the number of delays and non-delays are relatively stable. In 2020, we do see a huge dip in both due to the COVID lock downs.

# COMMAND ----------

#next we will investigate the Percentage of delays by airline
airline_delay_count_0 = airline_final_df.groupby('OP_CARRIER_AIRLINE_ID', 'DEP_DEL15').count().filter(airline_final_df.DEP_DEL15 != "1")
airline_delay_count_1 = airline_final_df.groupby('OP_CARRIER_AIRLINE_ID', 'DEP_DEL15').count().filter(airline_final_df.DEP_DEL15 != "0")
airline_delay_count_1 = airline_delay_count_1.selectExpr("OP_CARRIER_AIRLINE_ID as OP_CARRIER_AIRLINE_ID_1", "DEP_DEL15 as DEP_DEL15_1", "count as count_1")
airline_delay_count_0 = airline_delay_count_0.join(airline_delay_count_1,airline_delay_count_0.OP_CARRIER_AIRLINE_ID ==  airline_delay_count_1.OP_CARRIER_AIRLINE_ID_1,"inner")
airline_delay_count_0 = airline_delay_count_0.drop(airline_delay_count_0.OP_CARRIER_AIRLINE_ID_1)
airline_delay_count_0 = airline_delay_count_0.withColumn("relative_delay", (col("count_1") / (col("count")+col("count_1"))) * 100).orderBy(col('relative_delay').desc())
display(airline_delay_count_0)

# COMMAND ----------

# MAGIC %md
# MAGIC It appears that some airlines have more delays than others. 

# COMMAND ----------

#Flight delay by airport
airport_delay_count_0 = airline_final_df.groupby('ORIGIN', 'DEP_DEL15').count().filter(airline_final_df.DEP_DEL15 != "1")
airport_delay_count_1 = airline_final_df.groupby('ORIGIN', 'DEP_DEL15').count().filter(airline_final_df.DEP_DEL15 != "0")
airport_delay_count_1 = airport_delay_count_1.selectExpr("ORIGIN as ORIGIN_1", "DEP_DEL15 as DEP_DEL15_1", "count as count_1")
airport_delay_count_0 = airport_delay_count_0.join(airport_delay_count_1,airport_delay_count_0.ORIGIN ==  airport_delay_count_1.ORIGIN_1,"inner")
airport_delay_count_0 = airport_delay_count_0.drop(airport_delay_count_0.ORIGIN_1)
airport_delay_count_0 = airport_delay_count_0.withColumn("total", col("count")+col("count_1"))
airport_delay_count_0 = airport_delay_count_0.withColumn("relative_delay", (col("count_1") / (col("count")+col("count_1"))) * 100).orderBy(col('relative_delay').desc()).take(20)
display(airport_delay_count_0)

# COMMAND ----------

# Which 20 airports have the highest volume of flights?
display(airline_final_df.groupby('ORIGIN').count().orderBy(col('count').desc()).take(20))


# COMMAND ----------

#let's investigate the number of delays in the top of 20 airports by volume.

airport_delay_count_0 = airline_final_df.groupby('ORIGIN', 'DEP_DEL15').count().filter(airline_final_df.DEP_DEL15 != "1")
airport_delay_count_1 = airline_final_df.groupby('ORIGIN', 'DEP_DEL15').count().filter(airline_final_df.DEP_DEL15 != "0")
airport_delay_count_1 = airport_delay_count_1.selectExpr("ORIGIN as ORIGIN_1", "DEP_DEL15 as DEP_DEL15_1", "count as count_1")
airport_delay_count_0 = airport_delay_count_0.join(airport_delay_count_1,airport_delay_count_0.ORIGIN ==  airport_delay_count_1.ORIGIN_1,"inner")
airport_delay_count_0 = airport_delay_count_0.drop(airport_delay_count_0.ORIGIN_1)
airport_delay_count_0 = airport_delay_count_0.withColumn("total", col("count")+col("count_1"))
airport_delay_count_0 = airport_delay_count_0.withColumn("relative_delay", (col("count_1") / (col("count")+col("count_1"))) * 100).orderBy(col('relative_delay').desc())
airport_delay_count_00 = airport_delay_count_0.withColumn("relative_delay", col("count_1") / (col("count")+col("count_1"))).orderBy(col('total').desc()).take(20)
display(airport_delay_count_00)

# COMMAND ----------

#Next let's take a look at day of the month, day of the month and day of the week and see if there are any insights we can gain. 

display(airlines_final_df.groupBy('YEAR','QUARTER','MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_DEL15').count())

# COMMAND ----------

# MAGIC %md
# MAGIC Looking at the total number of delayed and non delayed on different timelines above, does not tell us much visually. Thera are minor difference in all timelines.

# COMMAND ----------

# Delay by Flight Distance

display(airlines_final_df.groupBy('DISTANCE_GROUP', 'DISTANCE', 'DEP_DEL15').count())

# COMMAND ----------

# MAGIC %md
# MAGIC It appears that the vast majority of flights are between 251-749 miles. We can also that the flights that are between 251-749 miles have the lowest cancellation rate.  

# COMMAND ----------

display(airlines_final_df.groupBy('origin_TZ', 'DEP_DEL15').count())

# COMMAND ----------

# MAGIC %md
# MAGIC It appears that most flights depart from a few major timezones

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Weather Dataset - EDA

# COMMAND ----------

# MAGIC %md
# MAGIC For clean up of the weather data, refer to the Data_cleanup_and_join notebook

# COMMAND ----------

display(weather_final_df)
print(f' Weather dataset: number of rows = {weather_final_df.count()}, number of columns =  {len(weather_final_df.columns)}')

# COMMAND ----------

#let's find the earliest and latest data from this dataset.
earliest_date_weather =weather_final_df.select(f.min(f.col("Date_Time_utc")).alias("MIN")).limit(1).collect()[0].MIN
latest_date_weather =weather_final_df.select(f.max(f.col("Date_Time_utc")).alias("MAX")).limit(1).collect()[0].MAX
print(f"Earliest data: {earliest_date_weather}")
print(f"Latest date: {latest_date_weather}")

# COMMAND ----------

# Checking for Nulls
df_Columns_w = weather_final_df.select([c for c in weather_final_df.columns if c not in {'Date_Time_utc', 'station_id', 'NAME', 'IATA'}])
weather_final_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_Columns_w.columns]).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Most columns have nulls and they will be dropped once the joins have been completed

# COMMAND ----------

weather_numerical_df = weather_final_df.select([c for c in weather_final_df.columns if c not in {'Date_Time_utc', 'station_id', 'NAME', 'IATA',
                                                                                                 'HourlyPresentWeatherType','HourlySkyConditions',
                                                                                                 'HourlyWindDirection','HourlyPressureChange'}])


weather_numerical_df = weather_numerical_df.withColumn("ELEVATION", weather_numerical_df["ELEVATION"].cast(IntegerType())) \
                                           .withColumn("HourlyAltimeterSetting", weather_numerical_df["HourlyAltimeterSetting"].cast(IntegerType())) \
                                           .withColumn("HourlyDewPointTemperature", weather_numerical_df["HourlyDewPointTemperature"].cast(IntegerType())) \
                                           .withColumn("HourlyDryBulbTemperature", weather_numerical_df["HourlyDryBulbTemperature"].cast(IntegerType())) \
                                           .withColumn("HourlyPrecipitation", weather_numerical_df["HourlyPrecipitation"].cast(IntegerType())) \
                                           .withColumn("HourlyPressureTendency", weather_numerical_df["HourlyPressureTendency"].cast(IntegerType())) \
                                           .withColumn("HourlyRelativeHumidity", weather_numerical_df["HourlyRelativeHumidity"].cast(IntegerType())) \
                                           .withColumn("HourlySeaLevelPressure", weather_numerical_df["HourlySeaLevelPressure"].cast(IntegerType())) \
                                           .withColumn("HourlyStationPressure", weather_numerical_df["HourlyStationPressure"].cast(IntegerType())) \
                                           .withColumn("HourlyVisibility", weather_numerical_df["HourlyVisibility"].cast(IntegerType())) \
                                           .withColumn("HourlyWetBulbTemperature", weather_numerical_df["HourlyWetBulbTemperature"].cast(IntegerType())) \
                                           .withColumn("HourlyWindGustSpeed", weather_numerical_df["HourlyWindGustSpeed"].cast(IntegerType())) \
                                           .withColumn("HourlyWindSpeed", weather_numerical_df["HourlyWindSpeed"].cast(IntegerType()))
                                            

display(weather_numerical_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC -From the numerical columns, it only appears that HourlyDewPointTemperature, HourlyDryBulbTemperature, and HourlyWetBulbTemperature are possibly normally ditributed.\
# MAGIC -HourlyPressureTendency and HourlyWindGustSpeed have the most number of nulls. \
# MAGIC -HourlyPrecipitation's most values are zero. 