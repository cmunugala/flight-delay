# Databricks notebook source
# MAGIC %md
# MAGIC #Feature Engineering
# MAGIC ###Last Flight delayed

# COMMAND ----------

# MAGIC %md
# MAGIC ###Loading all packages and datasets 

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


from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# Inspect the Mount's Final Project folder 
# Please IGNORE dbutils.fs.cp("/mnt/mids-w261/datasets_final_project/stations_data/", "/mnt/mids-w261/datasets_final_project_2022/stations_data/", recurse=True)
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

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
# MAGIC ### Loading datasets

# COMMAND ----------

df_stations = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")
# display(df_stations)

df_airlines = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data/")
# display(df_airlines)

df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data/")
#display(df_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Stations Dataset - Cleaning and EDA

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC To begin cleaning up the station data set, we will first import a new dataset that will give up airport codes and timezones as follows. 

# COMMAND ----------

#reference: https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat
#dictionary: https://openflights.org/data.html

#import dataset
airport_codes_with_time_zones = pd.read_csv('https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat', header=None)
column_namess = {0: 'AirportID', 1: 'Name', 2: 'City', 3: 'Country', 4: 'IATA', 5: 'ICAO', 6: 'airport_latitude', 
             7: 'airport_longitude', 8: 'airport_elevation', 9: 'Timezone', 10: 'Daylight_savings_time', 11: 'TZ_Timezone', 12: 'Type', 13: 'Source'}

#add column names 
airport_codes_with_time_zones.rename(columns=column_namess, inplace=True)
#selecting desired columns 
codes = airport_codes_with_time_zones[['Country','IATA','ICAO','Timezone', 'TZ_Timezone']]

# found one airport in the final dataset wiht invalid timezone
codes.loc[codes['IATA'] == 'BIH', 'TZ_Timezone'] = 'America/Los_Angeles' 

#converting to PySpark Dataframe
airport_codes = spark.createDataFrame(codes)

#filtering stations data set with airport_codes dataset
stations_data_filtered = df_stations.join(airport_codes).where(df_stations["neighbor_call"] == airport_codes["ICAO"])

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

#final station data
station_final_df = f_airport_stations.filter(f_airport_stations['distance_to_neighbor'] == f_airport_stations['min(distance_to_neighbor)'])
display(station_final_df)
station_final_df.count()


# COMMAND ----------

# airport_codes_with_time_zones

# COMMAND ----------

#codes[codes['IATA'] == 'BIH']

# COMMAND ----------

# station_test = station_final_df.filter(station_final_df.IATA == 'BIH')

# display(station_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Flight Dataset - Cleaning and EDA

# COMMAND ----------

#checking the number of delayed, cancelled, and diverted flights.   
display(df_airlines.groupBy('DEP_DEL15', 'CANCELLED', 'DIVERTED').count())

# COMMAND ----------

# we will drop all the null values for the delayed variable. 
# for this analysis, we will further drop all cancelled flights since they make up a tiny portion of the dataset. 
df_airline1 = df_airlines.where(f.col('DEP_DEL15').isNotNull()).filter(df_airlines.CANCELLED != "1")

# Furthermore, we will only select the columns that are relevant for our analysis 
 
columns_to_keep = ['YEAR', 'QUARTER', 'MONTH','DAY_OF_MONTH', 'DAY_OF_WEEK', 
                   'FL_DATE','OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID','OP_CARRIER',
                   'TAIL_NUM','ORIGIN','DEST', 'CRS_DEP_TIME', 'DEP_DEL15', 'CRS_ELAPSED_TIME',
                   'DISTANCE', 'DISTANCE_GROUP', 'CRS_ARR_TIME','ARR_DELAY']
 
# unused columns, ensure no leakage, remove columns that give information after departure
'''
                   'OP_CARRIER_FL_NUM','ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID',
                   'ORIGIN_CITY_MARKET_ID', 'ORIGIN_CITY_NAME','ORIGIN_STATE_ABR',
                   'ORIGIN_STATE_FIPS','ORIGIN_STATE_NM','ORIGIN_WAC', 'DEST_AIRPORT_ID',
                   'DEST_AIRPORT_SEQ_ID','DEST_CITY_MARKET_ID', 'DEST_CITY_NAME',
                   'DEST_STATE_ABR','DEST_STATE_FIPS','DEST_STATE_NM','DEST_WAC',
                   'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DELAY_GROUP','DEP_TIME_BLK','TAXI_OUT', 'WHEELS_OFF',
                   'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME','ARR_TIME','ARR_DELAY',
                   'ARR_DELAY_NEW','ARR_DEL15','ARR_DELAY_GROUP','ARR_TIME_BLK', 'ACTUAL_ELAPSED_TIME','AIR_TIME','FLIGHTS',
                   'DIVERTED', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'
'''  
  
df_airline3 = df_airline1.select(columns_to_keep)
 
#compared to our original data set, we have the following column count and row count 
print(f'Original Dataset : number of rows: {df_airlines.count()}, number of columns: {len(df_airlines.columns)}')
print(f'airline dataset after delayed Null values dropped : number of rows: {df_airline3.count()}, number of columns: {len(df_airline3.columns)}')

#finding all the unique airports that are in origin and dest columns in airline dataset
o_airports = df_airline3.select("ORIGIN").distinct().collect()
d_airports = df_airline3.select("DEST").distinct().collect()
combined_airports = set([i["ORIGIN"] for i in o_airports] + [j["DEST"] for j in d_airports])
airport_unique_combined = spark.createDataFrame([[a] for a in sorted(combined_airports)], ["airport"])
print(f'unique airport codes in origin and destination in airline dataset: {airport_unique_combined.count()}')


#airport codse that are in both station_final_df and airline dataset
airports = {a["airport"] for a in airport_unique_combined.select("airport").distinct().collect()}
stations_unique_airport_code = station_final_df.filter(station_final_df.IATA.isin(airports))
print(f'Airport codes that are in station and flight datasets = {stations_unique_airport_code.count()}')


#airport codes not found in airline dataset
airports_in_station_flight = {i["IATA"] for i in stations_unique_airport_code.select("IATA").distinct().collect()}
airports_not_found_in_flight = airports - airports_in_station_flight
print(f"Airports unaccounted for : {','.join(airports_not_found_in_flight)}")

# COMMAND ----------

#creating new columns for origin and destination timeszones.
station_final_df.select(["IATA", "Timezone", "TZ_Timezone"]).createOrReplaceTempView("timezones")
df_airline3.createOrReplaceTempView("air")

airlines_origin_timezone = f"""
SELECT * FROM 
(SELECT * FROM air) AS a
LEFT JOIN 
(SELECT IATA AS Origin_IATA, Timezone AS origin_Timezone, TZ_Timezone AS origin_TZ FROM timezones) AS t
ON a.ORIGIN = t.origin_IATA
"""
df_airline3 = spark.sql(airlines_origin_timezone)


station_final_df.select(["IATA", "Timezone", "TZ_Timezone"]).createOrReplaceTempView("timezones")
df_airline3.createOrReplaceTempView("air")

airlines_dest_timezone = f"""
SELECT * FROM 
(SELECT * FROM air) AS a
LEFT JOIN 
(SELECT IATA AS dest_IATA, Timezone AS dest_Timezone, TZ_Timezone AS dest_TZ FROM timezones) AS t
ON a.DEST = t.dest_IATA
"""
df_airline3 = spark.sql(airlines_dest_timezone)
display(df_airline3)

# COMMAND ----------

#Creates date-time formatted column
df_airline3 =df_airline3.withColumn("CRS_DEP_TIME",format_string("%04d","CRS_DEP_TIME"))
df_airline3 = df_airline3.withColumn("Date_Time_sched_dep", concat(substring(col("FL_DATE"), 1, 10), lit(' '), substring(col("CRS_DEP_TIME"), 1, 2), lit(':00:00')))

df_airline3 = df_airline3.withColumn("CRS_ARR_TIME",format_string("%04d","CRS_ARR_TIME"))
df_airline3 = df_airline3.withColumn("Date_Time_sched_arrival", concat(substring(col("FL_DATE"), 1, 10), lit(' '), substring(col("CRS_ARR_TIME"), 1, 2), lit(':00:00')))

#Creates date-time column 2 hours before scheduled flight
df_airline4 = df_airline3.withColumn('two_hrs_pre_flight', df_airline3.Date_Time_sched_dep - expr('INTERVAL 2 HOURS'))

#Converts to UTC
airline_final_df = df_airline4.withColumn('two_hrs_pre_flight_utc', to_utc_timestamp(df_airline4.two_hrs_pre_flight, df_airline4.origin_TZ))
airline_final_df = airline_final_df.withColumn('Date_Time_sched_dep_utc', to_utc_timestamp(airline_final_df.Date_Time_sched_dep, df_airline4.origin_TZ))
airline_final_df = airline_final_df.withColumn('Date_Time_sched_arrival_utc', to_utc_timestamp(airline_final_df.Date_Time_sched_arrival, df_airline4.dest_TZ))
display(airline_final_df)

# COMMAND ----------

airline_final_df.count()

# COMMAND ----------

airline_final_df = airline_final_df.dropDuplicates()
airline_final_df.count()

# COMMAND ----------

# airline_final_df = airline_final_df.orderBy("TAIL_NUM","FL_DATE", "DATE_TIME_sched_dep_")

# display(airline_final_df)

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id, lag
from pyspark.sql.window import Window

partition=Window.partitionBy("TAIL_NUM").orderBy("FL_DATE", "DATE_TIME_sched_dep_utc")

# COMMAND ----------

prev_delay_df = airline_final_df.withColumn('PREV_FL_DATE', lag("FL_DATE", 1).over(partition)) \
                                .withColumn('PREV_sched_arrival_utc', lag("Date_Time_sched_arrival_utc", 1).over(partition)) \
                                .withColumn('PREV_TAIL_NUM', lag("TAIL_NUM", 1).over(partition)) \
                                .withColumn('PREV_DEST', lag("DEST", 1).over(partition)) \
                                .withColumn('PREV_ARR_DELAY', lag("ARR_DELAY", 1).over(partition)) \
                                .withColumn('PREV_DELAY', lag("DEP_DEL15", 1).over(partition)).cache()
display(prev_delay_df)

# COMMAND ----------

prev_delay_df.columns# prev_delay_df.columns

# COMMAND ----------

columns_to_select = [
 'FL_DATE',
 'TAIL_NUM',
 'ORIGIN',
 'DEST',
 'DEP_DEL15',
 'Date_Time_sched_dep_utc',
 'Date_Time_sched_arrival_utc',
 'two_hrs_pre_flight_utc',
 'PREV_FL_DATE',
 'PREV_sched_arrival_utc',
 'ARR_DELAY',
 'PREV_TAIL_NUM',
 'PREV_ARR_DELAY',
 'PREV_DEST',
 'PREV_DELAY']

# COMMAND ----------

prev_delay_2_df = prev_delay_df.select(columns_to_select)
display(prev_delay_2_df)
prev_delay_2_df.count()

# COMMAND ----------

prev_delay_2_df = prev_delay_2_df.na.fill(0,["ARR_DELAY"]) \
                                 .na.fill(0,["PREV_ARR_DELAY"]) \
                                 .na.fill(0,["PREV_DELAY"])
display(prev_delay_2_df)

# COMMAND ----------

prev_delay_3_df = prev_delay_2_df.withColumn('TAIL_NUM_CHECK', prev_delay_2_df['TAIL_NUM'] == prev_delay_2_df['PREV_TAIL_NUM']) \
                             .withColumn('PREV_DEST_CHECK', prev_delay_2_df['ORIGIN'] == prev_delay_2_df['PREV_DEST']) \
                             .withColumn('DATE_CHECK', (prev_delay_2_df['Date_Time_sched_dep_utc'].cast('long') - prev_delay_2_df['PREV_sched_arrival_utc'].cast('long'))/60).cache()

prev_delay_3_df = prev_delay_3_df.na.fill(0,["DATE_CHECK"])

display(prev_delay_3_df)

# COMMAND ----------

prev_delay_4_df = prev_delay_3_df.withColumn('DATE_CHECK_0', prev_delay_3_df['DATE_CHECK'] >= 0) \
                                .withColumn('DATE_CHECK_1', ((prev_delay_3_df['PREV_ARR_DELAY'].cast('int') + 90) >= prev_delay_3_df['DATE_CHECK'].cast('int'))).cache()

display(prev_delay_4_df)

# COMMAND ----------

prev_delay_5_df = prev_delay_4_df.withColumn('PREV_FLIGHT_DELAYED', prev_delay_4_df['PREV_DELAY'].cast('int') * prev_delay_4_df['TAIL_NUM_CHECK'].cast('int')  * prev_delay_4_df['PREV_DEST_CHECK'].cast('int')  * prev_delay_4_df['DATE_CHECK_0'].cast('int')  * prev_delay_4_df['DATE_CHECK_1'].cast('int'))

display(prev_delay_5_df)

# COMMAND ----------

prev_delay_5_df.columns

# COMMAND ----------

columns_sel = ['TAIL_NUM',
 'ORIGIN',
 'two_hrs_pre_flight_utc',
 'DEP_DEL15',
 'PREV_FLIGHT_DELAYED']

# COMMAND ----------

prev_delay_5_filtered_df = prev_delay_5_df.select(columns_sel)

prev_delay_5_filtered_df= prev_delay_5_filtered_df.na.fill(0,["PREV_FLIGHT_DELAYED"]) \
                                                  .withColumnRenamed('ORIGIN', 'PREV_ORIGIN') \
                                                  .withColumnRenamed('TAIL_NUM', 'PREV_TAIL_NUM') \
                                                  .withColumnRenamed('two_hrs_pre_flight_utc', 'PREV_two_hrs_pre_flight_utc') \
                                                  .withColumnRenamed('DEP_DEL15', 'PREV_DEP_DEL15')

display(prev_delay_5_filtered_df)

# COMMAND ----------

prev_delay_5_filtered_df.groupBy('PREV_DEP_DEL15').count().show()

# COMMAND ----------

prev_delay_5_filtered_df.groupBy('PREV_FLIGHT_DELAYED').count().show()

# COMMAND ----------

prev_delay_5_filtered_df.stat.corr('PREV_FLIGHT_DELAYED', 'PREV_DEP_DEL15')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Blob Storage Pre-merge Checkpoint

# COMMAND ----------

# #Checkpoint Airlines Dataset before join
prev_delay_5_filtered_df.write.mode("overwrite").parquet(f"{blob_url}/previous_flight_delayed")






# COMMAND ----------

display(dbutils.fs.ls(f"{blob_url}"))

# COMMAND ----------

