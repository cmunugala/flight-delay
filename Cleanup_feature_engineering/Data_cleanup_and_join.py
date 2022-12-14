# Databricks notebook source
# MAGIC %md
# MAGIC #Final Joining and Cleaning
# MAGIC #####Section 4 Group 2

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Pipeline and CleanUp

# COMMAND ----------

# MAGIC %md
# MAGIC ###Loading all packages

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
# MAGIC # Loading datasets

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
# MAGIC #Stations Dataset - Clean up

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC To begin cleaning up the station data set, we will first import a new dataset that will give up airport codes and timezones as follows. 

# COMMAND ----------

display(df_stations)
display(df_stations.count())

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

# MAGIC %md
# MAGIC # Flight Dataset - Clean up

# COMMAND ----------

display(df_airlines)
display(df_airlines.count())

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
                   'TAIL_NUM','ORIGIN','DEST', 'CRS_DEP_TIME','CRS_ARR_TIME', 'ARR_DELAY','DEP_DEL15', 'CRS_ELAPSED_TIME',
                   'DISTANCE', 'DISTANCE_GROUP']
 
# unused columns, ensure no leakage, remove columns that give information after departure
'''
                   'OP_CARRIER_FL_NUM','ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID',
                   'ORIGIN_CITY_MARKET_ID', 'ORIGIN_CITY_NAME','ORIGIN_STATE_ABR',
                   'ORIGIN_STATE_FIPS','ORIGIN_STATE_NM','ORIGIN_WAC', 'DEST_AIRPORT_ID',
                   'DEST_AIRPORT_SEQ_ID','DEST_CITY_MARKET_ID', 'DEST_CITY_NAME',
                   'DEST_STATE_ABR','DEST_STATE_FIPS','DEST_STATE_NM','DEST_WAC',
                   'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DELAY_GROUP','DEP_TIME_BLK','TAXI_OUT', 'WHEELS_OFF',
                   'WHEELS_ON', 'TAXI_IN','ARR_TIME','ARR_DELAY',
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

#dropping duplicate rows:
print(f'with duplicates: {df_airline3.count()}')
df_airline3 = df_airline3.dropDuplicates()
print(f'duplicates were dropped: {df_airline3.count()}')

# COMMAND ----------

#Creates date-time formatted column
df_airline3 = df_airline3.withColumn("CRS_DEP_TIME",format_string("%04d","CRS_DEP_TIME"))
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

# check for failed UTC conversions

# invalid_utc = airline_final_df.filter(airline_final_df.two_hrs_pre_flight_utc.isNull())
# display(invalid_utc)
# invalid_utc.count()

# COMMAND ----------

# # check origins with invalid time conversions
# display(invalid_utc.select('ORIGIN').distinct())

# COMMAND ----------

# check invalids for a certain airport
# display(invalid_utc.filter(invalid_utc.ORIGIN == 'IFP'))# # check invalids for a certain airport
# display(invalid_utc.filter(invalid_utc.ORIGIN == 'IFP'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Weather Dataset - Clean Up

# COMMAND ----------

df_weather.count()

# COMMAND ----------

# we only want stations in the airline and stations

station_final_379_df = station_final_df.filter(station_final_df.IATA.isin(airports))
display(station_final_379_df)
station_final_379_df.count()

# COMMAND ----------

# to start we will filter the weather data based on its STATION with the station_id from df_final_station dataset. 
# next we will filter unique station id from the station data set and filter the weather data
# also filter to REPORT_TYPE = FM-15, these are the hourly routine weather conditions at air terminals
 
station_ids = set(x["station_id"] for x in station_final_379_df.select(col("station_id")).distinct().collect())
weather_full_filtered = df_weather.filter(df_weather.STATION.isin(station_ids)) \
                                  .filter(df_weather['REPORT_TYPE'] == 'FM-15') \
                                  .withColumnRenamed('STATION', 'station_id')

print(f'Full weather dataset row count: {df_weather.count()}')
print(f'Weather data row count after filter: {weather_full_filtered.count()}')


# COMMAND ----------

#keeping only the columns that we need
coloumns_to_keep = ['station_id','DATE','ELEVATION','NAME','HourlyAltimeterSetting','HourlyDewPointTemperature',
 'HourlyDryBulbTemperature','HourlyPrecipitation','HourlyPresentWeatherType','HourlyPressureChange',
 'HourlyPressureTendency','HourlyRelativeHumidity','HourlySkyConditions','HourlySeaLevelPressure',
 'HourlyStationPressure','HourlyVisibility','HourlyWetBulbTemperature','HourlyWindDirection',
 'HourlyWindGustSpeed','HourlyWindSpeed']
 
weather_full_filtered_columns = weather_full_filtered.select(coloumns_to_keep)

# truncate the DATE to remove minutes and seconds, add 1 hour to get to weather report in the next hour 
df_weather1 = weather_full_filtered_columns.withColumn('Date_Time_trunc', concat(substring('DATE', 1, 10), lit(' '), substring("DATE", 12, 2), lit(':00:00')))
df_weather2 = df_weather1.withColumn('Date_Time', df_weather1.Date_Time_trunc + expr('INTERVAL 1 HOURS'))
display(df_weather2)
df_weather2.count()

# COMMAND ----------

# find the latest weather report time for each hour

max_time = df_weather2.groupby('Date_Time', 'station_id').agg(max('DATE'))
# display(max_time)
# max_time.count()
 
df_weather3 = df_weather2.join(max_time, ['Date_Time', 'station_id'])
display(df_weather3)
df_weather3.count()

# COMMAND ----------

# filter down to one report per hour
df_weather4 = df_weather3.filter(df_weather3['DATE'] == df_weather3['max(DATE)'])
display(df_weather4)
df_weather4.count()

# COMMAND ----------

# join station onto weather table
weather_station_df = df_weather4.join(station_final_df, ['station_id'])
display(weather_station_df)
weather_station_df.count()

# COMMAND ----------

# # check for invalid TZ_timezone
# invalid_TZ = weather_station_df.filter(weather_station_df.TZ_timezone == '\\N')
# display(invalid_TZ)
# invalid_TZ.count()
# invalid_TZ.select('NAME').distinct().show()

# COMMAND ----------

#Converts to UTC
weather_station_final_df = weather_station_df.withColumn('Date_Time_utc', to_utc_timestamp(weather_station_df.Date_Time, weather_station_df.TZ_timezone))
display(weather_station_final_df)
weather_station_final_df.count()

# COMMAND ----------

# Clean up columns before saving
selected_cols = ['station_id', 'ELEVATION', 'NAME', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyPresentWeatherType',
                   'HourlyPressureChange', 'HourlyPressureTendency', 'HourlyRelativeHumidity', 'HourlySkyConditions', 'HourlySeaLevelPressure', 'HourlyStationPressure', 'HourlyVisibility', 
                 'HourlyWetBulbTemperature', 'HourlyWindDirection', 'HourlyWindGustSpeed', 'HourlyWindSpeed', 'IATA', 'Date_Time_utc']

# not used:   'Date_Time', 'DATE', 'Date_Time_trunc','max(DATE)',  'neighbor_call', 'neighbor_name', 'neighbor_state', 'distance_to_neighbor', 'Country', 'Timezone', 'TZ_timezone', 'min(distance_to_neighbor)',

weather_station_final_df = weather_station_final_df.select(selected_cols)
display(weather_station_final_df)
weather_station_final_df.count()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Blob Storage Pre-merge Checkpoint

# COMMAND ----------

# #Checkpoint Airlines Dataset before join
airline_final_df.write.mode("overwrite").parquet(f"{blob_url}/premerge_airline_data")

# #Checkpoint Weather Dataset before join
weather_station_final_df.write.mode("overwrite").parquet(f"{blob_url}/premerge_weather_station_data")


# COMMAND ----------

# MAGIC %md
# MAGIC # Join DataFrames

# COMMAND ----------

# Read Checkpoint Tables

display(dbutils.fs.ls(f"{blob_url}"))

flight_df = spark.read.parquet(f"{blob_url}/premerge_airline_data")

weather_df = spark.read.parquet(f"{blob_url}/premerge_weather_station_data")

display(flight_df)
display(weather_df)

print(f'Stored Airline Data, # of rows: {flight_df.count()}')
print(f'Stored Airline Data, # of columns: {len(flight_df.columns)}')
print(f'Stored Weather Data, # of rows: {weather_df.count()}')
print(f'Stored Weather Data, # of columns: {len(weather_df.columns)}')

# COMMAND ----------

# Join Airline and Weather Data

joined_df = flight_df.join(weather_df, (flight_df["ORIGIN"] == weather_df["IATA"]) & (flight_df["two_hrs_pre_flight_utc"] == weather_df["Date_Time_utc"]),"left")
display(joined_df)
joined_df.count()
                                                                          

# COMMAND ----------

print(f'Joined Data, # of rows: {joined_df.count()}')
print(f'Joined Data, # of columns: {len(joined_df.columns)}')


# COMMAND ----------

# Blob Storage Post-merge Checkpoint

joined_df.write.mode("overwrite").parquet(f"{blob_url}/merged_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merged Dataset - Read and Clean

# COMMAND ----------

# read merged table

display(dbutils.fs.ls(f"{blob_url}"))

df = spark.read.parquet(f"{blob_url}/merged_data")
display(df)
df.count()

# COMMAND ----------

# select columns we want to keep, reformat the datetime so we can check null

columns_to_keep = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK','FL_DATE','two_hrs_pre_flight_utc',
                   'Date_Time_sched_dep_utc', 'Date_Time_sched_arrival_utc', 'OP_CARRIER', 'TAIL_NUM', 'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'CRS_ARR_TIME','ARR_DELAY', 'DEP_DEL15', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DISTANCE_GROUP', 'ELEVATION', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature',
 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyPresentWeatherType', 'HourlyRelativeHumidity',
 'HourlySkyConditions', 'HourlySeaLevelPressure', 'HourlyStationPressure', 'HourlyVisibility','HourlyWetBulbTemperature', 'HourlyWindDirection','HourlyWindSpeed','HourlyWindGustSpeed']
 
# unused columns, ensure no leakage, remove columns that give information after departure
'''
'OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'Origin_IATA', 'origin_Timezone', 'origin_TZ', 'dest_IATA', 'dest_Timezone', 'dest_TZ',
 'Date_Time_flight', 'two_hrs_pre_flight', 'station_id', 'NAME',  'HourlyPressureChange', 'HourlyPressureTendency',
'''  
# HourlyPressureChange, HourlyPressureTendency removed due to high amount of Nulls and questionable relavence
# other columns left out due to similar columns being included and again questionable relavence, were used for joining
  
  
df_filtered = df.select(columns_to_keep)
display(df_filtered)


# COMMAND ----------

df_filtered.count()

# COMMAND ----------

#try to view how many nulls are in each column for feature selection 
# df_Columns= df_filtered
# check_df = df_filtered.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_Columns.columns]).toPandas()

# display(check_df)

# COMMAND ----------

# convert weather columns to numeric instead of categorical

double_list = ['ELEVATION', 'HourlyAltimeterSetting', 'HourlyPrecipitation', 'HourlySeaLevelPressure', 'HourlyStationPressure', 'HourlyVisibility']
int_list = ['HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 'HourlyWetBulbTemperature', 'HourlyWindDirection', 
              'HourlyWindSpeed', 'HourlyWindGustSpeed', 'HourlyRelativeHumidity', 'CRS_DEP_TIME','CRS_ARR_TIME','ARR_DELAY']

for d in double_list:
  df_filtered = df_filtered.withColumn(d, col(d).cast("double"))
  
for i in int_list:
  df_filtered = df_filtered.withColumn(i, col(i).cast("int"))

# COMMAND ----------

df_filtered.dtypes

# COMMAND ----------

display(df_filtered)

# COMMAND ----------

#move delayed column to front
df_filtered =df_filtered.select(['DEP_DEL15','YEAR',
 'QUARTER',
 'MONTH',
 'DAY_OF_MONTH',
 'DAY_OF_WEEK',
 'FL_DATE',
 'two_hrs_pre_flight_utc',
 'Date_Time_sched_dep_utc','Date_Time_sched_arrival_utc',
 'OP_CARRIER',
 'TAIL_NUM',
 'ORIGIN',
 'DEST',
 'CRS_DEP_TIME','CRS_ARR_TIME','ARR_DELAY',
 'CRS_ELAPSED_TIME',
 'DISTANCE',
 'DISTANCE_GROUP',
 'ELEVATION',
 'HourlyAltimeterSetting',
 'HourlyDewPointTemperature',
 'HourlyDryBulbTemperature',
 'HourlyPrecipitation',
 'HourlyPresentWeatherType',
 'HourlyRelativeHumidity',
 'HourlySkyConditions',
 'HourlySeaLevelPressure',
 'HourlyStationPressure',
 'HourlyVisibility',
 'HourlyWetBulbTemperature',
 'HourlyWindDirection',
 'HourlyWindSpeed',
 'HourlyWindGustSpeed'])

# COMMAND ----------

# Blob Storage Post-clean Checkpoint

df_filtered.write.mode("overwrite").parquet(f"{blob_url}/merged_cleaned_data")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Train/Test Split

# COMMAND ----------

df = spark.read.parquet(f"{blob_url}/merged_cleaned_data")

#TRAIN TEST SPLIT

train_data = df.filter(df.YEAR < 2021)
test_data = df.filter(df.YEAR >= 2021)

# COMMAND ----------

display(train_data)

# COMMAND ----------

display(test_data)

# COMMAND ----------

#MISSING VALUES

missing_counts = train_data.select([count(when( col(c).isNull(), c)).alias(c) for c in ['two_hrs_pre_flight_utc','Date_Time_sched_dep_utc','Date_Time_sched_arrival_utc']]).toPandas()

display(missing_counts)

# COMMAND ----------

#Dropped departure data where datetime is null. 

train_data = train_data.filter(col('Date_Time_sched_dep_utc').isNotNull()).sort('Date_Time_sched_dep_utc')


# COMMAND ----------

#Dropped departure data where datetime is null. 
train_data = train_data.filter(col('Date_Time_sched_arrival_utc').isNotNull()).sort('Date_Time_sched_arrival_utc')

# COMMAND ----------

#Blob storage post train split checkpoint
train_data.write.mode("overwrite").parquet(f"{blob_url}/merged_cleaned_data_train")

#Blob storage post test split checkpoint
test_data.write.mode("overwrite").parquet(f"{blob_url}/merged_cleaned_data_test")

# COMMAND ----------

