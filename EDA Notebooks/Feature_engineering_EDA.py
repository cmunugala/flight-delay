# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory Data Analysis (EDA)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In this notebook, we will be performing EDA on the advanced features that we have created. For more information on the feature generation, please refer to this notebook.
# MAGIC <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/632558266978843/command/632558266978844" target="_blank">Feature_engineering_Notebook</a>
# MAGIC 
# MAGIC **Note**: Since we would like to remain blind to the test data, we will perform our EDA on the training data post feature engineering.

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

# MAGIC %md
# MAGIC 
# MAGIC # Advanced Feature Engineering EDA

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# displaying dataset in our storage blob
display(dbutils.fs.ls(f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"))

# COMMAND ----------

train_df = spark.read.parquet(f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net/train_data_with_adv_features")
test_df = spark.read.parquet(f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net/test_data_with_adv_features")

print(f' Training Data: number of rows = {train_df.count()}, number of columns =  {len(train_df.columns)}')

print(f' Training Data: number of rows = {test_df.count()}, number of columns =  {len(test_df.columns)}')

# COMMAND ----------

# MAGIC %md 
# MAGIC ***Dataset Sizes:*** 
# MAGIC 
# MAGIC • Train Data: 35366654 rows (contains data from 2015-2020)
# MAGIC 
# MAGIC • Test Data: 5859306 rows (contains data from 2021)

# COMMAND ----------

display(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Null Values Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Here we look at missing values in our train data to see if there are any features that we would have to come up with a imputation strategy for.

# COMMAND ----------

#MISSING VALUES
 
missing_counts = train_df.select([count(when(col(c).isNull(), c)).alias(c) for c in train_df.columns]).toPandas()
 
display(missing_counts)

# COMMAND ----------

null_count = missing_counts.transpose()
null_count.plot(kind = 'bar',figsize = (20,8), title = 'Missing Values',
                legend = False, xlabel = 'Features', ylabel = 'Counts',fontsize = 10)

# COMMAND ----------

# MAGIC %md
# MAGIC From the plot above, we can see that some values are missing in our training dataset. If we decide to use these variables going forward, we will need to come up with some type of imputation strategy.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC In this section, we will perform a correlations analysis, to see if any features that we think are interesting are correlated with our variable of interest (DEP_DEL_15), which describes whether or not a flight was delayed.

# COMMAND ----------

interesting_columns = ['DEP_DEL15','YEAR','QUARTER','MONTH','ELEVATION','HourlyPrecipitation','HourlyVisibility','HourlyWetBulbTemperature','HourlyWindSpeed',
 'HourlyWindGustSpeed','Rain','Snow','Thunder','CloudySkyCondition','mean_carrier_delay','Pagerank_Score',
                      'PREV_FLIGHT_DELAYED','origin_flight_per_day','origin_delays_per_day','dest_flight_per_day','dest_delays_per_day',
                      'origin_percent_delayed','dest_percent_delayed','ORIGIN_Prophet_trend','ORIGIN_Prophet_pred',
                      'DEST_Prophet_trend','DEST_Prophet_pred']

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=interesting_columns, outputCol=vector_col)
df_vector = assembler.setHandleInvalid('skip').transform(train_df).select(vector_col)
 
# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
correlation_matrix = matrix.toArray().tolist()

corr_matrix_df = pd.DataFrame(data=correlation_matrix)
mask = np.triu(np.ones_like(corr_matrix_df, dtype=bool))
plt.figure(figsize=(21,17))
plt.title('Pearson Correlation Heatmap',size=30)
sns.heatmap(corr_matrix_df, 
            xticklabels=interesting_columns,
            yticklabels=interesting_columns,  cmap="coolwarm", annot=True, mask=mask)
 

# COMMAND ----------

# MAGIC %md
# MAGIC From the heatmap above, we can see that previous flight delayed feature is strongly correlated with the outcome variable. Mean_carrier_Delay has the second strongest correlation with the outcome variable. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploration of Delays by Route
# MAGIC Below, we take a look at which routes are the most popular and what the fraction of delayed flights are for those routes.

# COMMAND ----------

display(train_df.groupBy('Route').count().orderBy(col('count').desc()).take(20))

# COMMAND ----------

#Flight delay by airport route
airport_route_delay_count_0 = train_df.groupby('Route', 'DEP_DEL15').count().filter(train_df.DEP_DEL15 != "1")
airport_route_delay_count_1 = train_df.groupby('Route', 'DEP_DEL15').count().filter(train_df.DEP_DEL15 != "0")
airport_route_delay_count_1 = airport_route_delay_count_1.selectExpr("Route as Route_1", "DEP_DEL15 as DEP_DEL15_1", "count as count_1")
airport_route_delay_count_0 = airport_route_delay_count_0.join(airport_route_delay_count_1,airport_route_delay_count_0.Route ==  airport_route_delay_count_1.Route_1,"inner")
airport_route_delay_count_0 = airport_route_delay_count_0.drop(airport_route_delay_count_0.Route_1)
airport_route_delay_count_0 = airport_route_delay_count_0.withColumn("total", col("count")+col("count_1"))
airport_route_delay_count_0 = airport_route_delay_count_0.withColumn("relative_delay", (col("count_1") / (col("count")+col("count_1"))) * 100).orderBy(col('relative_delay').desc()).take(20)
display(airport_route_delay_count_0)

# COMMAND ----------

# let's investigate the number of delays in the top of 20 route by volume. 
airport_route_delay_count_0 = train_df.groupby('Route', 'DEP_DEL15').count().filter(train_df.DEP_DEL15 != "1")
airport_route_delay_count_1 = train_df.groupby('Route', 'DEP_DEL15').count().filter(train_df.DEP_DEL15 != "0")
airport_route_delay_count_1 = airport_route_delay_count_1.selectExpr("Route as Route_1", "DEP_DEL15 as DEP_DEL15_1", "count as count_1")
airport_route_delay_count_0 = airport_route_delay_count_0.join(airport_route_delay_count_1,airport_route_delay_count_0.Route ==  airport_route_delay_count_1.Route_1,"inner")
airport_route_delay_count_0 = airport_route_delay_count_0.drop(airport_route_delay_count_0.Route_1)
airport_route_delay_count_0 = airport_route_delay_count_0.withColumn("total", col("count")+col("count_1"))
airport_route_delay_count_00 = airport_route_delay_count_0.withColumn("relative_delay", col("count_1") / (col("count")+col("count_1"))).orderBy(col('total').desc()).take(20)
display(airport_route_delay_count_00)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Potentially Redundant Variables
# MAGIC Finally, we look at some weather-related variables that we did not look at in the the previous heatmap, to see if these variables are highly correlated and would thus be redundant.

# COMMAND ----------

final_df_numerical = train_df.withColumn("ELEVATION", train_df["ELEVATION"].cast(IntegerType())) \
                             .withColumn("HourlyAltimeterSetting", train_df["HourlyAltimeterSetting"].cast(IntegerType())) \
                             .withColumn("HourlyDewPointTemperature", train_df["HourlyDewPointTemperature"].cast(IntegerType())) \
                             .withColumn("HourlyDryBulbTemperature", train_df["HourlyDryBulbTemperature"].cast(IntegerType())) \
                             .withColumn("HourlyPrecipitation", train_df["HourlyPrecipitation"].cast(IntegerType())) \
                             .withColumn("HourlyRelativeHumidity", train_df["HourlyRelativeHumidity"].cast(IntegerType())) \
                             .withColumn("HourlySeaLevelPressure", train_df["HourlySeaLevelPressure"].cast(IntegerType())) \
                             .withColumn("HourlyStationPressure", train_df["HourlyStationPressure"].cast(IntegerType())) \
                             .withColumn("HourlyVisibility", train_df["HourlyVisibility"].cast(IntegerType())) \
                             .withColumn("HourlyWetBulbTemperature", train_df["HourlyWetBulbTemperature"].cast(IntegerType())) \
                             .withColumn("HourlyWindGustSpeed", train_df["HourlyWindGustSpeed"].cast(IntegerType())) \
                             .withColumn("HourlyWindSpeed", train_df["HourlyWindSpeed"].cast(IntegerType())) \
                             .withColumn("CRS_ELAPSED_TIME", train_df["CRS_ELAPSED_TIME"].cast(IntegerType())) \
                             .withColumn("DISTANCE", train_df["DISTANCE"].cast(IntegerType()))
 
final_df_numerical_columns = final_df_numerical.select('ELEVATION','DISTANCE','HourlyAltimeterSetting','HourlyDewPointTemperature','HourlyDryBulbTemperature',
                       'HourlyPrecipitation','HourlyRelativeHumidity','HourlySeaLevelPressure','HourlyStationPressure','HourlyVisibility','HourlyWetBulbTemperature',
                       'HourlyWindDirection','HourlyWindSpeed','HourlyWindGustSpeed')
 
correlation_columns = ['ELEVATION','DISTANCE','HourlyAltimeterSetting','HourlyDewPointTemperature','HourlyDryBulbTemperature',
                       'HourlyPrecipitation','HourlyRelativeHumidity','HourlySeaLevelPressure','HourlyStationPressure','HourlyVisibility','HourlyWetBulbTemperature',
                       'HourlyWindDirection','HourlyWindSpeed','HourlyWindGustSpeed']
vector_column = "correlation_features"
assembler = VectorAssembler(inputCols=correlation_columns, outputCol=vector_column)
df_vector = assembler.setHandleInvalid('skip').transform(final_df_numerical_columns).select(vector_column)
# display(df_vector)
 
# correlation matrix
matrix = Correlation.corr(df_vector, vector_column).collect()[0][0]
correlation_matrix = matrix.toArray().tolist()

# COMMAND ----------

#plot heatmap
mask1 = np.triu(np.ones_like(correlation_matrix, dtype=bool))
fig,ax=plt.subplots(figsize=(15,15))
_=sns.heatmap(correlation_matrix,mask=mask1,annot=True,fmt='.2f',
              ax=ax,xticklabels=correlation_columns,
              yticklabels=correlation_columns,
             cmap ='coolwarm',vmin=-1,vmax=1)
_=ax.set_title('Pearson Correlation Heatmap for Numerical Data')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC From the heatmap above, we see that some variables, such as "HourlyDewPointTemperature" and "HourlyDryBulbTemperature" are highly correlated, so therefore we do not need to include both in our models.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Creating a map of airports and their closest weather stations

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
 
df_stations = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")
 
df_airlines = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data/")
# display(df_airlines)
 
df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data/")
#display(df_weather)

#import dataset
airport_codes_with_time_zones = pd.read_csv('https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat', header=None)
column_namess = {0: 'AirportID', 1: 'Name', 2: 'City', 3: 'Country', 4: 'IATA', 5: 'ICAO', 6: 'airport_latitude', 
             7: 'airport_longitude', 8: 'airport_elevation', 9: 'Timezone', 10: 'Daylight_savings_time', 11: 'TZ_Timezone', 12: 'Type', 13: 'Source'}
 
#add column names 
airport_codes_with_time_zones.rename(columns=column_namess, inplace=True)
#selecting desired columns 
codes = airport_codes_with_time_zones[['Country','IATA','ICAO','Timezone', 'TZ_Timezone']]
 
# found one airport in the final dataset wiht invalid timezone
# codes.loc[codes['IATA'] == 'BIH', 'TZ_Timezone'] = 'America/Los_Angeles' 
 
#converting to PySpark Dataframe
airport_codes = spark.createDataFrame(codes)
 
#filtering stations data set with airport_codes dataset
stations_data_filtered = df_stations.join(airport_codes).where(df_stations["neighbor_call"] == airport_codes["ICAO"])
 
#selecting US, Puerto Rico, and Virgin Islands.
countries =['United States','Puerto Rico','Virgin Islands']
stations_data_filtered_US = stations_data_filtered.filter(stations_data_filtered.Country.isin(countries))
 
#selecting desired columns
cols_to_keeep = ['neighbor_call','IATA','station_id',
                 'lon','lat', 'neighbor_lat', 'neighbor_lon','distance_to_neighbor']
 
stations_data_us = stations_data_filtered_US.select(cols_to_keeep)
 
minimum_distance = stations_data_us.groupby('neighbor_call').agg(min('distance_to_neighbor'))
f_airport_stations = stations_data_us.join(minimum_distance, ['neighbor_call'])
 
#final station data
station_final_usa = f_airport_stations.filter(f_airport_stations['distance_to_neighbor'] == f_airport_stations['min(distance_to_neighbor)'])
airportss = station_final_usa.dropDuplicates(['IATA'])
airports_pd = airportss.toPandas()
stationss = station_final_usa.dropDuplicates(['station_id'])
stations_pd = stationss.toPandas()

# COMMAND ----------

#creating the map

import plotly.graph_objects as plotly
 
fig = plotly.Figure(data=plotly.Scattergeo(
         lat = airports_pd['neighbor_lat'],
         lon = airports_pd['neighbor_lon'],
         text = airports_pd['IATA'],
         name = 'Airport',
         marker_symbol = 'x-dot',
         marker_color = 'green',
         opacity = 0.5))
fig.add_trace(plotly.Scattergeo(
         lat = stations_pd['lat'],
         lon = stations_pd['lon'],
         marker_color = 'red',
         name = 'Weather Stations',
         marker_symbol = 'circle-dot',
         marker_size = 2))
fig.update_layout(
         title = 'Airports & their Closest Weather Stations',
         geo_scope='usa',
         autosize=False,
         width=1000,
         height=600,)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC From the graph above, we can see that even though we had included Puerto Rico and Virgin Island in our fitler, it does not show up in our map. Therefore, we would only focus on the US 50 states

# COMMAND ----------

# MAGIC %md
# MAGIC Next, let's explore whether if there is a spike in delays during holiday periods. 

# COMMAND ----------

display(train_df.groupBy('DEP_DEL15','holiday_period').count())

# COMMAND ----------

# MAGIC %md
# MAGIC As we can fromt the plot above, Holiday periods do not seem to have any major effects on the delay percentage.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Next, it would be interesting to if having a higher daily carrier delay average has any effects on the delays.

# COMMAND ----------

distinct_carrier_count = train_df.groupBy("OP_CARRIER").count()
distinct_carrier_mean_delay_agg = train_df.groupBy("OP_CARRIER",'DEP_DEL15').agg(f.sum("mean_carrier_delay"))
combined = distinct_carrier_mean_delay_agg.join(distinct_carrier_count, ["OP_CARRIER"])
combined_filter = combined.filter(f.col('DEP_DEL15') == '1')
aaa = combined_filter.withColumn('Percentage_daily_delayed_per_carrier',(col('sum(mean_carrier_delay)') / col('count')) * 100)
display(aaa)

# COMMAND ----------

# MAGIC %md
# MAGIC As we can from the graph above, some carrier are more likely to have delays than others. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We will now look at the PageRank Scores for the top 50 highest PageRank Scores per airport

# COMMAND ----------

display(train_df.groupBy('ORIGIN','Pagerank_Score').count().orderBy(col('Pagerank_Score').desc()).take(50))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's try to explore the origin percent delay for prior day and its relation to the current day delay 

# COMMAND ----------

origin_count = train_df.groupBy("ORIGIN").count()
origin_count_origin_percent_delayed_agg = train_df.groupBy("ORIGIN",'DEP_DEL15').agg(f.sum("origin_percent_delayed"))
comb = origin_count_origin_percent_delayed_agg.join(origin_count, ["ORIGIN"])
comb_filter = comb.filter(f.col('DEP_DEL15') == '1')
bbb = comb_filter.withColumn('Percent_daily_delayed_per_origin',(col('sum(origin_percent_delayed)') / col('count')) * 100)
ccc = bbb.orderBy(col('Percent_daily_delayed_per_origin').desc()).take(20)
display(ccc)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Last but not least, we also run prophet forecasting analysis, which is a time series analysis. More information about that can be found in the following notebook.
# MAGIC <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/632558266979478/command/632558266979479" target="_blank">Prophet_Analysis_Notebook</a>

# COMMAND ----------

