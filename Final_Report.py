# Databricks notebook source
# MAGIC %md
# MAGIC #Final Report
# MAGIC #####Section 4 Group 2

# COMMAND ----------

# MAGIC %md
# MAGIC ##Team Members

# COMMAND ----------

# MAGIC %md
# MAGIC #####Chetan Munugala - c.munugala@berkeley.edu
# MAGIC #####Evan Chan - evan.chan@berkeley.edu
# MAGIC #####Ahmad Azizi - aazizi@berkeley.edu
# MAGIC #####Kolby Devery - kdevery@berkeley.edu

# COMMAND ----------

# MAGIC %md
# MAGIC ##Team Photo
# MAGIC <img src="https://github.com/kolbyjedevery/image_files/blob/main/Team_image_1.png?raw=true" width=70%>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Phase Leader Plan
# MAGIC 
# MAGIC Phase Leader Plan for all phases can be found here: https://docs.google.com/spreadsheets/d/1_Fu7YnJMCSgj5JDAQebhh9njFW9kfnub4GDQX-s4-5c/edit#gid=0
# MAGIC 
# MAGIC | Final Phase                                                                             | Ahmad | Chetan | Evan | Kolby |
# MAGIC |-------------------------------------------------------------------------------------|-------|--------|------|-------|
# MAGIC | Phase Leader                                                                        | X     |        |      |       |
# MAGIC | Add additional EDA for feature engineering                                          | X     |        |      |       |
# MAGIC | Iterate on the baseline, improve with hyperparameter tuning or additional variables | X     | X      | X    | X     |
# MAGIC | Gridsearch, train, and evaluate GBT classifier                                      |       | X      | X    |       |
# MAGIC | Gridsearch, train, and evaluate MLP classifier                                      |       | X      |      | X     |
# MAGIC | Explore feasability of synthetic minority oversampling technique for model          | X     | X      | X    | X     |
# MAGIC | Develop undersampling technique                                                     | X     |        |      |       |
# MAGIC | Look at features which add most to predictability                                   |       | X      |      |       |
# MAGIC | Ensure end-to-end pipeline with fully functional and clean                          | X     | X      | X    | X     |
# MAGIC | Summary of project in the form of a final notebook                                  | X     | X      | X    | X     |

# COMMAND ----------

# MAGIC %md
# MAGIC ##Credit Assignment
# MAGIC | Project Steps     | Task                                | Description                                                                         | Status (%) | Owner         | Person-hours |
# MAGIC |-------------------|-------------------------------------|-------------------------------------------------------------------------------------|------------|---------------|--------------|
# MAGIC | Advanced Models   | EDA                                 | Add additional EDA for feature engineering                                          | 100        | Ahmad         | 2            |
# MAGIC |                   | Improve Baseline                    | Iterate on the baseline, improve with hyperparameter tuning or additional variables | 100        | All           | 3            |
# MAGIC |                   | Explore Gradient Boosting Model     | Gridsearch, train, and evaluate GBT classifier                                      | 100        | Chetan,Evan   | 8            |
# MAGIC |                   | Explore Multilayer Perceptron Model | Gridsearch, train, and evaluate MLP classifier                                      | 100        | Kolby, Chetan | 8            |
# MAGIC |                   | SMOTE application                   | Explore feasability of synthetic minority oversampling technique for model          | 100        | All           | 2            |
# MAGIC |                   | Undersampling                       | Develop undersampling technique                                                     | 100        | Ahmad         | 2            |
# MAGIC |                   | Explore feature significance        | Look at features which add most to predictability                                   | 100        | Chetan        | 2            |
# MAGIC | Final submissions | Clean up all code                   | Ensure end-to-end pipeline with fully functional and clean                          | 100        | All           | 10           |
# MAGIC |                   | Complete write-up                   | Summary of project in the form of a final notebook                                  | 100        | All           | 10           |

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Table of Contents
# MAGIC 
# MAGIC - Abstract
# MAGIC - Introduction
# MAGIC   - Data Description
# MAGIC   - EDA
# MAGIC - Methodology
# MAGIC    - Basic EDA of the datasets and preparations for the joins
# MAGIC    - Joining the Data
# MAGIC    - Feature Engineering
# MAGIC    - Model Pipeline - Train-Test Split
# MAGIC    - Model Pipeline - Feature scaling and imputation
# MAGIC    - Loss Function
# MAGIC    - Performance Metrics
# MAGIC - Model Building 
# MAGIC   - Logistic Regression and Random Forest Baseline
# MAGIC   - Baseline Gap Analysis
# MAGIC   - Logistic Regression with All Features
# MAGIC - Results and Discussion
# MAGIC - Conclusion

# COMMAND ----------

# MAGIC %md
# MAGIC # Abstract
# MAGIC To better notify consumers that their flights are more than 15 minutes delayed, we leveraged machine learning models to provide a prediction. These predictions will be made two hours before scheduled departure using known information at that time for all US domestic flights. 
# MAGIC 
# MAGIC We cleaned and joined three datasets incorporating historical flight information and weather from 2015-2021. After exploratory data analysis and feature engineering, we split the 2021 data as a test set and performed broken window cross validation on the remainder for training. For the evaluation metrics, F0.5 was used, since we wish to emphasize precision, thus reducing false positives (impact on consumer falsely believing their flight will be delayed). Our baseline consisted of logistic regression and random forest with F0.5 scores of 0.06 and 0.03, respectively. For advanced models, with feature engineering, hyperparameter tuning, and undersampling, we used logistic regression, random forest, GBT, MLPC, and ensemble with and without undersampling. The best F0.5 score for all models was 0.59. For future research, more feature engineering and a combination of undersampling and oversampling might be helpful.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Introduction
# MAGIC 
# MAGIC Flying is one of the main methods of transportation in the US. Ever year, however, flight delays cost airlines and passengers staggering amounts of money, not to mention the emotional distress that comes along with it. According to federal Aviation Administration, in 2019 alone, passengers lost around 18.1 billion dollars due to flight delays. If you add the impacts to Airlines, loss in future demand, and other related costs, this amount reaches to around 33 billion dollars, and the bulk of these costs are passed on to passengers.
# MAGIC 
# MAGIC In this project, we are going to tackle flight delays and build models to predict delays 2 hours prior to the scheduled flight time. A delay will be designated as 15 minutes or more. We will leverage machine learning in the context of distributed framework of Spark for data processing, joining, feature engineering, and model building. 
# MAGIC 
# MAGIC The success metric of our model will focus on reducing false positives, since it is fine if the flight is predicted not to be delayed, when in fact it is. The alternative may cause passengers to unnecessarily miss their flights. To that end, in machine learning syntax, we will be focusing on F 0.5 scores. We believe that with the vast number of datapoint that are at our disposal, we will be able to build machine learning models with decent F 0.5 scores. 
# MAGIC 
# MAGIC We will next introduce the datasets in detail and do exploratory data analysis. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Description
# MAGIC 
# MAGIC For this project, we have three data sets:
# MAGIC 
# MAGIC 1- Flights Dataset from the U.S. Department of Transportation. https://www.transtats.bts.gov/homepage.asp \
# MAGIC 2- Weather Dataset from the National Oceanic and Atmospheric Administration Repository. https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00679   
# MAGIC 3- Weather Station dataset from the U.S. Department of Transportation. https://www.transportation.gov/
# MAGIC 
# MAGIC All three datasets will be loaded as parquet files for the following reasons
# MAGIC 
# MAGIC * Parquet has a columnar format and since our data has a huge number of rows, it would make sense to load the data as a parquet file. 
# MAGIC * For our analysis, we will be performing aggregagtions which is also ideal for parquet. 
# MAGIC * We have huge datasets, and parquet will save cloud storage by its highly efficient column-wise compression and its flexible encoding schemes for columns with a variety of data types.  
# MAGIC 
# MAGIC Our goal is to use the aforementioned datasets to built a predictive model that will predict flight delays 2 hours before the departure time from all major US airports. We will be using data points from 2015-2021.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Exploratory Data Analysis (EDA)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC To start getting to know the data, we performed our EDA on the individual datasets (flights, weather, and stations). The exploratory data analysis can be found the in the notebook linked here <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/632558266979372/command/632558266979374" target="_blank">Preliminary_EDA_Notebook</a> \
# MAGIC We performed exploratory data analysis once feature engineering was completed. Notebook linked here <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238253226/command/1215577238253227" target="_blank">Feature_engineering_EDA_Notebook</a>. More information about feature engineering can be found in Methods' section
# MAGIC 
# MAGIC ####Summary description of each table
# MAGIC 
# MAGIC #####1- Flights Dataset
# MAGIC We will use this data set to obtain crucial features necessary for flight information. This dataset has 109 columns with over 39 million rows. Some of the important features that we will be using are as follows. \
# MAGIC _ **Tail_NUM**: (Str) This features identifies each plane. There are other variables for Airline information as well. \
# MAGIC _ **FL_Data**: (Str) This features gives us the flight date. There are other variables for quarter, month, day, etc. \
# MAGIC _ **Dest** and **Origin**: (Str) These two features point to the three letter airport codes for the the origin and Destination flights. \
# MAGIC _ **DepTime**,**CRSDepTime**, and **CRSArrTime**: (int, local time: hhmm) These two features point to the planned and actual departure times, and the planned arrival.\
# MAGIC _ **DepDel15**: (categorical) This is cateogorical variable that indicates whether there was a 15 min delay (1=Yes). \
# MAGIC _ **Cancelled** and **Diverted**: (Categorical) These tw features point to whether the flight was cancelled or divert for any reason. We will not use flights that were cancelled or diverted. 
# MAGIC 
# MAGIC There are more variables at our disposal that we will use. We will also be dropping columns that we would not need for this analysis. 
# MAGIC 
# MAGIC #####2- Weather Station Dataset
# MAGIC We will use this data set to find the closest weather stations to the airports. This dataset has 12 columns and over 5 millions rows. Some of the features are as follows: \
# MAGIC _ **station_id**: (str): This feature identifies each weather station and unique values for weather stations. \
# MAGIC _ **neighbor_call** (str): This feature contains the unique identifiers for the airports. \
# MAGIC _ **distance_to_neighbor** This feature has the distance between weather stations to the airports. We will use this variable to find the closes the weather stations to airports. 
# MAGIC 
# MAGIC #####3- Weather Dataset
# MAGIC This data set contains information about weather from weather stations. It has 177 columns and over 630,904,436 rows. We have a lot of null values across various columns. We will be using information about the wind, humidity, precipitation, seal level Pressure, and other metrics for weather. 
# MAGIC 
# MAGIC #####Data Dictionary (test description; data type: numerical, list, etc.)
# MAGIC 
# MAGIC DEP_DEL15: double  - flight delayed or not  
# MAGIC YEAR: int  - year  
# MAGIC QUARTER: int  - quarter of year  
# MAGIC MONTH: int   - month   
# MAGIC DAY_OF_MONTH: int - numerical day of month  
# MAGIC DAY_OF_WEEK: int  - day of week  
# MAGIC FL_DATE: str - flight date
# MAGIC two_hrs_pre_flight_utc: timestamp  - time two hours before flight (UTC)  
# MAGIC Date_Time_sched_dep_utc: timestamp  - scheduled departure time of flight (UTC)
# MAGIC Date_Time_sched_arrival_utc: timestamp - scheduled arrival time fo the flight (UTC)
# MAGIC OP_CARRIER: string  - carrier airline  
# MAGIC TAIL_NUM: string  - tail number of aircraft  
# MAGIC ORIGIN: string  - origin airport  
# MAGIC DEST: string  - destination airport  
# MAGIC CRS_DEP_TIME: int   - departure time  
# MAGIC CRS_ARR_TIME: int   - departure time
# MAGIC ARR_DELAY: int - arrival delay 
# MAGIC CRS_ELAPSED_TIME: double  - elapsed time     
# MAGIC DISTANCE: double  - flight distance  
# MAGIC DISTANCE_GROUP: - ordinal variable describing distance  
# MAGIC ELEVATION: double  - elevation   
# MAGIC HourlyAltimeterSetting: double - altimeter setting describing pressure     
# MAGIC HourlyDewPointTemperature: int - the temperature the air needs to be cooled to (at constant pressure) in order to achieve a relative humidity of 100%  
# MAGIC HourlyDryBulbTemperature: int - ambient temperature  
# MAGIC HourlyPrecipitation: double - precipitation (rainfall)   
# MAGIC HourlyRelativeHumidity: int - humidity    
# MAGIC HourlySeaLevelPressure: double - sea level pressure  
# MAGIC HourlyStationPressure: double - pressure at station level  
# MAGIC HourlyVisibility: double  - measure of visibility   
# MAGIC HourlyWetBulbTemperature: int  - temperature that accounts for both heat and humidity  
# MAGIC HourlyWindDirection: int  - wind direction  
# MAGIC HourlyWindSpeed: int - wind speed   
# MAGIC HourlyWindGustSpeed: int - gust speed (brief increases in wind)   
# MAGIC Route: string  - route of airport  
# MAGIC Rain: int  - Binary variable for rain   
# MAGIC Snow: int - Binary variable for Snow  \
# MAGIC Thunder: int - Binary variable for Thunder \
# MAGIC Fog: int -  Binary variable for Fog \
# MAGIC Mist: int - Binary variable for Mist \
# MAGIC Freezing: int - Binary variable for Freezing \
# MAGIC Blowing: int  - Binary variable for Wind Blowing \
# MAGIC Smoke: int  - Binary variable for Smoke \
# MAGIC Drizzle: int  - Binary variable for Drizzle \
# MAGIC Overcast: int  - Binary variable for Overcast \
# MAGIC Broken: int  - Binary variable for Broken \
# MAGIC Scattered: int  - Binary variable for Scattered \
# MAGIC CloudySkyCondition: int - Binary variable for Poor Cloud Sky Condition (either Overcast or Broken was true) \
# MAGIC holiday_period: Binary variable: - dates of the year considered to be holiday period \
# MAGIC mean_carrier_delay: int - average carrier delay for the flight date \
# MAGIC Pagerank_score: int - Pagerank score based on origin and destination \
# MAGIC PREV_FLIGHT_DELAYED: Binary variable for whether the previous flight was delayed \
# MAGIC origin_flight_per_day: int - number of flight for the previous day for the origin airport \
# MAGIC origin_delays_per_day: int - number of delays for the previous day for the origin airport \
# MAGIC dest_flight_per_day: int - number of flight for the previous day for the dest airport \
# MAGIC dest_delays_per_day: int - number of delays for the previous day for the destination airport \
# MAGIC origin_percent_delay: double - percentage of delays for the previous day for the origin \
# MAGIC dest_percent_delay: double - percentage of delays for the previous day for the destination \
# MAGIC ORIGIN_Prophet_trend: double - Percentage delay trend for the origin \
# MAGIC ORIGIN_Prophet_pred: double - Forecasted percentage delay for the origin \
# MAGIC DEST_Prophet_trend: double - Percentage delay percentage trend for the destination \
# MAGIC DEST_Prophet_pred: double - Forecasted delay percentage for the destination
# MAGIC 
# MAGIC #####Dataset size 
# MAGIC 
# MAGIC • Train Data: 35,659,347 rows (contains data from 2015-2020)
# MAGIC • Test Data: 5,892,460 rows (contains data from 2021)
# MAGIC 
# MAGIC Our validation data set size will be 20% of our train_data. We plan to do five fold cross validation. 
# MAGIC 
# MAGIC #####Correlation analysis, visualization of target features, and visualization of correlation analysis,missing value analysis, and more can be found in the EDA notebook (<a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238253226/command/1215577238253227" target="_blank">Feature_engineering_EDA_Notebook</a>)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Methodology
# MAGIC 
# MAGIC This project was conducted using databricks in Microsoft Azure cloud service using distributed framework of PySpark. We used 10 workers, 4 cores each with total of 40 cores. We were very methodical in our approach for processing, joining, analyzing, feature engineering, and model building. Each step will explored below.

# COMMAND ----------

# MAGIC %md
# MAGIC ####1. Basic EDA of the datasets and preparations for the joins
# MAGIC 
# MAGIC In this step, we performed Processing of text code weather columns and other easily calculated features. All missing data will need to be imputed accordingly. EDA on final dataset to determine which features are highly correlated, generate ideas for additional features. See Exploratory Data Analysis section above.

# COMMAND ----------

# MAGIC %md
# MAGIC ####2. Joining the data
# MAGIC 
# MAGIC Next, we prepared our data for joins as follows
# MAGIC 
# MAGIC ***Additional Data Used***
# MAGIC Openflights data: <a href="https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat" target="_blank">Github Openflights</a>
# MAGIC 
# MAGIC **Join Steps**
# MAGIC 1. Station Dataset Cleaning - 5,004,169 rows
# MAGIC     - Using data from Openflights, filter station dataset to 1304 rows, this is weather stations only at airports in the United States.
# MAGIC     - Join columns from Openflights onto the station dataset
# MAGIC         - Station table column "neighbor_call" contains codes for nearby airports by 4 letter ICAO designation
# MAGIC         - Openflights provides a translation to 3 letter IATA airport code which is used by the airlines data.
# MAGIC         - Openflight provides timezone information for all remaining stations
# MAGIC         - See **Stations Dataset - Cleaning and EDA** section below for code
# MAGIC 2. Flights Dataset Cleaning - 74,177,433 rows
# MAGIC     - 'DEP_DEL15' is our target column to predict, so we filter any rows with a Null value, results in 41,551,807 rows
# MAGIC         - This generally removes cancelled flights
# MAGIC         - Also dropped many duplicate rows
# MAGIC     - Filter to columns of interest only
# MAGIC         - Columns identified in EDA
# MAGIC         - We also remove columns that could leak information from the occurrances after we make are to make our prediction
# MAGIC         - This is mostly information post departure (arrival and flight time information)
# MAGIC     - Filter rows who's airport codes are not in the station dataset 
# MAGIC         - 10 airports (mostly on US territories) do not have an entry in the station dataset, therefore we cannot find their nearest weather station
# MAGIC         - Results in 379 unique airports to move forward with
# MAGIC     - Using station dataset, create columns for origin and destination timezones
# MAGIC     - Concatenate FL_DATE and departure time to get datatime for departure for each flight
# MAGIC         - Truncate minutes and seconds to get to hourly basis
# MAGIC     - Convert datatime to UTC time
# MAGIC     - Checkpoint Airline dataset
# MAGIC     - See **Airline Dataset - Cleaning and EDA** section below for code
# MAGIC         
# MAGIC 3. Weather Dataset Cleaning - 898,983,399 original rows
# MAGIC     - With station data filtered down to 379 remaining airport codes, filter to rows with station_ids in the station dataset
# MAGIC         - Also Filter to report_type FM-15, hourly aviation weather reports
# MAGIC         - Results in 25,844,376 remaining rows
# MAGIC     - Convert report time to hourly basis, round-up since we assume that that report will be used for the next hour
# MAGIC         - Filter to the nearest report time to the hour, ensures that there is only one report per hour, results in 23,117,906 rows
# MAGIC     - Join station dataset to weather dataset
# MAGIC         - Adds timezone information to convert report time to UTC
# MAGIC         - Resulting table has ICAO airport code and report time in UTC for joing to airline dataset 
# MAGIC     - Filter to column interest
# MAGIC         - Columns found through EDA
# MAGIC         - Remove Columns with too many Null values
# MAGIC     - Checkpoint Weather + Station merged dataset
# MAGIC     - See **Weather Dataset - Cleaning and EDA** section below for code
# MAGIC 
# MAGIC 4. Joining
# MAGIC     - Read Check Pointed Tables
# MAGIC     - Join Weather + station dataset to Airline dataset
# MAGIC         - Left Join, Weather + Station on to Airline
# MAGIC     - ORIGIN or DEST to IATA codes and two_hrs_pre_flight_utc to Date_Time_utc
# MAGIC     - Checkpoint merged dataset
# MAGIC     - See **Join Datasets** section below for code
# MAGIC 
# MAGIC **Data cleaning and Join Notebook**
# MAGIC <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/632558266978976/command/632558266978977" target="_blank">Data_cleanup_and_join</a>
# MAGIC 
# MAGIC **Join Statistics** 
# MAGIC 
# MAGIC **Table Sizes**
# MAGIC 
# MAGIC - Merged Weather + Station: 23,117,906 rows, 21 columns 
# MAGIC - Airline Dataset: 41,551,807 rows, 26 columns
# MAGIC - Merged Dataset: 41,551,807 rows, 47 columns 
# MAGIC 
# MAGIC **Time to Run Joins**
# MAGIC 
# MAGIC - Weather to Station datasets join: 3.1 minutes 
# MAGIC - Weather + Station to Airline datasets join: 2 minutes
# MAGIC - Total Join time: 5 minutes
# MAGIC 
# MAGIC **Cluster Size**
# MAGIC 
# MAGIC - 10 workers, 4 cores each
# MAGIC - 40 total cores

# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Feature Engineering
# MAGIC 
# MAGIC We had two main phases of feature engineering. We did preliminary feature engineering on the weather text data and advanced feature engineering using rest of the columns. 
# MAGIC In order to make good use of the data, we performed feature engineering on the merged data. For full details refer to this notebook. <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/632558266978843/command/632558266978844" target="_blank">Feature Engineering</a>. A summary of the features can be found below.
# MAGIC 
# MAGIC #####1. String Based Feature Engineering:
# MAGIC 
# MAGIC - 1- We primarily used HourlyPresentWeatherType and HourlySkyConditions to produce weather conditions such as snow, rain as well as sky conditions such as broken, overcast, etc. We are doing this to be able to decipher HourlyPresentWeatherType and HourlySkyConditions column properly, since at their original state, they are not very decipheribal.
# MAGIC 
# MAGIC - 2- We further created a new column named route, which will essentially have origin-destination columns concatentated. This will help us in our EDA and possibly shed some light. This feature will not be used in model building
# MAGIC 
# MAGIC 
# MAGIC #####2. Graph Based feature:
# MAGIC 
# MAGIC  - `PageRank`: Next feature that we added was leveraging the power of pagerank algorithm. Since airport location for the most parts are stationary, we used destination and origin airports as nodes to create a page rank score.
# MAGIC 
# MAGIC 
# MAGIC #####3. Frequency related features along with time component:
# MAGIC 
# MAGIC  - `Holiday Period` :
# MAGIC Most holidays bring about lots of traveling for Americans. And with lots of traveling comes the potential for more delays. According to NPR in 2022, during the thanksgiving holiday, more than 4,000 flights were delayed. Since we are trying to predict flight delays, we believe that it might be beneficial to have a holiday period feature. All holiday information was taken from https://www.officeholidays.com/countries/usa
# MAGIC 
# MAGIC  - `Mean Carrier Delay` :
# MAGIC Since we have a time series data, we believed that having a feature for the average carrier delays 48 hours prior to the flight scheduled time might be beneficial and might affect future delays. This features calculates the average of a flight carrier 26 hrs before the scheduled flight time. It takes 26 hrs rather than 24hr in to consideration since our model will predict delay 2 hrs prior to scheduled flight time. We do not want data leakage less than 2 hrs to the scheduled flight time.  
# MAGIC 
# MAGIC  - `Previous Flight Delay Flag` :
# MAGIC This feature is basically a flag for whether the previous flight was delayed. Interestingly, this feature had the highest Pearson Correlation with the outcome variable. This makes logical sense, in that if the previous flight was delayed, it will affect the next fligth time as well.
# MAGIC 
# MAGIC  - `Delayed flight percentage of previous day at Origin` : 
# MAGIC This feature looks at the average flight delay percentage for the origin airport. It calculate the average of flight delays for the origin 48 hours prior to the flight date. It calculates 48 hours rather than 24 hours to prevent possible data leakage from the the prior day for midnight flights. 
# MAGIC 
# MAGIC  - `Delayed flight percentage of previous day at Destination` :
# MAGIC This feature has the same logic as the delayed flight percentage of previous day at Origin, except it takes into accound the destination airport.
# MAGIC 
# MAGIC 
# MAGIC #####4. Time Series related features:
# MAGIC - `Prophet trends at Origin` :
# MAGIC Finally, due to the time series nature of our data, it would be been a waste, not to leverage a time series forecasting model. This features forecast delays at the origin airports. 
# MAGIC 
# MAGIC - `Prophet trends at Destination` :
# MAGIC This feature has the same logic as "Prophet trends at Origin" feature, except that it forecasts for the destination. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Our final dataset looks as follows:

# COMMAND ----------

# read train dataset from merged table
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
 
# display(dbutils.fs.ls(f"{blob_url}"))
 
final_df_train = spark.read.parquet(f"{blob_url}/train_data_with_adv_features")
display(final_df_train)


# COMMAND ----------

# MAGIC %md
# MAGIC ####4. Model Pipeline - Train-Test Split
# MAGIC 
# MAGIC For the modeling pipeline, first we had to split our data. Data from 2015-2020 will be used as training data, and data from 2021 will be as testing data. We further leveraged Broken Window Time Series cross validation on the train data, in that the 6 years of train data (2015-2020) will be broken into 5 time-chronologically sorted cross validation folds. The first 80% of flights for each fold will be the training data, and the last 20% will be used as validation. These folds will assist in assessing model performance and the tweaking of our hyperparameters. 
# MAGIC 
# MAGIC #####Cross Validation on Time Series Visualization:
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/kolbyjedevery/image_files/blob/main/broken_window_cross.png?raw=true" width=30%>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ####5. Model Pipeline - Preprocessing
# MAGIC #####Imputation, Scaling, and Vector Assembler 
# MAGIC 
# MAGIC For the next stage of the modeling pipeline, StandardScaler will be fitted on the training data for each cross validation set only, the validation set will be fit to the same scaler. All null binary features will be imputed with 0's and null numeric features will be imputed by filling the null with the mean values. The chart below shows the Modeling pipeline.
# MAGIC 
# MAGIC The imputed and scaled data by cross validation fold and for the entire train and test set, were saved to blob storage for retrieval during the modeling phase. This is possible since our method of cross validation splitting is not random, so each fold is always the same. This method saved a large amount of time since it removed having to reprocess the data before training each model.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ####Modeling Pipeline Visual
# MAGIC 
# MAGIC <img src="https://github.com/kolbyjedevery/image_files/blob/main/final_modeling_pipeline.png?raw=true" width=50%>

# COMMAND ----------

# MAGIC %md
# MAGIC ####6. Loss function
# MAGIC Our outcome variable is a binary variable. That is, it can have a value of `1` for delayed and `0` for non-delayed. As we model, we will be using the binary cross entropy loss function as we train our models since this function lends itself very well to a binary classification problem. Equation below:
# MAGIC 
# MAGIC #####BCE = \\(-y\log(p) + (1 - y)\log(1 - p)\\)
# MAGIC 
# MAGIC ####7. Performance metrics
# MAGIC 
# MAGIC For the performance metrics, since our data has class imbalance, that is there are more non delays than delays for our outcome variable, we have decided to use F 0.5 score to evaluate all our models. We chose F 0.5 score, because our focus is to give more weight to precision and less to recall, since we would like to minimize false positives. Formula for the F 0.5 score is as follows:
# MAGIC 
# MAGIC ######F.5 = \\(1.5\frac{PrecisionRecall}{.25Precision+Recall}\\)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Model Building

# COMMAND ----------

# MAGIC %md
# MAGIC ##Logistic Regression and Random Forest Baseline
# MAGIC 
# MAGIC <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/2647101326228523/command/2647101326229236" target="_blank">Baseline Model</a>
# MAGIC 
# MAGIC We first created a baseline logistic regression model, using a small subset of basic features. This model resulted in a F0.5 score of 0.06. The purpose of this baseline model was to ensure that our pipeline was working well, we did not tune hyperparameters and just used a threshold value of 0.5 for the logistic regression. 
# MAGIC 
# MAGIC In addition, we created a random forest model (max depth=10,num_trees=64) using the same basic features. This model resulted in an F0.5 score of 0.03.
# MAGIC 
# MAGIC Clearly, based on the results of the baseline models, we needed to engineer more predictive features.
# MAGIC 
# MAGIC ##Baseline Gap Analysis
# MAGIC 
# MAGIC Our baseline was trained with a limited set of features to speed up computation time; see the baseline notebook linked above for a listing of features used. Our intent was simply to validate the functionality of our feature engineering, cross validation, and model training methods. This resulted in a logistic regression and random forest baseline F0.5 scores of 0.06 and 0.03, respectively.
# MAGIC 
# MAGIC This is of course lower than the only other baseline F0.5 score on the leaderboard which was 0.62 using logistic regression. Using our full set of features with some hyperparameter tuning should go a long way towards making up the difference. 

# COMMAND ----------

# MAGIC %md
# MAGIC ##Logistic Regression with all Features, Hyperparameter Tuning with Gridsearch
# MAGIC <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/632558266979148/command/632558266979149" target="_blank">Logistic Regression</a>
# MAGIC 
# MAGIC <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238241523/command/1215577238241524" target="_blank">Logistric Regression with Undersampling</a>
# MAGIC 
# MAGIC ####Hyperparameter Dictionary
# MAGIC ***threshold*** : Probability threshold
# MAGIC 
# MAGIC ***regParam*** : L2 regularization
# MAGIC 
# MAGIC ***elasticNetParam*** : L1 regularization
# MAGIC 
# MAGIC ***maxIter*** : Maximum iterations the model will run
# MAGIC 
# MAGIC ####Gridsearch
# MAGIC 
# MAGIC For Logistic Regression, we performed grid search over 3 values for the threshold, 5 for regParam, 4 for elasticNetParam, and 5 for maxIter. Through cross-validation, we identified the optimal values for all 4 hyperparameters. see the table below for a summary of results for both the model with and without undersampling applied. The hyperparameters for the best model are show below along with it's performance on the blind test set. 
# MAGIC 
# MAGIC Even with so many hyperparameters considered, and the subsequent large number of permutations, our execution times were reasonable. When 10 workers were available computation time was less than 30 minutes. When we were limited on number of workers allocated due to high demand this increased to almost 2 hours. 
# MAGIC 
# MAGIC The best tuned model produced comparable results with other models with a F0.5 score on the test set of 0.59.
# MAGIC 
# MAGIC ####Gridsearch Summary Table
# MAGIC |   | Model               | Class Imbalance Handling | Validation F0.5 Score | Best Model Hyperparameters                                   | treshhold     | regParam                | elasticNetParam             | maxIter          | Execution Time | Computation Resource              |
# MAGIC |---|---------------------|--------------------------|-----------------------|--------------------------------------------------------------|---------------|-------------------------|-----------------------------|------------------|----------------|---------------------------|
# MAGIC |   | Logistic Regression | None                     |                  0.60 | threshold=0.3, regParam=0.01, elasticNetParam=1.0, maxIter=5 | [0.3,0.5,0.8] | [0.01,0.1,0.5,1.0, 2.0] | [0.0, 0.25,0.50, 0.75, 1.0] | [1, 5,10,20, 50] | 1.74 hours     | 5-10 workers, 20-40 cores |
# MAGIC |   | Logistic Regression | Undersampling            |                  0.73 | threshold=0.5, regParam=0.01, elasticNetParam=1.0, maxIter=5 | [0.3,0.5,0.8] | [0.01,0.1,0.5,1.0, 2.0] | [0.0, 0.25,0.50, 0.75, 1.0] | [1, 5,10,20, 50] | 23.54 minutes  | 10 workers, 40 cores      |
# MAGIC 
# MAGIC ####Best Logistic Regression Model Score
# MAGIC 
# MAGIC |    | Model               | Class Imbalance Handling  | Test Data F0.5 Score | Hyperparameters                                              | Execution Time   | Computation Resources |
# MAGIC |----|---------------------|---------------------------|---------------------|--------------------------------------------------------------|------------------|-----------------------|
# MAGIC |    | Logistic Regression | None                      | 0.59                | threshold=0.3, regParam=0.01, elasticNetParam=1.0, maxIter=5 | 2.36 minutes     | 5 workers, 20 cores   |

# COMMAND ----------

# MAGIC %md
# MAGIC ##Random Forest with all Features, Hyperparameter Tuning with Gridsearch
# MAGIC <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238241570/command/1215577238241571" target="_blank">Random Forest</a>
# MAGIC 
# MAGIC <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238243132/command/1215577238243133" target="_blank">Random Forest with Undersampling</a>
# MAGIC 
# MAGIC ####Hyperparameter Dictionary
# MAGIC ***maxDepth*** : maximum number of splits each decision tree is allowed to make
# MAGIC 
# MAGIC ***numTrees*** : number of trees for the random forest 
# MAGIC 
# MAGIC ####Gridsearch
# MAGIC 
# MAGIC For Random Forest, we performed grid search over 2 values for maxDepth and 3 for numTrees. Through cross-validation, we identified the optimal values for both hyperparameters. See the table below for a summary of results for both the model with and without undersampling applied. The hyperparameters for the best model are show below along with it's performance on the blind test set. 
# MAGIC 
# MAGIC We limited the number of hyperparamenters to search to minimize computation time. Once again, when 10 workers were available our computation time were less than 30 minutes. However when we were resource constrained execution could take more than 2.5 hours. 
# MAGIC 
# MAGIC The best tuned random forest model produced the same result for F0.5 score on the test set as with logistic regression at 0.59.
# MAGIC 
# MAGIC ####Gridsearch Summary Table
# MAGIC |   | Model         | Class Imbalance Handling | Validation F0.5 Score| Best Model HyperParameters | maxDepth | numTrees    | Execution Time | Computation               |
# MAGIC |---|---------------|--------------------------|----------------------|----------------------------|----------|-------------|----------------|---------------------------|
# MAGIC |   | Random Forest | None                     |                 0.60 | maxDepth=10, numTrees=32   | [5,10]   | [32,64,128] | 2.54 hours     | 5-10 workers, 20-40 cores |
# MAGIC |   | Random Forest | Undersampling            |                 0.73 | maxDepth=10, numTrees=128  | [5,10]   | [32,64,128] | 23.54 minutes  | 10 workers, 40 cores      |
# MAGIC 
# MAGIC ####Best Random Forest Model Score
# MAGIC 
# MAGIC |    | Model               | Class Imbalance Handling  | Test Data F0.5 Score| HyperParameters                                              | Execution Time   | Computation Resources |
# MAGIC |----|---------------------|---------------------------|---------------------|--------------------------------------------------------------|------------------|-----------------------|
# MAGIC |    | Random Forest       | None                      | 0.59                | maxDepth=10, numTrees=32                                     | 13.56 minutes    | 10 workers, 40 cores  |

# COMMAND ----------

# MAGIC %md
# MAGIC ##Gradient Boosted Tree with all Features, Hyperparameter Tuning with Gridsearch
# MAGIC <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238245152/command/1215577238245153" target="_blank">GBT</a>
# MAGIC 
# MAGIC <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1860389250615615/command/1860389250615616" target="_blank">GBT with Undersampling</a>
# MAGIC 
# MAGIC ####Hyperparameter Dictionary
# MAGIC ***maxDepth*** : Maximum number of splits each decision tree is allowed to make
# MAGIC 
# MAGIC ***minInfoGain*** : Minimum information gain for a split to be considered at a tree node
# MAGIC 
# MAGIC ***maxBins*** : Max number of bins for discretizing continuous features
# MAGIC 
# MAGIC ####Gridsearch
# MAGIC 
# MAGIC For GBT, we performed grid search over 2 values for maxDepth, 3 for minInfoGain, and 3 for maxBins. Through cross-validation, we identified the optimal values for all hyperparameters. See the table below for a summary of results for both the model with and without undersampling applied. The hyperparameters for the best model are show below along with it's performance on the blind test set. The F.5 score of 0.6 without undersampling is consistent with the best model's score on the blind test data. 
# MAGIC 
# MAGIC When we trained this model 10 workers were available, so our computation times were less than 1 hour, which seems reasonable given the complexity of this model.
# MAGIC 
# MAGIC The best tuned GBT model failed to improve on the best result for logistic regression and Random Forest with a F0.5 score on the test set of 0.59.
# MAGIC 
# MAGIC ####Gridsearch Summary Table
# MAGIC |   | Model | Class Imbalance Handling | Validation F0.5 Score | Best Model HyperParameters            | maxDepth | minInfoGain   | maxBins | Execution Time | Computation          |
# MAGIC |---|-------|--------------------------|----------------------|---------------------------------------|----------|---------------|---------|----------------|----------------------|
# MAGIC |   | GBT   | None                     |                 0.60 | maxDepth=5, minInfoGain=0, maxBins=64 | [5,10]   | [0.0,0.2,0.4] | [32,64] | 42.78 minutes  | 10 workers, 40 cores |
# MAGIC |   | GBT   | Undersampling            |                 0.72 | maxDepth=5, minInfoGain=0, maxBins=64 | [5,10]   | [0.0,0.2,0.4] | [32,64] | 20.59 minutes  | 10 workers, 40 cores |
# MAGIC 
# MAGIC 
# MAGIC ####Best GBT Model Score
# MAGIC |    | Model               | Class Imbalance Handling  | Test Data F0.5 Score | HyperParameters                                              | Execution Time   | Computation Resources |
# MAGIC |----|---------------------|---------------------------|---------------------|--------------------------------------------------------------|------------------|-----------------------|
# MAGIC |    | GBT                 | None                      | 0.59                | maxDepth=5, minInfoGain=0, maxBins=64                        | 19.12 minutes    | 10 workers, 40 cores  |

# COMMAND ----------

# MAGIC %md
# MAGIC ##Multilayer Perceptron with all Features, Hyperparameter Tuning with Gridsearch
# MAGIC <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238245517/command/1215577238247251" target="_blank">MLPC Model</a>
# MAGIC 
# MAGIC ####Hyperparameter Dictionary
# MAGIC ***MaxIter*** :Maximum iterations the model will run
# MAGIC 
# MAGIC ***Layers*** :Layer sizes starting with input layer and ending with output layer
# MAGIC 
# MAGIC ***BlockSize*** : Block size for stacking input data
# MAGIC 
# MAGIC ***Solver*** : The optimization algorithm. This MLPC model tried 'gd'(gradient descent) and 'l-bfgs' (limited memory Broyden–Fletcher–Goldfarb–Shanno algorithm )
# MAGIC 
# MAGIC ####Gridsearch
# MAGIC For this model, we used all the same engineered features as the rest of the models, but also performed grid search over 3 values for the number of iterations (50,100,200), 2 values for layers (1 hidden layer of 26 nodes and 2 hidden layers of 26 nodes), 2 values for BlockSize (32,64), and 2 values for solver (gd, and l-bfgs). Through cross-validation, we found the optimal model to have a MaxIter of 100, Layers of [2,26,38], BlockSize of 64, and a solver of 'l-bfgs'. Full GridSearch table can be found in the appendix. 
# MAGIC 
# MAGIC This model resulted in an average F0.5 score of 0.724 across the five folds in the cross validation. After identifying these hyperparameters and training the MLPC model on the entire training data set, we evaluated its performance on the test set (2021 blind data). The model reulted in a F0.5 score of 0.59 on the test data, similar to all other models. 
# MAGIC 
# MAGIC ####Gridsearch Summary Table
# MAGIC |   | Model | Class Imbalance Handling | Validation F0.5 Score | Best Model HyperParameters                                           | maxIter      | layers                   | blockSize | solver           | Execution Time | Computation          |
# MAGIC |---|-------|--------------------------|----------------------|----------------------------------------------------------------------|--------------|--------------------------|-----------|------------------|----------------|----------------------|
# MAGIC |   | MLPC  | Undersampling            |                 0.72 | maxIter = 100, layers = [39,26,2], blockSize = 64, solver = 'l-bfgs' | [50,100,200] | [[38,26,2],[38,26,26,2]] | [32, 64]  | ['gd', 'l-bfgs'] | 2.44 hours     | 10 workers, 40 cores |
# MAGIC 
# MAGIC ####Best MLPC Model Score
# MAGIC 
# MAGIC |    | Model               | Class Imbalance Handling  | Test Data F0.5 Score | HyperParameters                                              | Execution Time   | Computation Resources |
# MAGIC |----|---------------------|---------------------------|---------------------|--------------------------------------------------------------|------------------|-----------------------|
# MAGIC |    | MLPC                | None                      | 0.59                | maxIter=100, layers=[39,26,2], blockSize=64, solver='l-bfgs' | 34.69 minutes    | 10 workers, 40 cores  |

# COMMAND ----------

# MAGIC %md
# MAGIC ##Results
# MAGIC 
# MAGIC The tables below, summarize the results of our best models with and without undersampling. The best F0.5 scores were nearly identical accross all models at 0.59. This indicates the need to revisit feature engineering. With our preprocessed data saved to the blob or execution times are reasonable enough to allow for more experimentation in the future.
# MAGIC 
# MAGIC We combined the predictions from each of these results into ensembles using majority or any vote for a delayed flight. Once again, we split the ensembles into versions that included undersampled models and those that didn't. No ensemble was able to substantially improve of F0.5 score.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Experiment Results Table
# MAGIC 
# MAGIC |    | Model               | Class Imbalance Handling  | Test Data F.5 Score | HyperParameters                                              | Execution Time   | Computation Resources |
# MAGIC |----|---------------------|---------------------------|---------------------|--------------------------------------------------------------|------------------|-----------------------|
# MAGIC |    | Logistic Regression | None                      | 0.59                | threshold=0.3, regParam=0.01, elasticNetParam=1.0, maxIter=5 | 2.36 minutes     | 5 workers, 20 cores   |
# MAGIC |    | Logistic Regression | Undersampling             | 0.56                | threshold=0.5, regParam=0.01, elasticNetParam=1.0, maxIter=5 | 24.83 seconds    | 10 workers, 40 cores  |
# MAGIC |    | Random Forest       | None                      | 0.59                | maxDepth=10, numTrees=32                                     | 13.56 minutes    | 10 workers, 40 cores  |
# MAGIC |    | Random Forest       | Undersampling             | 0.57                | maxDepth=10, numTrees=128                                    | 10.25 minutes    | 10 workers, 40 cores  |
# MAGIC |    | GBT                 | None                      | 0.59                | maxDepth=5, minInfoGain=0, maxBins=64                        | 19.12 minutes    | 10 workers, 40 cores  |
# MAGIC |    | GBT                 | Undersampling             | 0.56                | maxDepth=5, minInfoGain=0, maxBins=64                        | 14.88 minutes    | 10 workers, 40 cores  |
# MAGIC |    | MLPC                | None                      | 0.59                | maxIter=100, layers=[39,26,2], blockSize=64, solver='l-bfgs' | 34.69 minutes    | 10 workers, 40 cores  |
# MAGIC |    | MLPC                | Undersampling             | 0.55                | maxIter=100, layers=[39,26,2], blockSize=64, solver='l-bfgs' | 19.27 minutes    | 10 workers, 40 cores  |

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Ensemble Results Table
# MAGIC 
# MAGIC |    | Voting              | Class Imbalance Handling  | Test Data F.5 Score | 
# MAGIC |----|---------------------|---------------------------|---------------------|
# MAGIC |    | Majority Vote       | None and Undersampling    | 0.58                | 
# MAGIC |    | Majority Vote       | None                      | 0.59                | 
# MAGIC |    | Any Vote            | None and Undersampling    | 0.53                | 
# MAGIC |    | Any Vote            | None                      | 0.59                | 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Discussion
# MAGIC 
# MAGIC Ultimately, the F.05 score from our best model (Gradient Boosting Classifier) was ~0.59, which is mediocre at best. While we performed advanced feature engineering, hyperparameters tuning, and undersampling in an effort to improve model performance, these efforts ultimately had minimal effect. Still, we discuss the changes we made below and reason about why model performance is still poor.
# MAGIC 
# MAGIC ##### Model Comparison
# MAGIC 
# MAGIC The four types of models that we trained were Logistic Regression, Random Forest, Gradient Boosting Classifier, and a Multilayer Perceptron. We started with the logistic regression model, although we expected to need non-linear models, as there weren't too many features that had a strong pearson correlation with the target variable.
# MAGIC 
# MAGIC All the models we trained had similar performance in terms of F0.5 score, but the execution time for Logistic Regression was by far the shortest. This, however, may have been because compute traffic was low when that model was trained, compared to when other models were trained. We expected that Gradient Boosting Classifier to improve upon the performance of the Random Forest through its algorithm that focuses on mistakes of previous decision trees, but we did not see any noticeable change in performance. 
# MAGIC 
# MAGIC Finally, we built an ensemble model, which took into account the predictions of all four models. However, this did not improve performance, once again highlighting the fact that our features were likely not predictive of flight delays. 
# MAGIC 
# MAGIC 
# MAGIC ##### Feature Importance 
# MAGIC Below we see, feature importance scores for variables with non-zero importance scores from our best model (GBT without undersampling). These scores are the average of each feature's importance across all trees in the ensemble, as is suggested by Hastie et al. in *Elements of Statistical Learning*.
# MAGIC \
# MAGIC \
# MAGIC <img src="https://github.com/cmunugala/DataBricks_images/blob/main/feature_imoprtance.png?raw=true" width=60%; display=block; margin-left=auto; margin-right= auto; text-align=center>
# MAGIC 
# MAGIC As you can see from the figure above, the most important features were whether or not the previous flight was delayed, as well as the average proportion of delays for a given carrier in the last 24 hours. Surprisingly, our graph feature (PageRank) did not contribute much value to this model. 
# MAGIC 
# MAGIC 
# MAGIC Another takeaway from this chart is that many features are not important to our model, and thus if we expected overfitting, we could consider removing feature from our model. That being said, we doubt that this would have made a huge difference in model performance given that we tried L1 regularization (LASSO) of our logistic regression model and did not experience better results.  
# MAGIC 
# MAGIC ##### Leakage Analysis
# MAGIC 
# MAGIC After performing some initial feature engineering, we achieved a high F0.5 score of 0.92, which led us to believe that we were dealing with some leakage. We discovered that we had made a typo when creating the `Previous Flight Delay` flag, and had accidently included the label in the calculation of this feature. As a result, this feature was highly predictive of delays. By removing this typo, we fixed this leakage issue which in turn brought our model performance down. Furthermore, the `Delayed flight percentage of previous day at Origin` and `Delayed flight percentage of previous day at Destination` also had data leakage, addressed in the feature engineering section. We were able to adjust for the data leakage. After reviewing all of our features, our EDA strategy, and our modeling pipeline, we do not believe that we have any more data leakage. 
# MAGIC 
# MAGIC 
# MAGIC ##### Hyperparameter Tuning
# MAGIC 
# MAGIC During cross validation, we performed grid search of various hyperparameters across all four model types. 
# MAGIC 
# MAGIC In our Logistic Regression models, we optimized threshold, the regularization parameter, the elastic net parameter, and the maximum number of iterations. Because we had chosen F0.5 score as our metric, we expected that the threshold would be higher which would generally be better for improving precision. However, the optimal threshold was found to be 0.5. This result tells us that the model is not predicting delays with a high probability/confidence, and thus needs to keep the threshold lower to optimize performance. Once again, this points to our features not having enough predictive power. The optimal elastic net parameter was 1, which indicates that the model performed the best as a LASSO regression. The optimal maximum number of iterations was determined to be 5, meaning that the model was not getting much better with more training. 
# MAGIC 
# MAGIC In our Random Forest models, we optimized maxDepth, and the number of trees. The optimal maximum depth of a tree was determined to be 10. However, we only searched over two possible values (5 and 10). While it could be argued that it would have been beneficial to look over a wider range of values, we saw that our cross validation performance was higher than our test set, perhaps indicating that we were overfitting. If we had increased max depth even more, it would have led to a more complex model which could have worsened the overfitting problem. The optimal number of trees was determined to be 32 (as opposed to 64 or 128). This may be because we do not have a large number of features, and there is not much benefit to training additional decision trees. 
# MAGIC 
# MAGIC In our Gradient Boosting models, we optimized maxDepth, minimum info gain, and maxBins. Unlike in the random forest models, max depth of the trees was set to be 5. This optimal hyperparameter suggests that a complex tree was not needed to improve upon mistakes of previous trees. We believe that this once again points to the fact that only a few features in our feature space were predictive of flight delays. The optimal minimum info gain was 0, which could suggest either that the the information gain from splitting on our features was low, or that it was beneficial to have more than just a few splits in our tree. The optimal number of maxBins was determined to be 64, which suggests that it is useful to be able to discretize our continuous features into numerous bins.  
# MAGIC 
# MAGIC Finally, in our Multilayer Perceptron models, we optimized the maximum number of iterations, the number of hidden layers (1 or 2), block size, and the optimization algorithm. We see that the optimal number of hidden layers is 1, suggesting that a complex model is not required given our current feature set. The optimal block size was 64, suggesting that there was not any performance benefit to updating weights more often (but there was likely a benefit in terms of compute time). Finally, the maximum number of iterations or epochs was determined to be 100, which suggests that the model weights do not converge that quickly, but also do not improve after a certain point.  
# MAGIC 
# MAGIC 
# MAGIC ##### Class Imbalance
# MAGIC 
# MAGIC Most of the flights in our dataset were not delayed. However, we wanted our model to learn the characteristics of delayed flights as well as not delayed flights and not just to predict the higher probability class. Thus, we decided to use undersampling, which reduced the volume of data that we trained on, but theoretically might help us to predict delays better. However, in practice, we did not see any benefits from undersampling. In fact, we saw that the models trained without undersampling outperformed the models trained with undersampling.
# MAGIC 
# MAGIC One potential explanation for this is that by undersampling we are throwing away a lot of the data that helps the model to learn about what makes a flight delayed. 

# COMMAND ----------

# MAGIC %md
# MAGIC ##Phase 4 Gap Analysis
# MAGIC 
# MAGIC With advanced features, our best models achieved a F0.5 score of 0.59 on the test data set. This is quite similar to other groups using this metric - other group's received F0.5 scores of ~.52 to .62. Although, the group with the highest F0.5 score seems to have not yet updated their features, as their most important features looked to only be weather data. If they are getting this score with only weather data we would definitely be interested in their modeling pipeline and what other avenues they explored. 
# MAGIC 
# MAGIC On the whole, we appear to be leveraging many of the same features as other groups such as PageRank and previous delay flags. However, it is difficult to directly assess their implementation of features compared to ours since only a few groups are using the F0.5 metric.  

# COMMAND ----------

# MAGIC  %md
# MAGIC # Conclusion
# MAGIC 
# MAGIC The goal was to build models to predict flight delays 2 hours before the scheduled flight times for a delay of 15 mins or more. The focus was to tailor our predictions towards the passengers emphasizing on minimizing false positives and focusing on the precision. The metric of the choice was F 0.5 score. Our hypothesis was that with the large amount of data to our disposal, a model can be refined to reasonable effectiveness. We cleaned and joined three datasets incorporating historical flight information and weather from 2015-2021. After EDA and basic feature engineering, we split the date with year 2021 as a test set, and performed broken window cross validation on the train data set. 
# MAGIC 
# MAGIC For the baseline, we chose logistic regression and random forest models using only a few weather variables, resulting in a F.5-Score of 0.0605 and 0.0262 during validation respectively. For the advance models, we performed feature engineering and refined the pipeline. We choose to incorporate logistic regression, random forest, GBT, and MLPC, and an ensemble of the models. Due to the unbalanced nature of the outcome variable, for the aforementioned models, cross validation, parameter grid search, hyperparameter tuning, and advanced features with and without undersampling were investigated. Most of the advanced models performed relatively similar. The worst performing models with the lowest F 0.5 score were logistic regression, GBT, and esembling with undersampling with a F 0.5 scores of 0.56, 0.56, and 0.53, respectively. Logistic regression with no undersampling, Random Forest with no undersampling, GBT with no undersampling, MLPC with no undersampling, emsembling with majority vote and no undersampling, and esembling with any vote and no undersampling all result with a F 0.5 score of approximately 0.59. 
# MAGIC 
# MAGIC The theme here seems to be that undersampling drastically reduces the performance of our models. It would be interesting to explore different techniques for undersampling as well run the models using oversampling to account for the class imbalance. Furthermore, we believe that with more analysis and feature engineering, it would be possible to increase the F 0.5 score. A few new features that might help model performance would be to calculate Pagerank Score for the airlines, rolling average for airport and carrier delays with different times lines, and airline reputation.  

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Citations
# MAGIC 
# MAGIC https://www.faa.gov/data_research/aviation_data_statistics/media/cost_delay_estimates.pdf
# MAGIC 
# MAGIC https://www.npr.org/2022/11/27/1139327883/flights-delayed-canceled-holiday-travel-thanksgiving
# MAGIC 
# MAGIC https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4#:~:text=Cross%20Validation%20 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Appendix
# MAGIC 
# MAGIC #####GridSearch Results for MLPC Models of Undersampled Data
# MAGIC 
# MAGIC |    | MaxIter | Layers       | BlockSize | Solver | Cross Validation F.5 Score |
# MAGIC |----|---------|--------------|-----------|--------|----------------------------|
# MAGIC | 1  | 50      | [38,26,2]    | 32        | gd     | 0.4539                     |
# MAGIC | 2  | 50      | [38,26,2]    | 32        | l-bfgs | 0.7131                     |
# MAGIC | 3  | 50      | [38,26,2]    | 64        | gd     | 0.4538                     |
# MAGIC | 4  | 50      | [38,26,2]    | 64        | l-bfgs | 0.709                      |
# MAGIC | 5  | 50      | [38,26,26,2] | 32        | gd     | 0.369                      |
# MAGIC | 6  | 50      | [38,26,26,2] | 32        | l-bfgs | 0.6996                     |
# MAGIC | 7  | 50      | [38,26,26,2] | 64        | gd     | 0.369                      |
# MAGIC | 8  | 50      | [38,26,26,2] | 64        | l-bfgs | 0.7022                     |
# MAGIC | 9  | 100     | [38,26,2]    | 32        | gd     | 0.4855                     |
# MAGIC | 10 | 100     | [38,26,2]    | 32        | l-bfgs | 0.7236                     |
# MAGIC | 11 | 100     | [38,26,2]    | 64        | gd     | 0.4857                     |
# MAGIC | 12 | 100     | [38,26,2]    | 64        | l-bfgs | 0.724                      |
# MAGIC | 13 | 100     | [38,26,26,2] | 32        | gd     | 0.3817                     |
# MAGIC | 14 | 100     | [38,26,26,2] | 32        | l-bfgs | 0.7176                     |
# MAGIC | 15 | 100     | [38,26,26,2] | 64        | gd     | 0.3816                     |
# MAGIC | 16 | 100     | [38,26,26,2] | 64        | l-bfgs | 0.72                       |
# MAGIC | 17 | 200     | [38,26,2]    | 32        | gd     | 0.5562                     |
# MAGIC | 18 | 200     | [38,26,2]    | 32        | l-bfgs | 0.7203                     |
# MAGIC | 19 | 200     | [38,26,2]    | 64        | gd     | 0.5561                     |
# MAGIC | 20 | 200     | [38,26,2]    | 64        | l-bfgs | 0.7198                     |
# MAGIC | 21 | 200     | [38,26,26,2] | 32        | gd     | 0.4118                     |
# MAGIC | 22 | 200     | [38,26,26,2] | 32        | l-bfgs | 0.7204                     |
# MAGIC | 23 | 200     | [38,26,26,2] | 64        | gd     | 0.4116                     |
# MAGIC | 24 | 200     | [38,26,26,2] | 64        | l-bfgs | 0.7171                     |

# COMMAND ----------

