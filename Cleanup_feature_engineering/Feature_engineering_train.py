# Databricks notebook source
# MAGIC %md
# MAGIC #Final Feature Engineering - Train
# MAGIC #####Section 4 Group 2

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In this notebook, we are going to create new features and our reasoning behind why we chose said features. 

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

# #original datasets

# data_BASE_DIR = "dbfs:/mnt/mids-w261/"
# display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# # Inspect the Mount's Final Project folder 
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
# MAGIC # Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In this notebook, all feature engineering has been done on ***TRAIN_DF***

# COMMAND ----------

# read train dataset from merged table

display(dbutils.fs.ls(f"{blob_url}"))

df_train = spark.read.parquet(f"{blob_url}/merged_cleaned_data_train")
display(df_train)
df_train.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # 1- Creating weather types and sky conditions features.

# COMMAND ----------

# MAGIC %md 
# MAGIC We will use HourlyPresentWeatherType and HourlySkyConditions to produce weather conditions such as snow, rain as well as sky conditions such as broken, overcast, etc. We are doing this to be able to decipher HourlyPresentWeatherType and HourlySkyConditions column properly, since at their original state, they are not very decipheribal. 
# MAGIC 
# MAGIC We will further create a new column named route, which will essentially have origin-destination columns concatentated. This will help us in our EDA and possibly shed some light. 

# COMMAND ----------


def process_features(df):
  """
  Function to create new features for weather types, sky conditions, and route
  """
  
  # will null with empty string for HourlyPresentWeatherType and HourlySkyConditions

  df = df.na.fill("",["HourlyPresentWeatherType"]) \
         .na.fill("",["HourlySkyConditions"]) \
         .na.fill(0,["HourlyWindGustSpeed"])
  
  # route column, origin-dest pair
  # indicators for rain, snow, other weather conditions

  df = df.withColumn('Route', concat(col('ORIGIN'), lit('_'), col('DEST'))) \
         .withColumn('Rain', (instr(df.HourlyPresentWeatherType, 'RA') > 0).cast('int')) \
         .withColumn('Snow', (instr(df.HourlyPresentWeatherType, 'SN') > 0).cast('int')) \
         .withColumn('Thunder', (instr(df.HourlyPresentWeatherType, 'TS') > 0).cast('int')) \
         .withColumn('Fog', (instr(df.HourlyPresentWeatherType, 'FG') > 0).cast('int')) \
         .withColumn('Mist', (instr(df.HourlyPresentWeatherType, 'BR') > 0).cast('int')) \
         .withColumn('Freezing', (instr(df.HourlyPresentWeatherType, 'FZ') > 0).cast('int')) \
         .withColumn('Blowing', (instr(df.HourlyPresentWeatherType, 'BL') > 0).cast('int')) \
         .withColumn('Smoke', (instr(df.HourlyPresentWeatherType, 'FU') > 0).cast('int')) \
         .withColumn('Drizzle', (instr(df.HourlyPresentWeatherType, 'DZ') > 0).cast('int')) \
         .withColumn('Overcast', (instr(df.HourlySkyConditions, 'OVC') > 0).cast('int')) \
         .withColumn('Broken', (instr(df.HourlySkyConditions, 'BKN') > 0).cast('int')) \
         .withColumn('Scattered', (instr(df.HourlySkyConditions, 'SCT') > 0).cast('int')) 
  
  df = df.withColumn('CloudySkyCondition',when(col('Overcast')==1,1).when(col('Broken')==1,1).otherwise(0))
  
  return df

# COMMAND ----------

# process the train_df
processed_df_train = process_features(df_train)

#dropping two columns
processed_df_train = processed_df_train.drop('HourlyPresentWeatherType','HourlySkyConditions')
display(processed_df_train)

# COMMAND ----------

#add index for use in train/validation folds
processed_df_train = processed_df_train.select("*", f.row_number().over(Window.partitionBy().orderBy("Date_Time_sched_dep_utc")).alias("Index"))

display(processed_df_train)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 2- Creating Holiday Period Feature

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Next we decided to create a new feature called holiday period feature, because we understand that traveling and therefore delays might be affected in these particular times. The holiday periods were taken from https://www.officeholidays.com/countries/usa \
# MAGIC For thanksgiving and Christmas, +/- 1 days were also included as a holiday period due to travel. For New years one day after was also included as a holiday period due to travel. 

# COMMAND ----------

def holiday_indicator(data):
  """
  Adds a holiday range indicator.
  """
  
  if data == processed_df_train:
 
 #     train_df1 = processed_df_train.withColumn('FL_Date', concat(col('YEAR'), lit('-'), col('MONTH'), lit('-'), col('DAY_OF_MONTH'))) 
    train_df2 = processed_df_train.withColumn('holiday_period', expr("""CASE WHEN FL_Date in ('2015-1-1','2015-1-2','2015-1-19', '2015-5-25','2015-7-3',
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
    
  if data == processed_df_test:
#     test_df1 = test_df.withColumn('FL_Date', concat(col('YEAR'), lit('-'), col('MONTH'), lit('-'), col('DAY_OF_MONTH')))
 
    test_df2 = processed_df_test.withColumn('holiday_period', expr("""CASE WHEN FL_Date in ('2021-1-1','2021-1-2','2021-1-18','2021-5-31',
                                                                        '2021-7-5','2021-9-6','2021-11-24','2021-11-25','2021-11-26',
                                                                        '2021-12-24','2021-12-25','2021-12-26') THEN '1' """ 
                                          "ELSE '0' END"))
    return test_df2


# COMMAND ----------

#processing training df 
processed_df_train1 = holiday_indicator(processed_df_train)
display(processed_df_train1)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 3- Class Imbalance Functions

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Since the data contains class imbalance for the outcome variable (i.e. more non-delays than delays), we have created three functions to address class imbalance. We will be using these function to address imbalance while we are building our models

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

# MAGIC %md
# MAGIC 
# MAGIC # 4- Average Delay Per Carrier

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Next, we decided to find out the average carrier delay 26 hours before the scheduled departure time, since having a higher average delay for a carrier might affect future delays. We are choosing 26 hrs prior to the scheduled departure time due to the model prediction that comes two hours before scheduled departure time. This will result in no information leakage in calculating the average delay per carrier.

# COMMAND ----------

processed_df_train2 = processed_df_train1.drop('Index')
partition = Window.partitionBy('OP_CARRIER')
 
parition1 = partition.orderBy(f.unix_timestamp('two_hrs_pre_flight_utc')).rangeBetween(-86400, -1)
 
processed_df_train2 = processed_df_train2.withColumn('mean_carrier_delay', f.avg('DEP_DEL15').over(parition1))
display(processed_df_train2)

# COMMAND ----------

col_to_look_at = processed_df_train2.select('OP_CARRIER','TAIL_NUM','ORIGIN','DEST','CRS_ELAPSED_TIME','DISTANCE','DISTANCE_GROUP','ELEVATION','mean_carrier_delay')
null_values = col_to_look_at.select([count(when(col(c).isNull(), c)).alias(c) for c in col_to_look_at.columns]).toPandas()
null_values

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # 5-Page Rank

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Next, we will apply pagerank on the train_df, pagerank score should be representative of which airports are the busiest since those nodes will converge to a higher mass

# COMMAND ----------

# select origin and dest only
node_df = processed_df_train2.select('ORIGIN', 'DEST')

# convert to RDD
airport_rdd = node_df.rdd

# convert to same format 
def format(line):
  origin = line[0]
  dest = line[1]
  yield ((origin, dest), 1)
         
airRDD = airport_rdd.flatMap(format).reduceByKey(lambda x,y: x+y)

# cache rdd
airRDD.cache()

def matchformat(line):
  node = line[0][0]
  edge = line [0][1]
  count = line[1]
  edge_dict = {edge:count}
  yield (node, str(edge_dict))


# COMMAND ----------

airRDD1 = airRDD.flatMap(matchformat)

# initialize graph
 
def initGraph(dataRDD):
    """
    Spark job to read in the raw data and initialize an 
    adjacency list representation with a record for each
    node (including dangling nodes).
    
    Returns: 
        graphRDD -  a pair RDD of (node_id , (score, edges))
        
    NOTE: The score should be a float, but you may want to be 
    strategic about how format the edges... there are a few 
    options that can work. Make sure that whatever you choose
    is sufficient for Question 8 where you'll run PageRank.
    """
    
    def parse(line):
        node = line[0]
        edges = line[1]
        return (node, ast.literal_eval(edges))
 
    def getEdgeNodes(line):
        yield line[0]
    
    def getAllNodes(line):
        yield line[0]
        part2 = set(line[1].keys())
        for p in part2:
            yield p
            
    def dict_to_string(d):
        '''from stackoverflow'''
        edge_dict = d
        res = []
        for key, val in edge_dict.items():
            res += [key] * val
        return res
    
    def emitEntries(line):
        node = line[0]
        edges = dict_to_string(line[1])
        yield (node, (1.0/N.value, edges))
 
    # write your main Spark code here
    
    # parse and cache dataRDD
    parsed_rdd = dataRDD.map(parse).cache()
    
    # get all nodes
    all_nodes = parsed_rdd.flatMap(getAllNodes).distinct().cache() 
    
    #broadcast count of all nodes
    N = sc.broadcast(all_nodes.count())
    
    # get list of edge nodes
    nodes_with_edges = parsed_rdd.flatMap(getEdgeNodes).cache()
    
    # subtract to get dangling nodes, map proper format
    nodes_wo_edgesRDD = all_nodes.subtract(nodes_with_edges) \
                            .map(lambda x: (x, (1.0/N.value, [])))
    
    # map proper format on all edge nodes
    nodes_w_edgesRDD = parsed_rdd.flatMap(emitEntries)
 
    # combine edges and non edges rdds
    graphRDD = nodes_w_edgesRDD \
                    .union(nodes_wo_edgesRDD) \
                    .reduceByKey(lambda x,y : x+y)
    
    return graphRDD

# COMMAND ----------

import re
import ast
import time
 
start = time.time()
flightGraph = initGraph(airRDD1)
print(f'... test graph initialized in {time.time() - start} seconds.')

# COMMAND ----------

# custom accumulator
from pyspark.accumulators import AccumulatorParam
 
class FloatAccumulatorParam(AccumulatorParam):
    """
    Custom accumulator for use in page rank to keep track of various masses.
    
    IMPORTANT: accumulators should only be called inside actions to avoid duplication.
    We stringly recommend you use the 'foreach' action in your implementation below.
    """
    def zero(self, value):
        return value
    def addInPlace(self, val1, val2):
        return val1 + val2

# COMMAND ----------

# pagerank calculation
def runPageRank(graphInitRDD, alpha = 0.15, maxIter = 10, verbose = True):
    """
    Spark job to implement page rank
    Args: 
        graphInitRDD  - pair RDD of (node_id , (score, edges))
        alpha         - (float) teleportation factor
        maxIter       - (int) stopping criteria (number of iterations)
        verbose       - (bool) option to print logging info after each iteration
    Returns:
        steadyStateRDD - pair RDD of (node_id, pageRank)
    """
    # teleportation:
    a = sc.broadcast(alpha)
    
    # damping factor:
    d = sc.broadcast(1-a.value)
    
    # initialize accumulators for dangling mass & total mass
    mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    
    ############## YOUR CODE HERE ###############  
    # write your helper functions here, 
    # please document the purpose of each clearly 
    # for reference, the master solution has 5 helper functions.
    
    def getNoInNodes(line):
        '''produce an rdd of all nodes with their graph structure and zero mass'''
        node = line[0]
        edges = line[1][1]
        yield(node, (edges, 0.0))
            
    def parseNeighbors(line):
        """Parses payload to get neighbor list in correct format"""
        node = line[0]
        rank = line[1][0]
        neighbors = line[1][1]
        yield(node, (neighbors, rank))
 
    def computeContribs(line):
        """Calculates neighbor contributions to the rank of others."""
        node = line[0]
        neighbors = line[1][0]
        rank = line[1][1]
        num_neigh = len(neighbors)
        if neighbors != []:
            for n in neighbors:
                yield (n, ([], rank / num_neigh))
                #totAccum.add(rank / num_neigh)
            
    def computeDanglingMass(line, mmAccum):
        """Adds dangling mass to accumulator"""
        neighbors = line[1][0]
        rank = line[1][1]
        if neighbors == []:
            #totAccum.add(rank)
            mmAccum.add(rank)        
 
  #     write your main Spark Job here (including the for loop to iterate)
  #     for reference, the master solution is 21 lines including comments & whitespace
 
    N = sc.broadcast(graphInitRDD.count())
    
    # rdd of edges for mass to be distributed
    currentRDD = graphInitRDD.flatMap(parseNeighbors).cache()
        
    # all nodes RDD
    allNodesRDD = graphInitRDD.flatMap(getNoInNodes).cache() 
    
    for i in range(maxIter):
        #for each entry, check if there are no edges and add to dangling mass, broadcast value
        danglingmass = currentRDD.foreach(lambda x: computeDanglingMass(x, mmAccum))
        mmAccum_bc = sc.broadcast(mmAccum.value)
 
        #distributed mass RDD  
        distRDD = currentRDD.flatMap(computeContribs) \
                            .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])).cache()
        # join all nodes with structure to distributed RDD, reduce, and distribute mass with teleport 
        stateRDD = allNodesRDD.union(distRDD) \
                                    .reduceByKey(lambda x,y : (x[0]+y[0], x[1]+y[1])) \
                                    .mapValues(lambda x: (x[0], (((x[1] + mmAccum_bc.value/N.value) * d.value) + (a.value * (1/N.value)))))
 
        if verbose:
            print('Iteration #:', i)
            print("Dangling Mass:", mmAccum_bc.value)
#             print("Total Mass:", totAccum.value)             
 
        currentRDD = stateRDD.cache()
        mmAccum.value = 0.0
 
    steadyStateRDD = stateRDD.map(lambda x: ((x[0]), x[1][1])).cache() 
    ############## (END) YOUR CODE ###############
    
    return steadyStateRDD


# COMMAND ----------

nIter = 20
start = time.time()
full_results = runPageRank(flightGraph, alpha = 0.15, maxIter = nIter, verbose = True)
print(f'\n...trained {nIter} iterations in {time.time() - start} seconds.\n')

# COMMAND ----------

print(f'Top 40 ranked nodes:\n')
top_40 = full_results.takeOrdered(40, key=lambda x: -x[1])
# print results from 20th to 40th highest PageRank
for result in top_40:
    print(result)

# COMMAND ----------

pagerank_df = full_results.toDF()
pagerank_df = pagerank_df.withColumnRenamed("_1","ICAO")
pagerank_df = pagerank_df.withColumnRenamed("_2","Pagerank_Score")
 
display(pagerank_df)

# COMMAND ----------

# joing the pagrank_df to processed_df_train2

processed_df_train3 = processed_df_train2.join(pagerank_df, processed_df_train2.ORIGIN == pagerank_df.ICAO).drop('ICAO')

# COMMAND ----------

display(processed_df_train3)

# COMMAND ----------

# Save pagerank scores
pagerank_df.write.mode("overwrite").parquet(f"{blob_url}/pagerank_scores")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 6- Previous Flight Delay Feature

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Next, we created a new feature for whether previous flight was delayed. The previous flight was found by partitioning the dataset by TAIL_NUM and Scheduled Departure time, then using a lag function to find the previous flight. Logic was also built in to ensure the following conditions:
# MAGIC 
# MAGIC 1 . The previous flight was only considered delayed, if it's actual arrival was less than 90 minutes before the scheduled departure of the same flight. This was done to remove cases where the previous flight was delayed, but there was enough buffer time between flights that the previous delayed did not likely have an impact on the next flight.
# MAGIC 
# MAGIC 2. Previous flight delayed arrival time is before the current scheduled flight's departure time.
# MAGIC 
# MAGIC 3. The tail number of the plane matches, and the previous flight destination matches the current flight departure
# MAGIC  
# MAGIC See this notebook for full code for the implementation: <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1325974983871036/command/1325974983871037" target="_blank">Last Flight Delayed Feature Link</a>

# COMMAND ----------

# read previous flight delayed results from from blob

df_prev_delayed = spark.read.parquet(f"{blob_url}/previous_flight_delayed")
display(df_prev_delayed)

# COMMAND ----------

# merge previous flight delayed

processed_df_train4 = processed_df_train3.join(df_prev_delayed, 
                                              (processed_df_train3["TAIL_NUM"] == df_prev_delayed["PREV_TAIL_NUM"]) & 
                                              (processed_df_train3["ORIGIN"] == df_prev_delayed["PREV_ORIGIN"]) &
                                              (processed_df_train3["two_hrs_pre_flight_utc"] == df_prev_delayed["PREV_two_hrs_pre_flight_utc"])
                                               ,"left")

# drop columns brought in as a byproduct of the join
processed_df_train4 = processed_df_train4.drop('PREV_DEP_DEL15') \
                                                .drop('PREV_two_hrs_pre_flight_utc') \
                                                .drop('PREV_TAIL_NUM') \
                                                .drop('PREV_ORIGIN')


display(processed_df_train4)
processed_df_train4.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # 7- Number of flights for Prior Day, Percent Delayed for Prior Day

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we wanted to look at the total number of fligth per prior day for origin and destination airports as well percent delays for both origin and destination. 
# MAGIC 
# MAGIC All data was for the previous day to avoid data leakage.
# MAGIC 
# MAGIC See this notebook for full code for the implementation: <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/4248541777006369/command/2647101326234944" target="_blank">Prior Day Flight Count and Percent Delayed</a>

# COMMAND ----------

# read previous flight count and percent delayed results from from blob, process for merge

df_num_flights_delay = spark.read.parquet(f"{blob_url}/number_flights_and_delay_rate")
df_num_flights_delay = df_num_flights_delay.select('TAIL_NUM', 'ORIGIN', 'two_hrs_pre_flight_utc', 'origin_flight_per_day', 'origin_delays_per_day', 
                                                   'dest_flight_per_day', 'dest_delays_per_day', 'origin_percent_delayed', 'dest_percent_delayed')

df_num_flights_delay = df_num_flights_delay.withColumnRenamed('TAIL_NUM', 'PREV_TAIL_NUM') \
                                          .withColumnRenamed('ORIGIN', 'PREV_ORIGIN') \
                                          .withColumnRenamed('two_hrs_pre_flight_utc', 'PREV_two_hrs_pre_flight_utc')
display(df_num_flights_delay)

# COMMAND ----------

# merge origin and destination flights per day, delays per day, percent delayed

processed_df_train5 = processed_df_train4.join(df_num_flights_delay, 
                                              (processed_df_train4["TAIL_NUM"] == df_num_flights_delay["PREV_TAIL_NUM"]) & 
                                              (processed_df_train4["ORIGIN"] == df_num_flights_delay["PREV_ORIGIN"]) &
                                              (processed_df_train4["two_hrs_pre_flight_utc"] == df_num_flights_delay["PREV_two_hrs_pre_flight_utc"])
                                               ,"left")

# drop columns brought in as a byproduct of the join
processed_df_train5 = processed_df_train5.drop('PREV_two_hrs_pre_flight_utc') \
                                                .drop('PREV_TAIL_NUM') \
                                                .drop('PREV_ORIGIN')


display(processed_df_train5)
processed_df_train5.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # 8- Prophet Forecast on Percent Delayed

# COMMAND ----------

# MAGIC %md
# MAGIC We ran Facebook Prophet on the time series of FL_DATE and percent flights delayed by airport, this gives us a predicted percent delayed and trend to use as additional features.
# MAGIC 
# MAGIC See this notebook for full code for the implementation: <a href="https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/4248541777006369/command/2647101326234944" target="_blank">Prophet Percent Delayed</a>

# COMMAND ----------

# read prophet results from from blob, process for merge

df_prophet = spark.read.parquet(f"{blob_url}/prophet_delay_rate")

df_prophet = df_prophet.select('ds', 'trend', 'yhat', 'IATA')

df_prophet = df_prophet.withColumnRenamed('trend', 'Prophet_trend') \
                       .withColumnRenamed('yhat', 'Prophet_pred') \
                       .withColumnRenamed('IATA', 'P_IATA') \
                       .withColumn('P_FL_DATE', f.to_date(f.col('ds')))
                       
df_prophet = df_prophet.drop('ds')

display(df_prophet)

# COMMAND ----------

# merge prophet results on Origin and Destination

processed_df_train6 = processed_df_train5.join(df_prophet, 
                                              (processed_df_train5["ORIGIN"] == df_prophet["P_IATA"]) &
                                              (processed_df_train5["FL_DATE"] == df_prophet["P_FL_DATE"])
                                               ,"left")

# drop columns brought in as a byproduct of the join
processed_df_train6 = processed_df_train6.withColumnRenamed('Prophet_trend', 'ORIGIN_Prophet_trend') \
                                         .withColumnRenamed('Prophet_pred', 'ORIGIN_Prophet_pred') \
                                         .drop("P_IATA") \
                                         .drop("P_FL_DATE")

processed_df_train7 = processed_df_train6.join(df_prophet, 
                                              (processed_df_train6["DEST"] == df_prophet["P_IATA"]) &
                                              (processed_df_train6["FL_DATE"] == df_prophet["P_FL_DATE"])
                                               ,"left")

# drop columns brought in as a byproduct of the join
processed_df_train7 = processed_df_train7.withColumnRenamed('Prophet_trend', 'DEST_Prophet_trend') \
                                         .withColumnRenamed('Prophet_pred', 'DEST_Prophet_pred') \
                                         .drop("P_IATA") \
                                         .drop("P_FL_DATE")


display(processed_df_train7)
processed_df_train7.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Blob Checkpoint Post Feature-Engineering

# COMMAND ----------

# save to blob
processed_df_train7.write.mode("overwrite").parquet(f"{blob_url}/train_data_with_adv_features")


# COMMAND ----------

