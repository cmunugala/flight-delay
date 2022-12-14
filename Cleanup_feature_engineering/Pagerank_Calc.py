# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering Tests
# MAGIC 
# MAGIC ### - Pagerank 

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
# MAGIC 
# MAGIC ### Pagerank

# COMMAND ----------

# Read merged data data

df = spark.read.parquet(f"{blob_url}/merged_cleaned_data")

# COMMAND ----------

display(df)

# COMMAND ----------

df.count()

# COMMAND ----------

df = df.dropDuplicates()
display(df)

# COMMAND ----------

df.count()

# COMMAND ----------

train_df = df.filter(df.YEAR < 2021)

# COMMAND ----------

train_df.count()

# COMMAND ----------

# select origin and dest only

node_df = train_df.select('ORIGIN', 'DEST')
display(node_df)
node_df.count()

# COMMAND ----------

# convert to RDD

airport_rdd = node_df.rdd
airport_rdd.take(5)

# COMMAND ----------

# convert to same format

def format(line):
  origin = line[0]
  dest = line[1]
  yield ((origin, dest), 1)
         
airRDD = airport_rdd.flatMap(format).reduceByKey(lambda x,y: x+y)

# COMMAND ----------

# cache rdd, reduced down to 8298 rows
airRDD.cache()

# COMMAND ----------

def matchformat(line):
  node = line[0][0]
  edge = line [0][1]
  count = line[1]
  edge_dict = {edge:count}
  yield (node, str(edge_dict))

airRDD1 = airRDD.flatMap(matchformat)

# COMMAND ----------

airRDD1.take(5)

# COMMAND ----------

# from HW5 wikiRDD Format

# 'node' \t {'edge' : number}

# ["2\t{'3': 1}",
#  "3\t{'2': 2}",
#  "4\t{'1': 1, '2': 1}",
#  "5\t{'4': 3, '2': 1, '6': 1}",
#  "6\t{'2': 1, '5': 2}",
#  "7\t{'2': 1, '5': 1}",
#  "8\t{'2': 1, '5': 1}",
#  "9\t{'2': 1, '5': 1}",
#  "10\t{'5': 1}",
#  "11\t{'5': 2}"]

# COMMAND ----------

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
    ############## YOUR CODE HERE ###############

    # write any helper functions here
    
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

    ############## (END) YOUR CODE ##############
    
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

nIter = 30
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

# write dataframe to blob
pagerank_df.write.mode("overwrite").parquet(f"{blob_url}/pagerank_scores")

# COMMAND ----------

display(dbutils.fs.ls(f"{blob_url}"))

# COMMAND ----------

