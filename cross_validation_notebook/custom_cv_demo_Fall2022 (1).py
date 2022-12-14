# Databricks notebook source
# MAGIC %md
# MAGIC # PLEASE CLONE THIS NOTEBOOK INTO YOUR PERSONAL FOLDER IF NEEDED
# MAGIC # DO NOT RUN OR EDIT CODE IN THE SHARED FOLDER

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demo for using custom CV function
# MAGIC - LDC implementation of custom CV function described in this [blog post](https://www.timlrx.com/blog/creating-a-custom-cross-validation-function-in-pyspark)
# MAGIC - Custom CV has same functionality as PySpark CrossValidator - see documentation [here](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html)
# MAGIC - Toy model using 3m airline dataset

# COMMAND ----------

# import libraries
import pyspark.sql.functions as F
from pyspark.sql.types import *
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.ml.linalg import DenseVector, SparseVector, Vectors
from pyspark.ml.feature import VectorAssembler, StandardScaler

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Load Data

# COMMAND ----------

# inspect mount's final project folder
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project_2022"))

# COMMAND ----------

# load 3m airline dataset
df = spark.read.parquet("/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/")

# drop null values in outcome
df = df.na.drop(subset=['DEP_DEL15', 'DEP_DELAY_NEW'])

# keep only needed columns
include = ['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_DEL15']
df = df[include]

display(df)

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Prepare Data for Modeling

# COMMAND ----------

# create inpute vectors for modeling
feature_cols = ['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK']

assemble = VectorAssembler(inputCols=feature_cols, outputCol='features')
rf_input = assemble.transform(df) \
                   .withColumnRenamed('DEP_DEL15', 'label') \
                   .cache()

# sanity check
display(rf_input)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Run Grid Search with Custom CV
# MAGIC Steps:
# MAGIC - Import custom CV module
# MAGIC - Set up grid search
# MAGIC - Create dictionary of dataframes for custom CV function to loop through
# MAGIC - Run cross validation
# MAGIC - Retrieve best model from CV

# COMMAND ----------

# Import all functions from custom_cv module

# COMMAND ----------

# MAGIC %run "/Shared/custom_cv"

# COMMAND ----------

# set up grid search: estimator, set of params, and evaluator
rf = RandomForestClassifier(labelCol="label", featuresCol="features")
grid = ParamGridBuilder()\
            .addGrid(rf.maxDepth, [5, 10])\
            .addGrid(rf.numTrees, [10, 15])\
            .build()

# Example using F0.5 score for evaluator
evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=0.5)

# COMMAND ----------

# create dictionary of dataframes for custom cv fn to loop through
# assign train and test based on time series split 
# train: month 1, months 1 & 2, test: month 2, month 3
d = {}

d['df1'] = rf_input.filter(rf_input.MONTH <= 2)\
                   .withColumn('cv', F.when(rf_input.MONTH == 1, 'train')
                                         .otherwise('test'))

d['df2'] = rf_input.filter(rf_input.MONTH <= 3)\
                   .withColumn('cv', F.when(rf_input.MONTH <= 2, 'train')
                                         .otherwise('test'))

# COMMAND ----------

d['df1'].createOrReplaceTempView('tester_view')

# COMMAND ----------

d['df1'].groupby('MONTH','cv').count().orderBy('MONTH').show()
d['df2'].groupby('MONTH','cv').count().orderBy('MONTH').show()

# COMMAND ----------

display(d['df1'])

# COMMAND ----------

# run cross validation & return the crossvalidation F0.5 score for 'test' set
cv = CustomCrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator,
     splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)

cvModel = cv.fit(d)

# COMMAND ----------

rf_input

# COMMAND ----------

# make predictions
predictions = cvModel.transform(d['df1'])

display(predictions.groupby('label', 'prediction').count())

# COMMAND ----------



# COMMAND ----------

display(predictions)

# COMMAND ----------

