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

data_BASE_DIR = "dbfs:/mnt/mids-w261-joined/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# Read in training and test data

df_2015 = spark.read.parquet(f"{data_BASE_DIR}YEAR=2015/*")
df_2016 = spark.read.parquet(f"{data_BASE_DIR}YEAR=2016/*")
df_2017 = spark.read.parquet(f"{data_BASE_DIR}YEAR=2017/*")
df_2018 = spark.read.parquet(f"{data_BASE_DIR}YEAR=2018/*")
df_2019 = spark.read.parquet(f"{data_BASE_DIR}YEAR=2019/*")
df_2020 = spark.read.parquet(f"{data_BASE_DIR}YEAR=2020/*")
df_2021 = spark.read.parquet(f"{data_BASE_DIR}YEAR=2021/*")

# COMMAND ----------

display(df_2015)

# COMMAND ----------

df_2015.count()

# COMMAND ----------

df_2016.count()

# COMMAND ----------

df_2017.count()

# COMMAND ----------

df_2018.count()

# COMMAND ----------

df_2019.count()

# COMMAND ----------

df_2020.count()

# COMMAND ----------

df_2021.count()

# COMMAND ----------

total = df_2015.count() + df_2016.count() + df_2017.count() + df_2018.count() + df_2019.count() + df_2020.count() + df_2021.count()

# COMMAND ----------

total

# COMMAND ----------

