# -*- coding: utf-8 -*-
"""Trying out PySpark.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J2_bToq88vwMEfR6CN_WxlMG03U9WR8U

Mounting Google Drive
"""

#from google.colab import drive
#drive.mount('/content/drive')

"""# **Spark Seesion Creation**"""
"""
!apt-get install openjdk-8-jdk-headless -qq > /dev/null

!wget -q https://archive.apache.org/dist/spark/spark-3.0.1/spark-3.0.1-bin-hadoop2.7.tgz

!tar xf spark-3.0.1-bin-hadoop2.7.tgz

!pip install -q findspark
"""

"""
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.0.1-bin-hadoop2.7"

import findspark
findspark.init()

findspark.find()

"""

#@title
from pyspark.sql import SparkSession

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

#@title
#!wget --continue https://raw.githubusercontent.com/GarvitArya/pyspark-demo/main/sample_books.json -O /tmp/sample_books.json

"""# Simple Data Frame examples"""

#@title
df = spark.read.json("/tmp/sample_books.json")

#@title
df.printSchema()

#@title
df.show(4,False)

#@title
df.count()

#@title
df.select("title", "price", "year_written").show(5)

#@title

"""# [Learning from here](https://sparkbyexamples.com/pyspark-tutorial/)"""

print(spark)

spark

rdd = spark.sparkContext.parallelize([1,2,3,4,5])
print("RDD count = ",rdd.count())

# Create RDD from parallelize    
dataList = [("Java", 20000), ("Python", 100000), ("Scala", 3000)]
rdd=spark.sparkContext.parallelize(dataList)

sc = spark.sparkContext

data = ["Project",
"Gutenberg’s",
"Alice’s",
"Adventures",
"in",
"Wonderland",
"Project",
"Gutenberg’s",
"Adventures",
"in",
"Wonderland",
"Project",
"Gutenberg’s"]

rdd=sc.parallelize(data)

rdd2=rdd.map(lambda x: (x,1))
for element in rdd2.collect():
    print(element)

data = [1,2,3,4,5]
rdd = sc.parallelize(data)
rdd2 = rdd.flatMap(lambda x: [x*x])
K = rdd2.collect()
print(K)



data = [1,2,3,4,5]*100000
rdd=spark.sparkContext.parallelize(data)

rddCollect = rdd.collect()
print("Number of Partitions: "+str(rdd.getNumPartitions()))
print("Action: First element: "+str(rdd.first()))
print(rddCollect)

emptyRDD = spark.sparkContext.emptyRDD()
emptyRDD2 = rdd=spark.sparkContext.parallelize([])

print(""+str(emptyRDD2.isEmpty()))



"""# Trying out Fitness Function using PySpark"""

#@title
import math
import random
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from scipy.special import expit

from numpy.random import default_rng

# noinspection SpellCheckingInspection

from threading import Thread
import time

def sigmoid(vectorSig):
  sig = expit(vectorSig)
  return sig

#@title
class InputData():
    # input and output
    # ================== Input dataset and corresponding output ========================= #
    def __init__(self):
        data = pd.read_csv("/content/iris.csv")

        output_values_expected = []
        input_values = []

        # ~~~~ encoding ~~~~#

        # labelencoder = LabelEncoder()
        # data[data.columns[-1]] = labelencoder.fit_transform(data[data.columns[-1]])

        # one hot encoding - for multi-column
        # enc = OneHotEncoder(handle_unknown='ignore')
        # combinedData = np.vstack((data[data.columns[-2]], data[data.columns[-1]])).T
        # print(combinedData)
        # y = enc.fit_transform(combinedData).toarray()
        # y = OneHotEncoder().fit_transform(combinedData).toarray()

        y = LabelBinarizer().fit_transform(data[data.columns[-1]])
        # print(y)

        # ~~~~ encoding ends~~~~#

        for j in range(len(data)):
            output_values_expected.append(y[j])

        # print(output_values_expected)

        for j in range(len(data)):
            b = []
            for i in range(1, len(data.columns) - 1):
                b.append(data[data.columns[i]][j])
            input_values.append(b)

        self.X = input_values[:]
        self.Y = output_values_expected[:]

    def main(self):
        return (self.X, self.Y)

i = InputData()
input_val, output_val = i.main()
dimension = [100,10]

#@title
def initialize(dimensions=dimension,
                 initialPopSize=100, iterations=10, gamma=0.001, beta_base=2.5,
                 alpha=0.2, alpha_damp=0.97, delta=0.05, exponent=2, m=5,
                 input_values=input_val, output_values_expected=output_val, seed=None):

        """
        Args:
            n_iterations (int): maximum number of iterations, default = 10
            initialPopSize (int): number of population size, default = 100
            gamma (float): Light Absorption Coefficient, default = 0.001
            beta_base (float): Attraction Coefficient Base Value, default = 2
            alpha (float): Mutation Coefficient, default = 0.2
            alpha_damp (float): Mutation Coefficient Damp Rate, default = 0.97
            delta (float): Mutation Step Size, default = 0.05
            exponent (int): Exponent (m in the paper), default = 2
        """

        initialPopSize = initialPopSize
        allPop_Weights = []
        allPopl_Chromosomes = []
        allPop_ReceivedOut = []
        allPop_ErrorVal = []
        n_iterations = iterations

        gamma = gamma  # (0, 1.0)
        beta_base = beta_base  # (0, 3.0)
        alpha = alpha  # (0, 1.0)
        alpha_damp = alpha_damp  # (0, 1.0)
        delta = delta  # (0, 1.0)
        exponent = exponent  # [2, 4]

        fitness = []

        distribution_factor = m

        # input and output
        X = input_values[:]
        Y = output_values_expected[:]

        dimensions = dimensions
        dimension = [len(X[0])]

        for i in dimensions:
            dimension.append(i)
        dimension.append(len(Y[0]))

        # print(self.dimension)

        # ================ Finding Initial Weights ================ #

        pop = []  # weights
        for g in range(initialPopSize):
            W = []
            for i in range(len(dimension) - 1):
                w = np.random.randint(-100, 100, (dimension[i + 1], dimension[i]))
                W.append(w)
            pop.append(W)

        init_pop = []  # chromosomes
        for W in pop:
            chromosome = []
            for w in W:
                chromosome.extend(w.ravel().tolist())
            init_pop.append(chromosome)
        return (init_pop, dimension)
pop, dimension = initialize()
print(len(pop))
print(len(pop[0]))

#@title
def mean_square_error(expected, predicted):
  total_error = 0.0
  for i in range(len(predicted)):
      total_error += ((predicted[i] - expected[i]) ** 2)
  return (-total_error)

#@title
def Fitness(population, X=input_val, Y=output_val):
  fitness = []
  for chromo in population:
    total_error = 0
    m = []
    k1 = 0
    for i in range(len(dimension) - 1):
        p = dimension[i]
        q = dimension[i + 1]
        k2 = k1 + p * q
        mtemp = chromo[k1:k2]
        m.append(np.reshape(mtemp, (p, q)))
        k1 = k2

    for x, y in zip(X, Y):

        yo = x

        for mCount in range(len(m)):
            yo = np.dot(yo, m[mCount])
            yo = sigmoid(yo)

        for i in range(len(yo)):
            total_error += mean_square_error(yo, y)

    fitness.append(total_error)

  return fitness

k = Fitness(pop)
print(len(k))

def func1(x): 
  for i in range(10):
    x = x+1
  return x

"""# **Simple Numpy example in PySpark**"""

def mult(x):
    y = np.array([2])
    k = x*y
    return k
 
x = np.arange(10000)
distData = sc.parallelize(x)
 
results = distData.map(mult).collect()
print(results)

temp = pop
data = temp[:]
print(data[0])
rdd = sc.parallelize(data)

K = rdd.map(mult).collect()
print(K)

"""# **Fitness in PySpark**"""

dimension=[4,100,10,3]

def Fitness(population, X=input_val, Y=output_val):
  fitness = []
  for chromo in population:
    total_error = 0
    m = []
    k1 = 0
    for i in range(len(dimension) - 1):
        p = dimension[i]
        q = dimension[i + 1]
        k2 = k1 + p * q
        mtemp = chromo[k1:k2]
        m.append(np.reshape(mtemp, (p, q)))
        k1 = k2


    for x, y in zip(X, Y):

        yo = x

        for mCount in range(len(m)):
            yo = np.dot(yo, m[mCount])
            yo = sigmoid(yo)

        for i in range(len(yo)):
            total_error += mean_square_error(yo, y)

    fitness.append(total_error)

  return fitness

#@title
def mult(x):
  fitness = []
  for chromo in x:
    total_error = 0
    m = []
    k1 = 0
    for i in range(len(dimension) - 1):
        p = dimension[i]
        q = dimension[i + 1]
        k2 = k1 + p * q
  y = np.array([2])
  k = x*y
  return k
temp = pop
data = temp[:]
rdd = sc.parallelize(data)

K = rdd.map(mult).collect()
print(K)

i = InputData()
X, Y = i.main()
dimension = [100,10]

def Fitness(x):
  fitness = []


  """ This code returns [array([-194]), array([-114]) ... ]    ==  The last value in each chromosome *look above👆*

  for a in x:
    k=np.array([2])*a
  """
  total_error = 0
  m = []
  k1 = 0
  mtemp=x[:]         

  for i in range(len(dimension) - 1):
    p = dimension[i]
    q = dimension[i + 1]
    k2 = k1 + p * q
    mt = mtemp[k1:k2]
    m.append(np.reshape(mt, (p, q)))

  for x, y in zip(X, Y):

    yo = x

    for mCount in range(len(m)):
        yo = np.dot(yo, m[mCount])
        yo = 1/(1 + np.exp(-yo))

    for i in range(len(yo)):
      
        total_error += mean_square_error(yo, y)

  return total_error

temp = pop
data = temp[:]

rdd = sc.parallelize(data)
#print(rdd)
#print(rdd.collect())
#print(rdd.first())
#print(rdd.take(2))

#rd2 = sc.parallelize([("p",5),("q",0),("r", 10),("q",3)])
#print(rd2.getNumPartitions())
#print(rd2.glom().collect())
#rd3 = rdd.union(rd2)


#rd11 = sc.parallelize(["a", "b", "c", "d", "e"])
#rdda = sc.parallelize([1, 2, 3, 4, 5])
#rda_11 = rdda.zip(rd11)
#rda_11.collect()


rdd2 = rdd.map(Fitness).collect()
print(rdd2)



"""# **Version 0.1 PySpark - GA ANN**"""





lines = spark.read.option("header", "true").option("inferSchema", "true").csv("file:///SparkCourse/fakefriends-header.csv")