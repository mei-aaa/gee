#coding=utf-8
import pyspark
import os

os.environ["PYSPARK_DRIVER_PYTHON"]="/usr/bin/python3"
os.environ["PYSPARK_PYTHON"]="/usr/bin/python3"

from pyspark import SparkConf,SparkContext
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

conf=SparkConf().setMaster("local").setAppName("My App")
#sc = SparkContext(conf =conf)
sc = SparkContext(conf=conf)

spark=SparkSession.builder.master("local").appName("My App").config(conf=SparkConf()).getOrCreate()

lines=sc.textFile("/home/mei/Documents/ex-3/")
data1=spark.createDataFrame(lines,"string")
import pyspark.sql.functions as F
g3=data1.select(F.get_json_object("value","$.request_time").alias("request_time"),\
    F.get_json_object("value","$.captcha_id").alias("captcha_id"))
g4=g3.select(F.substring("request_time",0,19).alias("request_time"),"captcha_id").\
      filter("captcha_id is not NULL").\
      groupby(["captcha_id","request_time"]).count().withColumnRenamed("count","qps").\
      orderBy(["captcha_id","request_time"],ascending=[0, 1])

g4.coalesce(1).write.option("header", "true").csv("/home/mei/Documents/qps.csv")