from pyspark.sql import SparkSession
import ipaddress
import numpy as np
import pandas as pd
import glob
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import LongType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

def encode_ip(ip):
    return int(ipaddress.IPv4Address(ip))

spark = SparkSession.builder.getOrCreate()

#Create the dataframe
df = pd.read_excel('/home/sat3812/Downloads/log.xlsx')
spark_df = spark.createDataFrame(df)

#Drop Detailed Label as it is only ever Comamnd and Control in this example
spark_df = spark_df.drop('detailed-label')

#Drop entirely empty columns (used in other samples, but not this one)
spark_df = spark_df.drop('service', 'local_orig', 'local_resp', 'tunnel_parents')

#Drop unnecessary columns
spark_df = spark_df.drop('uid', 'ts')

#Encode IP addresses into readable numbers
convert_ip = udf(encode_ip, LongType())
spark_df = spark_df.withColumn('id_orig_h', convert_ip(spark_df['id_orig_h']))
spark_df = spark_df.withColumn('id_resp_h', convert_ip(spark_df['id_resp_h']))

#Fill duration with median values and get whole list as func
spark_df = spark_df.withColumn('duration', when(col('duration') == '-', None).otherwise(col('duration')).cast('float'))
median = spark_df.approxQuantile('duration', [0.5], 0)[0]
spark_df = spark_df.na.fill({'duration':median})

#Fill resp_buytes
spark_df = spark_df.withColumn('resp_bytes', when(col('resp_bytes') == '-', None).otherwise(col('resp_bytes')).cast('float'))
median = spark_df.approxQuantile('resp_bytes', [0.5], 0)[0]
spark_df = spark_df.na.fill({'resp_bytes':median})

#Fill orig_bytes
spark_df = spark_df.withColumn('orig_bytes', when(col('orig_bytes') == '-', None).otherwise(col('orig_bytes')).cast('float'))
median = spark_df.approxQuantile('orig_bytes', [0.5], 0)[0]
spark_df = spark_df.na.fill({'orig_bytes':median})

cat_cols = ['proto', 'conn_state', 'history']

#Reduce the variance in features in history
history_map = spark_df.groupBy('history').count()
history_map = history_map.withColumn('others', when(col('count') < 10, 'Other').otherwise(col('history')))
spark_df = spark_df.join(history_map, on='history', how='left')
spark_df = spark_df.withColumn('history', when(col('others') == 'Other', 'Other').otherwise(col('history')))

#Hot Encode remaining category features
for col in cat_cols:
    indexer = StringIndexer(inputCol = col, outputCol = col + '_ind')
    spark_df = indexer.fit(spark_df).transform(spark_df)
    encoder = OneHotEncoder(inputCol = col + '_ind', outputCol=col + '_hot')
    spark_df = encoder.fit(spark_df).transform(spark_df)
    spark_df = spark_df.withColumn(col, F.col(col + '_hot'))
    spark_df = spark_df.drop(col + '_ind', col + '_hot')

#Convert to strings the purpose of saving
spark_df = spark_df.withColumn('history', F.expr('CAST(history AS STRING)'))
spark_df = spark_df.withColumn('proto', F.expr('CAST(proto AS STRING)'))
spark_df = spark_df.withColumn('conn_state', F.expr('CAST(conn_state AS STRING)'))

final_df = spark_df
final_df.write.csv('/home/sat3812/Downloads/temp', header=True, mode='overwrite')
temp = glob.glob('/home/sat3812/Downloads/temp/part-*')
final = pd.concat((pd.read_csv(f) for f in temp), ignore_index=True)
final.to_excel('/home/sat3812/Downloads/final-log.xlsx', index=False)
