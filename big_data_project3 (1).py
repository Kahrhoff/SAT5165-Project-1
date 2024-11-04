from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import LongType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
import time
import pandas as pd
import ipaddress

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

#Fill resp_bytes
spark_df = spark_df.withColumn('resp_bytes', when(col('resp_bytes') == '-', None).otherwise(col('resp_bytes')).cast('float'))
median = spark_df.approxQuantile('resp_bytes', [0.5], 0)[0]
spark_df = spark_df.na.fill({'resp_bytes':median})

#Fill orig_bytes
spark_df = spark_df.withColumn('orig_bytes', when(col('orig_bytes') == '-', None).otherwise(col('orig_bytes')).cast('float'))
median = spark_df.approxQuantile('orig_bytes', [0.5], 0)[0]
spark_df = spark_df.na.fill({'orig_bytes':median})

#Define categories that need to be changed to classifications
cat_cols = ['proto', 'conn_state', 'history']

#Reduce the variance in features in history
history_counts = spark_df.groupBy('history').count()
replace = history_counts.filter(history_counts['count'] < 10).select('history').rdd.flatMap(lambda x: x).collect()
if replace:
    spark_df = spark_df.withColumn('history', F.when(F.col('history').isin(replace), 'Other').otherwise(F.col('history')))

#Label encode the columns
for col in cat_cols:
    indexer = StringIndexer(inputCol=col, outputCol=col+'_new')
    spark_df = indexer.fit(spark_df).transform(spark_df)
    spark_df = spark_df.drop(col)

#Split the data and limit how many datapoints from the malicious traffic is used
train_data, test_data = spark_df.randomSplit([0.8, 0.2], seed=42)
train_data = train_data.sampleBy('label', fractions={0: 0.8, 1: 0.2}, seed=42)
test_data = test_data.sampleBy('label', fractions={0: 0.8, 1: 0.2}, seed=42)

#Feature selection
feature_columns = [col for col in spark_df.columns if col not in ['label']]
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
scaler = StandardScaler(inputCol='features', outputCol='scaled_features')
train_data = assembler.transform(train_data)
scaler_model = scaler.fit(train_data)
train_data = scaler_model.transform(train_data)
test_data = assembler.transform(test_data)
test_data = scaler_model.transform(test_data)

#Start timer
start_time = time.time()

#Create the spark forest model
forest = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=100)
forest_model = forest.fit(train_data)
predictions = forest_model.transform(test_data)

#Evaluate the performance
evaluator_acc = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
evaluator_prec = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='weightedPrecision')
evaluator_rec = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='weightedRecall')
evaluator_auc = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='probability', metricName='areaUnderROC')

#Print Evaluation
accuracy = evaluator_acc.evaluate(predictions)
print('-----------------------')
print()
print(f'Accuracy: {accuracy}')
print()
print('-----------------------')

precision = evaluator_prec.evaluate(predictions)
print('-----------------------')
print()
print(f'Precision: {precision}')
print()
print('-----------------------')

recall = evaluator_rec.evaluate(predictions)
print('-----------------------')
print()
print(f'Recall: {recall}')
print()
print('-----------------------')

auc = evaluator_auc.evaluate(predictions)
print('-----------------------')
print()
print(f'Area Under Curve: {auc}')
print()
print('-----------------------')

#Find total time taken
end_time = time.time()
total_time = end_time - start_time
print('-----------------------')
print()
print(f'Time for execution: {total_time:.2f} seconds')
print()
print('-----------------------')
