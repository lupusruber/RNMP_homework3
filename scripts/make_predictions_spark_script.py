from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructField, FloatType, StructType
from pyspark.ml.feature import VectorAssembler
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).parent.parent
MODEL_PATH = f"{PROJECT_ROOT_PATH}/model/random_forest.pkl"

BOOTSTRAP_SERVERS = "kafka:9092"
INPUT_TOPIC = "health_data_topic"
OUTPUT_TOPIC = "health_data_topic_results_2"

# os.environ["PYSPARK_SUBMIT_ARGS"] = (
#     "--packages "
#     "org.apache.spark:spark-streaming-kafka-0-10_2.12:3.2.0,"
#     "org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0,"
#     "pyspark-shell"
# )


def get_spark_session() -> SparkSession:

    spark = (
        SparkSession.builder.master("local[*]").appName("Streaming App").getOrCreate()
    )

    return spark


def get_kafka_data_stream(spark, topic, schema, bootstrap_servers):

    stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .load()
    )

    data_stream = (
        stream.selectExpr("CAST(value as STRING) as value")
        .select(F.from_json("value", schema=schema).alias("data"))
        .select("data.*")
    )

    return data_stream


def predict(features):
    feature_array = np.array(features).reshape(1, -1)
    prediction = model.predict(feature_array)
    return float(prediction[0])


def send_predicitons_to_kafka(predictions, topic, bootstrap_servers):

    query = (
        predictions.selectExpr("to_json(struct(*)) AS value")
        .writeStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("topic", topic)
        .outputMode("append")
        .option("checkpointLocation", f"{PROJECT_ROOT_PATH}/checkpoints")
        .start()
    )

    return query


if __name__ == "__main__":

    model = joblib.load(MODEL_PATH)

    spark = get_spark_session()

    assembler = VectorAssembler(
        inputCols=[
            "HighBP",
            "HighChol",
            "CholCheck",
            "BMI",
            "Smoker",
            "Stroke",
            "HeartDiseaseorAttack",
            "PhysActivity",
            "Fruits",
            "Veggies",
            "HvyAlcoholConsump",
            "AnyHealthcare",
            "NoDocbcCost",
            "GenHlth",
            "MentHlth",
            "PhysHlth",
            "DiffWalk",
            "Sex",
            "Age",
            "Education",
            "Income",
        ],
        outputCol="features",
    )

    schema = StructType(
        [
            StructField("HighBP", FloatType(), False),
            StructField("HighChol", FloatType(), False),
            StructField("CholCheck", FloatType(), False),
            StructField("BMI", FloatType(), False),
            StructField("Smoker", FloatType(), False),
            StructField("Stroke", FloatType(), False),
            StructField("HeartDiseaseorAttack", FloatType(), False),
            StructField("PhysActivity", FloatType(), False),
            StructField("Fruits", FloatType(), False),
            StructField("Veggies", FloatType(), False),
            StructField("HvyAlcoholConsump", FloatType(), False),
            StructField("AnyHealthcare", FloatType(), False),
            StructField("NoDocbcCost", FloatType(), False),
            StructField("GenHlth", FloatType(), False),
            StructField("MentHlth", FloatType(), False),
            StructField("PhysHlth", FloatType(), False),
            StructField("DiffWalk", FloatType(), False),
            StructField("Sex", FloatType(), False),
            StructField("Age", FloatType(), False),
            StructField("Education", FloatType(), False),
            StructField("Income", FloatType(), False),
        ]
    )

    data_stream = get_kafka_data_stream(spark, INPUT_TOPIC, schema, BOOTSTRAP_SERVERS)

    vectorized_rows = assembler.transform(data_stream)

    predict_udf = F.udf(predict, FloatType())

    predictions = vectorized_rows.withColumn(
        "prediction", predict_udf(F.col("features"))
    ).drop(F.col("features"))

    query = send_predicitons_to_kafka(predictions, OUTPUT_TOPIC, BOOTSTRAP_SERVERS)
    query.awaitTermination()
