spark-submit --packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.5.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 scripts/make_predictions_spark_script.py

chmod -R 777 ./data ./scripts ./checkpoints/