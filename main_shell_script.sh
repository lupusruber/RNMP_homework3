mkdir checkpoints

chmod -R 777 ./data ./scripts ./checkpoints/

source data/get_data_script.sh

python model/best_model_script.py

docker-compose -f infrastructure/kafka-docker-compose.yml up -d
docker-compose -f infrastructure/spark-docker-compose.yml up -d

python scripts/produce_kafka_msgs.py

source scripts/start_stream.sh

python consume_kafka_msgs.py