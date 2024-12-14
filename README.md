# Рударење на масивни податоци: Домашна работа 3

This project is designed to process health data using Spark Streaming, Kafka, and machine learning. 
The main pipeline reads health indicators from Kafka, makes predictions using a trained model, and sends the results back to Kafka.

## Project Structure

```
.
├── checkpoints                 # Directory for storing Spark checkpoint data
├── data                        # Data-related files and scripts
│   ├── get_data_script.sh      # Script to fetch data
│   └── split_data.py           # Script to split data into train/test sets
├── infrastructure              # Infrastructure configuration files
│   ├── kafka-docker-compose.yml  # Docker Compose for Kafka
│   └── spark-docker-compose.yml  # Docker Compose for Spark
├── model                       # Model-related files and scripts
│   ├── best_model_script.py    # Script to find the best model
├── poetry.lock                 # Poetry lockfile for dependency management
├── pyproject.toml              # Poetry configuration file
├── README.md                   # Main project documentation
└── scripts                     # Main scripts
    ├── consume_kafka_msgs.py   # Script to consume messages from Kafka
    ├── make_predictions_spark_script.py  # Main Spark script for predictions
    ├── produce_kafka_msgs.py   # Script to produce messages to Kafka
    └── start_stream.sh         # Shell script to start the streaming pipeline
```

## Usage Instructions

### 1. Setting up the Environment
- Ensure Docker and Docker Compose are installed on your system.
- Use the provided Docker Compose files to set up Kafka and Spark:
  ```bash
  docker-compose -f infrastructure/kafka-docker-compose.yml up -d
  docker-compose -f infrastructure/spark-docker-compose.yml up -d
  ```

### 2. Preparing the Data
- Use `data/get_data_script.sh` to fetch the raw data.
- Split the data into training and testing sets using `data/split_data.py`.

### 3. Training the Model
- Run `model/best_model_script.py` to train models and save the best one as `model/model.pkl`.

### 4. Streaming Pipeline
- Produce messages to Kafka using `scripts/produce_kafka_msgs.py`.
- Run the Spark streaming script `scripts/make_predictions_spark_script.py` to process the data and produce predictions.
- Consume the prediction results from Kafka using `scripts/consume_kafka_msgs.py`.


## Dependencies
- Python 3.10 or higher
- Poetry for dependency management
- Apache Spark
- Apache Kafka

## Commands
- Start the pipeline:
  ```bash
  bash scripts/start_stream.sh
  ```

- View Kafka messages:
  ```bash
  python scripts/consume_kafka_msgs.py
  ```

## Notes
- Make sure all containers are running on the same Docker network for communication.
- Update Kafka broker addresses in the scripts as necessary.

