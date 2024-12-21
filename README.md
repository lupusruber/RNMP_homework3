# Рударење на масивни податоци: Домашна работа 3

This project is designed to process health data using Spark Streaming, Kafka, and machine learning. 
The main pipeline reads health indicators from Kafka, makes predictions using a trained model, and sends the results back to Kafka.

## Project Structure

```
.
├── checkpoints                           # Directory for storing Spark checkpoint data
├── data                                  # Data-related files and scripts
│   ├── get_data_script.sh                # Script to fetch data
│   └── split_data.py                     # Script to split data into train/test sets
├── infrastructure                        # Infrastructure configuration files
│   ├── kafka-docker-compose.yml          # Docker Compose for Kafka
│   └── spark-docker-compose.yml          # Docker Compose for Spark
├── model                                 # Model-related files and scripts
│   ├── best_model_script.py              # Script to find the best model
├── poetry.lock                           # Poetry lockfile for dependency management
├── pyproject.toml                        # Poetry configuration file
├── README.md                             # Main project documentation
├── main_shell_script.sh                  # Main project script, run this to build everything needed
└── scripts                               # Main scripts
    ├── consume_kafka_msgs.py             # Script to consume messages from Kafka
    ├── make_predictions_spark_script.py  # Main Spark script for predictions
    ├── produce_kafka_msgs.py             # Script to produce messages to Kafka
    └── start_stream.sh                   # Shell script to start the streaming pipeline
```

## Usage Instructions

### 1. Setting up the Environment
- Use the provided Docker Compose files to set up Kafka and Spark:
  ```bash
  docker-compose -f infrastructure/kafka-docker-compose.yml up -d
  docker-compose -f infrastructure/spark-docker-compose.yml up -d
  ```

### 2. Preparing the Data
- Use `data/get_data_script.sh` to fetch the raw data.
- Split the data into training and testing data for the model (offline data) and data for making the predictions (online data) using `data/split_data.py`.

### 3. Training the Model
- Run `model/best_model_script.py` to train models and save the best one as `model/model.pkl`.
- This script trains 3 models (Logistic Regression, XGBoost and Random Forest) using 3 fold CV for finding the best model.


### 4. Making the predictions using Spark Structured Streaming
- Use `scripts/make_predictions_spark_script.py` script as a spark streaming job
- This script creates a Spark Session, reads the steaming data, adds a prediction column to each row and sends the resultant rows to another Kafka topic.


### 5. The whole streaming pipeline
- Produce messages to Kafka using `scripts/produce_kafka_msgs.py`.
- Run the Spark streaming script `scripts/make_predictions_spark_script.py` to process the data and produce predictions.
- Consume the prediction results from Kafka using `scripts/consume_kafka_msgs.py`.


## Dependencies
- Python 3.10 or higher
- Poetry for dependency management
- Docker and Docker Compose
  - Apache Spark
  - Apache Kafka

## Commands
- Generate the offline.csv and online.csv files:
```bash
  source data/get_data_script.sh
```

- Find the best model and save it as a pkl file:
```bash
  python model/best_model_script.py
```

- Produce the needed Kafka messages for making predictions:
```bash
  python scripts/consume_kafka_msgs.py
  ```

- Start the pipeline inside the spark container:
```bash
  source scripts/start_stream.sh
  ```

- View Kafka messages:
```bash
  python scripts/consume_kafka_msgs.py
  ```

## Notes
- There is one script that runs everything needed to get the data, find the best model and make the streaming pipelines, as well as start the docker containers.
```bash
  source main_shell_script.sh
  ```
- Install the needed python packages using poetry
- Make sure all containers are running on the same Docker network for communication.
- Update Kafka broker addresses in the scripts as necessary.