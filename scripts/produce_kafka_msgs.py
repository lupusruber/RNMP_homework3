import json
import logging
import sys

import pandas as pd

from kafka import KafkaProducer


logger = logging.getLogger("kafka_producer_script")


BOOTSTRAP_SERVERS = "localhost:9094"
TOPIC = "health_data_topic"


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger.info(f"RUnning Kafka Bootstrap servers on {BOOTSTRAP_SERVERS}")

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS, security_protocol="PLAINTEXT"
    )

    df = pd.read_csv("data/online_data.csv", iterator=True, chunksize=1)

    for row_iter in df:

        index, row = next(row_iter.iterrows())
        record = row.to_dict()

        try:

            value = json.dumps(record)
            logger.info(f" Producted record: {value}")
            producer.send(topic=TOPIC, value=value.encode("utf-8"))

        except KeyboardInterrupt:
            logger.info("Exiting gracefully...")
            producer.close()
            sys.exit(0)
