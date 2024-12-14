#!/bin/bash

cd data

curl -L -o diabetes-health-indicators-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/alexteboul/diabetes-health-indicators-dataset

unzip -o diabetes-health-indicators-dataset.zip

rm -f diabetes-health-indicators-dataset.zip

rm -f diabetes_012_health_indicators_BRFSS2015.csv diabetes_binary_5050split_health_indicators_BRFSS2015.csv

echo "Dataset has been downloaded and extracted in $(pwd)."

python split_data.py

echo "Data split into offline and online data"

cd ..