import joblib
import logging

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import pandas as pd
import numpy as np

from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = f"{PROJECT_ROOT}/data/offline_data.csv"
MODEL_PATH = f"{PROJECT_ROOT}/model/best_model.pkl"


logger = logging.getLogger("best_model_script")


models = {
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    # "xgboost": XGBClassifier(),
}

param_grids = {
    "logistic_regression": {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs", "saga"],
        "max_iter": [500],
        "class_weight": ["balanced"],
    },
    "random_forest": {
        "class_weight": ["balanced"],
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    },
    # "xgboost": {
    #     "scale_pos_weight": [len(y_train[y_train == 0]) / len(y_train[y_train == 1])],
    #     "learning_rate": [0.01, 0.1],
    #     "n_estimators": [100, 200],
    #     "max_depth": [3, 6],
    # },
}


def read_and_split_data(
    data_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    df = pd.read_csv(data_path)

    X = df.drop(columns="DiabetesBinary").values
    y = df["DiabetesBinary"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    return X_train, X_test, y_train, y_test


def get_and_save_best_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_path: str,
) -> None:

    best_model = None
    best_f1_score = 0
    best_model_name = ""

    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")

        grid_search = GridSearchCV(model, param_grids[model_name], scoring="f1", cv=2)
        grid_search.fit(X_train, y_train)
        f1 = f1_score(y_test, grid_search.predict(X_test))

        logger.info(f"Best f1 score for {model_name} is {f1}")

        if f1 > best_f1_score:
            best_f1_score = f1
            best_model = grid_search.best_estimator_
            best_model_name = model_name

    joblib.dump(best_model, model_path)

    logger.info(f"The best model is {best_model_name}")
    logger.info(f"Best model F1 score: {best_f1_score}")
    logger.info(f"Model saved at {model_path}")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    X_train, X_test, y_train, y_test = read_and_split_data(DATA_PATH)
    get_and_save_best_model(X_train, X_test, y_train, y_test, MODEL_PATH)

    logger.info("Script finished successfully")
