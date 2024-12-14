import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent
df = pd.read_csv(f"{DATA_PATH}/data/offline_data.csv")

X = df.drop(columns="DiabetesBinary")
y = df["DiabetesBinary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

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

best_model = None
best_f1_score = 0
best_model_name = ""

for model_name, model in models.items():
    print(f"Evaluating model: {model_name}")

    grid_search = GridSearchCV(model, param_grids[model_name], scoring="f1", cv=2)
    grid_search.fit(X_train, y_train)
    f1 = f1_score(y_test, grid_search.predict(X_test))

    print(f"Best f1 score for {model_name} is {f1}")

    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = grid_search.best_estimator_
        best_model_name = model_name

model_path = f"model/{best_model_name}.pkl"
joblib.dump(best_model, model_path)
print(f"The best model is {best_model_name} with F1 score: {best_f1_score}")
print(f"Model saved at {model_path}")
