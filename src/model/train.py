import os
import sys
import pandas as pd
import xgboost as xgb
import mlflow

sys.path.append(os.path.join(os.getcwd(), "src"))

from text_loader.loader import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set experiment name
mlflow.set_experiment("tweet_classifier")

def train(df: pd.DataFrame):
    with mlflow.start_run():
        data_loader = DataLoader()

        df.Tweet = df.Tweet.apply(data_loader.clean_text)
        X = data_loader.vectorize_text(df.Tweet.values) # type: ignore

        df.Party = df.Party.apply(data_loader.clean_text)
        y = data_loader.label_encoder(df.Party.values)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            "objective": "binary:logistic",   # binary classification
            "eval_metric": "logloss",         # good for probabilistic classification

            # Tree parameters
            "max_depth": 6,                   # controls tree complexity (3–7 is typical for text data)
            "min_child_weight": 1,            # minimum sum of instance weight (hessian) needed in a child
            "gamma": 0,                       # min loss reduction to make a split
            "subsample": 0.8,                 # % of training samples used per tree
            "colsample_bytree": 0.8,          # % of features used per tree

            # Regularization
            "lambda": 1,                      # L2 regularization term
            "alpha": 0,                       # L1 regularization term

            # Learning rate
            "eta": 0.1,                       # step size shrinkage (0.05–0.3 works well)
            "n_estimators": 300,              # number of boosting rounds
        }

        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.xgboost.log_model(model, "model", signature=signature)

        y_pred = model.predict(X_test)

        mlflow.log_metric("accuracy", float(accuracy_score(y_test, y_pred)))
        mlflow.log_metric("precision", float(precision_score(y_test, y_pred, average="weighted", zero_division=0)))
        mlflow.log_metric("recall", float(recall_score(y_test, y_pred, average="weighted", zero_division=0)))
        mlflow.log_metric("f1_score", float(f1_score(y_test, y_pred, average="weighted", zero_division=0)))


def main():
    df = pd.read_csv("data/Tweets.csv")
    train(df)

if __name__ == "__main__":
    main()