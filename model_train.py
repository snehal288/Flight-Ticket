"""Train a simple model to predict flight price from the synthetic dataset."""
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(path):
    df = pd.read_csv(path)
    return df

def build_pipeline():
    numeric_features = ["days_to_departure", "duration", "stops"]
    categorical_features = ["origin", "destination", "airline"]
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
    )
    return model

def train(args):
    df = load_data(args.data)
    X = df[["origin", "destination", "days_to_departure", "airline", "duration", "stops"]]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", mean_squared_error(y_test, preds, squared=False))
    joblib.dump(pipeline, args.out)
    print(f"Saved model to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/flights.csv")
    parser.add_argument("--out", default="model/model.pkl")
    args = parser.parse_args()
    train(args)