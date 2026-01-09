from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import os

app = Flask(__name__)
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.pkl")
DATA_PATH = os.environ.get("DATA_PATH", "data/flights.csv")

print("Loading model...")
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded.")
else:
    print("Model not found at", MODEL_PATH)

# Load dataset for showing sample offers
if os.path.exists(DATA_PATH):
    df_data = pd.read_csv(DATA_PATH)
else:
    df_data = pd.DataFrame()

@app.route("/", methods=["GET"])
def index():
    # Basic form fields
    airports = sorted(set(df_data["origin"].tolist() + df_data["destination"].tolist())) if not df_data.empty else ["JFK","LAX","SFO","ORD"]
    airlines = sorted(df_data["airline"].unique().tolist()) if not df_data.empty else ["Delta","United"]
    return render_template("index.html", airports=airports, airlines=airlines)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model is not available. Train the model first (model/train.py).", 500
    origin = request.form["origin"]
    destination = request.form["destination"]
    days_to_departure = int(request.form["days_to_departure"])
    airline = request.form.get("airline", "")
    duration = int(request.form.get("duration", 180))
    stops = int(request.form.get("stops", 0))
    # Build input frame
    X = pd.DataFrame([{
        "origin": origin,
        "destination": destination,
        "days_to_departure": days_to_departure,
        "airline": airline,
        "duration": duration,
        "stops": stops,
    }])
    pred_price = model.predict(X)[0]
    # Find cheap offers from dataset that match origin/destination and dates closish
    recommendations = []
    if not df_data.empty:
        cond = (df_data.origin == origin) & (df_data.destination == destination)
        subset = df_data[cond].copy()
        subset["score"] = (subset["price"] - pred_price).abs()
        subset = subset.sort_values("price").head(10)
        recommendations = subset.to_dict(orient="records")
    return render_template("result.html",
                           origin=origin, destination=destination,
                           days_to_departure=days_to_departure,
                           predicted_price=round(pred_price,2),
                           recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)