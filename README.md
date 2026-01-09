# Cheap Flights Mini Project

This mini-project demonstrates a simple pipeline that uses machine learning to predict flight fares and a Flask website that lets users search for cheap tickets.

Structure:
- data/generate_sample_data.py — generate a synthetic dataset of flights
- model/train.py — train a model and save `model/model.pkl`
- app.py — Flask web app (serves forms and returns predicted cheap fares)
- templates/ — HTML templates for the front-end
- static/style.css — basic styling
- requirements.txt — dependencies

Quickstart:
1. Create a virtual environment and install requirements:
   ```
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Generate sample data:
   ```
   python data/generate_sample_data.py --out data/flights.csv --n 5000
   ```

3. Train model:
   ```
   python model/train.py --data data/flights.csv --out model/model.pkl
   ```

4. Run the app:
   ```
   python app.py
   ```

5. Open http://127.0.0.1:5000 in your browser.

Notes:
- This project uses synthetic data. Replace `data/flights.csv` with a real dataset or integrate a live flights API for production use.
- For recommendations, the app predicts price and also shows the cheapest matching entries from the dataset to simulate real offers.