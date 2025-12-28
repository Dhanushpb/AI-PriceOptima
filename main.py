from fastapi import FastAPI
import joblib
import numpy as np
from datetime import datetime

app = FastAPI(title="AI PriceOptima – Automated Features API")

# Load trained XGBoost model
model = joblib.load("pricing_model.joblib")

@app.get("/")
def home():
    return {"message": "FastAPI backend is running"}

@app.post("/predict")
def predict_price(
    cost: float,
    demand: float,
    inventory: int,
    competitor_price: float
):
    # -------------------------------
    # AUTO-GENERATED FEATURES (14)
    # -------------------------------
    discount = 0.0
    seasonality = 1.0
    price_elasticity = -1.0

    rolling_avg_price = competitor_price
    lag_price = competitor_price

    now = datetime.now()
    day_of_week = now.weekday()
    week_of_year = now.isocalendar()[1]
    month = now.month
    is_weekend = 1 if day_of_week >= 5 else 0

    inventory_ratio = inventory / max(demand, 1)
    demand_growth = 0.0
    price_change = 0.0
    promo_flag = 0
    holiday_flag = 0

    # -------------------------------
    # FINAL 18-FEATURE INPUT VECTOR
    # (ORDER MUST MATCH TRAINING)
    # -------------------------------
    input_data = np.array([[
        cost, demand, inventory, competitor_price,
        discount, seasonality, price_elasticity,
        rolling_avg_price, lag_price,
        day_of_week, week_of_year, is_weekend,
        month, inventory_ratio, demand_growth,
        price_change, promo_flag, holiday_flag
    ]])

    prediction = model.predict(input_data)[0]

    return {
        "recommended_price": round(float(prediction), 2)
    }
#Run using --> uvicorn main:app --reload