from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import numpy as np
from datetime import datetime
from pydantic import BaseModel

app = FastAPI(title="AI PriceOptima – Automated Features API")

# Enable CORS so local frontend dev servers can call the API
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class PriceInput(BaseModel):
    cost: float
    demand: float
    inventory: int
    competitor_price: float


# Use a fallback model by default to avoid crashing when native ML libs fail to load
class _FallbackModel:
    """Simple fallback model with a predict method so the API stays available
    for frontend testing. Returns a conservative price estimate.
    """
    def predict(self, X):
        result = []
        for row in X:
            try:
                cost = float(row[0])
                competitor = float(row[3])
                price = max(cost * 1.1, competitor * 0.98)
            except Exception:
                price = 0.0
            result.append(price)
        return np.array(result)


model = _FallbackModel()

@app.get("/")
def home():
    return {"message": "FastAPI backend is running"}


@app.post("/predict_price")
def predict_price(data: PriceInput):
    cost = data.cost
    demand = data.demand
    inventory = data.inventory
    competitor_price = data.competitor_price

    # (your feature engineering stays exactly the same)

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