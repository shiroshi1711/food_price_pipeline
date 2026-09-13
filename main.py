from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
 
app = FastAPI(title="Food Price Forecast API")
 

model = joblib.load("price_forecast_model_monthly_2.pkl")
feature_columns = joblib.load("feature_columns.pkl")
 
DATABASE_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
 
 
class PredictRequest(BaseModel):
    variant_name: str
    region: str
 
 
@app.get("/")
def root():
    return {"message": "Food Price Forecast API is running"}
 
 
@app.post("/predict")
def predict(request: PredictRequest):
    query = """
        SELECT date, price FROM food_prices
        WHERE variant_name = %(variant_name)s AND region = %(region)s
        ORDER BY date DESC LIMIT 120
    """
    recent = pd.read_sql(query, engine, params={
        "variant_name": request.variant_name,
        "region": request.region
    })
 
    if recent.empty:
        raise HTTPException(status_code=404, detail="No data found for this item/region")
 
    recent['date'] = pd.to_datetime(recent['date'])
    recent = recent.sort_values('date')
    recent['year_month'] = recent['date'].dt.to_period('M')
 
    monthly_recent = recent.groupby('year_month')['price'].mean().reset_index()
    monthly_recent['price_log'] = np.log1p(monthly_recent['price'])
    monthly_recent = monthly_recent.sort_values('year_month')
 
    price_lag1 = monthly_recent['price_log'].iloc[-1]
    price_lag2 = monthly_recent['price_log'].iloc[-2] if len(monthly_recent) >= 2 else price_lag1
    
    price_lag12 = monthly_recent['price_log'].iloc[-12] if len(monthly_recent) >= 12 else price_lag1
    
    price_roll3 = monthly_recent['price_log'].tail(3).mean()
    price_roll12 = monthly_recent['price_log'].tail(12).mean()
    
    momentum = price_lag1 - price_roll3


    row = {col: 0 for col in feature_columns}
    row['price_lag1'] = price_lag1
    row['price_lag2'] = price_lag2
    row['price_lag12'] = price_lag12
    row['price_roll3'] = price_roll3
    row['price_roll12'] = price_roll12
    row['momentum'] = momentum
 
    item_col = f"item_{request.variant_name}"
    region_col = f"region_{request.region}"
    if item_col in row:
        row[item_col] = 1
    if region_col in row:
        row[region_col] = 1
 
    X = pd.DataFrame([row])[feature_columns]
    predicted_change = model.predict(X)[0]
    predicted_price_log = price_lag1 + predicted_change
    predicted_price = float(np.expm1(predicted_price_log))
 
    return {
        "variant_name": request.variant_name,
        "region": request.region,
        "current_month_price": round(float(np.expm1(price_lag1)), 2),
        "predicted_next_month_price": round(predicted_price, 2)
    }
 