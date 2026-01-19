from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="House Price Prediction API")

# load saved artifacts
model = joblib.load("final_house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
num_cols = joblib.load("numerical_cols.pkl")
cat_cols = joblib.load("categorical_cols.pkl")


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict(data: dict):
    # create DataFrame from input
    df = pd.DataFrame([data])

    # 🔑 IMPORTANT FIX: ensure all columns exist
    df = df.reindex(columns=list(num_cols) + list(cat_cols), fill_value=np.nan)


    # fill missing values
    df[cat_cols] = df[cat_cols].fillna("None")
    df[num_cols] = df[num_cols].fillna(0)

    # preprocessing
    X_num = scaler.transform(df[num_cols])
    X_cat = encoder.transform(df[cat_cols])

    X_final = np.hstack([X_num, X_cat])

    pred_log = model.predict(X_final)
    price = np.expm1(pred_log)[0]

    return {"predicted_price": round(price, 2)}

