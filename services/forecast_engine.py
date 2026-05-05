from services.data_fetcher import fetch_indicator, fetch_fuel_data, fetch_stock_data
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


def forecast_market(market, target_year):

    # 🔗 Map markets to real data sources
    if market == "gdp":
        df = fetch_indicator("NY.GDP.MKTP.CD")

    elif market == "inflation":
        df = fetch_indicator("FP.CPI.TOTL.ZG")

    elif market == "renewable":
        df = fetch_indicator("EG.FEC.RNEW.ZS")

    elif market == "fuel":
        df = fetch_fuel_data()

    elif market == "ev":
        df = fetch_stock_data("TSLA")  # proxy for EV market

    elif market == "automobile":
        df = fetch_stock_data("TM")  # Toyota

    else:
        return []

    # 🚨 safety check
    if df is None or df.empty:
        return []

    df = df.dropna()

    # ML training
    X = df["year"].values.reshape(-1, 1)
    y = df["value"].values

    model = LinearRegression()
    model.fit(X, y)

    last_year = df["year"].max()

    # ✅ FIX: handle past years (NO CRASH)
    if target_year <= last_year:
        filtered = df[df["year"] <= target_year]

        return [
            {
                "year": int(row["year"]),
                "value": float(row["value"])
            }
            for _, row in filtered.iterrows()
        ]

    # 🔮 future prediction
    future_years = np.array(
        [year for year in range(last_year + 1, target_year + 1)]
    ).reshape(-1, 1)

    predictions = model.predict(future_years)

    # combine past + future
    result = []

    # historical
    for _, row in df.iterrows():
        result.append({
            "year": int(row["year"]),
            "value": float(row["value"])
        })

    # forecast
    for i in range(len(future_years)):
        result.append({
            "year": int(future_years[i][0]),
            "value": float(predictions[i])
        })

    return result