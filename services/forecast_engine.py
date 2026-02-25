import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def train_and_forecast(df, years_ahead=5):
    df = df.dropna()
    X = df["year"].values.reshape(-1, 1)
    y = df["value"].values

    model = LinearRegression()
    model.fit(X, y)

    last_year = df["year"].max()

    future_years = np.array(
        [last_year + i for i in range(1, years_ahead + 1)]
    ).reshape(-1, 1)

    predictions = model.predict(future_years)

    result = []
    for i in range(len(future_years)):
        result.append({
            "year": int(future_years[i][0]),
            "predicted_value": float(predictions[i])
        })

    return result
