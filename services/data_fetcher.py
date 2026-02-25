import requests
import pandas as pd
from fredapi import Fred
import os
import yfinance as yf


WORLD_BANK_BASE_COUNTRY = "https://api.worldbank.org/v2/country/IND/indicator"
WORLD_BANK_BASE_GLOBAL = "https://api.worldbank.org/v2/indicator"


INDICATORS = {
    "gdp": "NY.GDP.MKTP.CD",
    "oil_price": "FP.CPI.TOTL.ZG",
    "renewable_energy": "EG.FEC.RNEW.ZS"
}

def fetch_indicator(indicator_code):
    # Fuel price is global indicator
    if indicator_code == "POILBREUSDM":
        url = f"{WORLD_BANK_BASE_GLOBAL}/{indicator_code}?format=json&per_page=100"
    else:
        url = f"{WORLD_BANK_BASE_COUNTRY}/{indicator_code}?format=json&per_page=100"

    response = requests.get(url)
    data = response.json()

    if len(data) < 2:
        return None

    records = data[1]

    clean_data = []
    for entry in records:
        if entry["value"] is not None:
            clean_data.append({
                "year": int(entry["date"]),
                "value": float(entry["value"])
            })

    if not clean_data:
        return None

    df = pd.DataFrame(clean_data)
    df = df.sort_values("year")
    return df

def fetch_fuel_data():
    api_key = "07009760ff2929fdf0796bcb3a67bca6"
    
    if not api_key:
        return None

    fred = Fred(api_key=api_key)

    data = fred.get_series('DCOILBRENTEU')  # Brent Crude Oil

    df = data.reset_index()
    df.columns = ['date', 'value']

    df['date'] = pd.to_datetime(df['date'])

    df['year'] = df['date'].dt.year
    df = df.groupby('year').mean().reset_index()

    df = df[['year', 'value']]
    df = df.sort_values('year')

    return df

def fetch_stock_data(ticker):
    data = yf.download(ticker, period="15y", interval="1mo")

    if data.empty:
        return None

    df = data.reset_index()
    df['year'] = df['Date'].dt.year

    df = df.groupby('year')['Close'].mean().reset_index()
    df.columns = ['year', 'value']

    df = df.sort_values('year')

    return df
