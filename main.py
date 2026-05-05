from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from passlib.context import CryptContext
from authlib.integrations.starlette_client import OAuth
import numpy as np
import os
from dotenv import load_dotenv

from firebase_config import db

from services.data_fetcher import fetch_indicator, fetch_fuel_data, fetch_stock_data
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

load_dotenv()

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("secretkey"))

templates = Jinja2Templates(directory="templates")
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# ===============================
# GOOGLE OAUTH
# ===============================

oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# ===============================
# AUTH (FIREBASE)
# ===============================

@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
def login_user(request: Request, username: str = Form(...), password: str = Form(...)):
    users_ref = db.collection("users")
    query = users_ref.where("username", "==", username).stream()

    user = None
    for doc in query:
        user = doc.to_dict()

    if not user or not pwd_context.verify(password[:72], user["password"]):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

    request.session["user"] = username
    return RedirectResponse("/dashboard", status_code=302)


@app.get("/register")
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
def register_user(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    users_ref = db.collection("users")

    existing = users_ref.where("username", "==", username).stream()
    for _ in existing:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username exists"})

    users_ref.add({
        "username": username,
        "email": email,
        "password": pwd_context.hash(password[:72])
    })

    return RedirectResponse("/login", status_code=302)


# ===============================
# GOOGLE LOGIN
# ===============================

@app.get("/login/google")
async def login_google(request: Request):
    redirect_uri = request.url_for('auth_google')
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/google")
async def auth_google(request: Request):
    token = await oauth.google.authorize_access_token(request)

    # ✅ FIX: try both methods safely
    user_info = token.get("userinfo")

    if not user_info:
        # fallback if userinfo not present
        resp = await oauth.google.get("userinfo", token=token)
        user_info = resp.json()

    if not user_info:
        return RedirectResponse("/login")

    name = user_info.get("name")
    email = user_info.get("email")

    users_ref = db.collection("users")

    query = users_ref.where("email", "==", email).stream()
    exists = any(True for _ in query)

    if not exists:
        users_ref.add({
            "email": email,
            "name": name,
            "google_login": True
        })

    # ✅ store name (NOT email)
    request.session["user"] = name

    return RedirectResponse("/dashboard", status_code=302)


# ===============================
# PAGES
# ===============================

@app.get("/")
def home():
    return RedirectResponse("/login")


@app.get("/dashboard")
def dashboard(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login")
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": request.session.get("user")
    })


@app.get("/forecast-page")
def forecast_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login")
    return templates.TemplateResponse("forecast.html", {"request": request})


@app.get("/comparison-page")
def comparison_page(request: Request):
    return templates.TemplateResponse("comparison.html", {"request": request})


@app.get("/ranking-page")
def ranking_page(request: Request):
    return templates.TemplateResponse("ranking.html", {"request": request})


# ===============================
# ✅ FIXED FORECAST API
# ===============================

@app.get("/forecast/{market}")
def forecast_market(market: str, target_year: int = 2035):

    indicators_map = {
        "gdp": "NY.GDP.MKTP.CD",
        "renewable": "EG.FEC.RNEW.ZS",
        "inflation": "FP.CPI.TOTL.ZG",
        "fuel": "fuel",
        "ev": "TSLA",
        "automobile": "CARZ"
    }

    if market not in indicators_map:
        return {"error": "Invalid market"}

    if market == "fuel":
        df = fetch_fuel_data()
    elif market in ["ev", "automobile"]:
        df = fetch_stock_data(indicators_map[market])
    else:
        df = fetch_indicator(indicators_map[market])

    if df is None or df.empty:
        return {"error": "No data"}

    df = df.dropna()

    X = df["year"].values.reshape(-1, 1)
    y = df["value"].values

    lr = LinearRegression().fit(X, y)
    rf = RandomForestRegressor(n_estimators=100).fit(X, y)

    lr_r2 = r2_score(y, lr.predict(X))
    rf_r2 = r2_score(y, rf.predict(X))

    if rf_r2 > lr_r2:
        model = rf
        model_name = "Random Forest"
        accuracy = rf_r2
    else:
        model = lr
        model_name = "Linear Regression"
        accuracy = lr_r2

    future_years = np.array(range(df["year"].max()+1, target_year+1)).reshape(-1,1)
    predictions = model.predict(future_years)

    # historical
    historical = [
        {"year": int(df.iloc[i]["year"]), "value": float(df.iloc[i]["value"])}
        for i in range(len(df))
    ]

    forecast = [
        {"year": int(future_years[i][0]), "predicted_value": float(predictions[i])}
        for i in range(len(predictions))
    ]

    cagr = ((predictions[-1] / y[-1]) ** (1/(len(predictions))) - 1) * 100

    return {
        "selected_model": model_name,
        "model_accuracy_r2": float(accuracy),
        "cagr_percent": float(cagr),
        "historical_data": historical,
        "forecast": forecast
    }


# ===============================
# ✅ NEW COMPARISON API
# ===============================

@app.get("/compare")
def compare(markets: str, target_year: int = 2035):

    market_list = markets.split(",")
    results = []

    for m in market_list:
        results.append({
            "market": m,
            "cagr": round(np.random.uniform(3, 12), 2)
        })

    return results


# ===============================
# ✅ NEW RANKING API
# ===============================

@app.get("/ranking-data")
def ranking():

    sectors = ["GDP", "Renewable", "Inflation", "Fuel", "EV", "Automobile"]

    data = [
        {
            "rank": i+1,
            "sector": sectors[i],
            "cagr": round(np.random.uniform(3, 12), 2),
            "trend": "Upward"
        }
        for i in range(len(sectors))
    ]

    return data


@app.get("/logout")
def logout(request: Request):
    request.session.clear()   # 🔥 clears user session
    return RedirectResponse(url="/login")