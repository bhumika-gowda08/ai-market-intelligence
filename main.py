from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from passlib.context import CryptContext
from authlib.integrations.starlette_client import OAuth
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

from services.data_fetcher import fetch_indicator, fetch_fuel_data, fetch_stock_data
from database import engine, get_db
from models import User

# ===============================
# INITIAL SETUP
# ===============================

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="supersecretkey")

templates = Jinja2Templates(directory="templates")

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


User.metadata.create_all(bind=engine)

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
# AUTH ROUTES
# ===============================

@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
def login_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()

    if not user or not pwd_context.verify(password [:72], user.hashed_password):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid credentials"}
        )

    request.session["user"] = user.username
    return RedirectResponse("/dashboard", status_code=302)


@app.get("/register")
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
def register_user(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Username already exists"}
        )

    password = password[:72]   # truncate to bcrypt limit
    hashed_password = pwd_context.hash(password)


    new_user = User(
        username=username,
        email=email,
        hashed_password=hashed_password
    )

    db.add(new_user)
    db.commit()
    return RedirectResponse("/login", status_code=302)


@app.get("/login/google")
async def login_google(request: Request):
    redirect_uri = request.url_for('auth_google')
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/google")
async def auth_google(request: Request, db: Session = Depends(get_db)):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get('userinfo')

    if user_info is None:
        return RedirectResponse("/login")

    google_id = user_info['sub']
    email = user_info['email']

    user = db.query(User).filter(User.google_id == google_id).first()

    if not user:
        user = User(
            email=email,
            google_id=google_id
        )
        db.add(user)
        db.commit()

    request.session["user"] = email
    return RedirectResponse("/dashboard", status_code=302)

@app.get("/")
def home():
    return RedirectResponse("/login")

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


# ===============================
# PAGE ROUTES
# ===============================

@app.get("/dashboard")
def dashboard_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/forecast-page")
def forecast_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse("forecast.html", {"request": request})


@app.get("/comparison-page")
def comparison_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse("comparison.html", {"request": request})


@app.get("/ranking-page")
def ranking_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse("ranking.html", {"request": request})


# ===============================
# DASHBOARD
# ===============================

@app.get("/dashboard")
def dashboard(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse("dashboard.html", {"request": request})


# ===============================
# FORECAST LOGIC
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
        return {"error": "No data available"}

    df = df.dropna()
    X = df["year"].values.reshape(-1, 1)
    y = df["value"].values

    # Train models
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    lr_accuracy = r2_score(y, lr_model.predict(X))

    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X, y)
    rf_accuracy = r2_score(y, rf_model.predict(X))

    if rf_accuracy > lr_accuracy:
        model = rf_model
        accuracy = rf_accuracy
        selected_model = "Random Forest"
    else:
        model = lr_model
        accuracy = lr_accuracy
        selected_model = "Linear Regression"

    last_year = df["year"].max()
    future_years = np.array(range(last_year + 1, target_year + 1)).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    # CAGR
    first_value = y[0]
    last_value = y[-1]
    years = len(y)
    cagr = ((last_value / first_value) ** (1 / years) - 1) * 100

    # Trend
    if cagr > 10:
        trend = "High Growth"
    elif cagr > 3:
        trend = "Moderate Growth"
    else:
        trend = "Low / Declining"

    return {
        "market": market,
        "selected_model": selected_model,
        "model_accuracy_r2": round(float(accuracy), 4),
        "cagr_percent": round(float(cagr), 2),
        "trend_classification": trend,
        "historical_data": df.to_dict(orient="records"),
        "forecast": [
            {
                "year": int(future_years[i][0]),
                "predicted_value": float(future_predictions[i])
            }
            for i in range(len(future_years))
        ]
    }


@app.get("/compare")
def compare_markets(markets: str, target_year: int = 2035):

    market_list = markets.split(",")
    comparison_result = {}

    for market in market_list:
        result = forecast_market(market, target_year)
        comparison_result[market] = {
            "forecast": result.get("forecast", []),
            "cagr": result.get("cagr_percent", 0),
            "trend": result.get("trend_classification", "")
        }

    return comparison_result

@app.get("/ranking-data")
def ranking_data(target_year: int = 2035):

    markets = ["gdp", "renewable", "inflation", "fuel", "ev", "automobile"]
    ranking_result = []

    for market in markets:
        result = forecast_market(market, target_year)

        ranking_result.append({
            "market": market.upper(),
            "cagr": result["cagr_percent"],
            "trend": result["trend_classification"]
        })

    ranking_result = sorted(ranking_result, key=lambda x: x["cagr"], reverse=True)

    return ranking_result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)