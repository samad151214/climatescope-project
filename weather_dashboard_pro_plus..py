# weather_dashboard_pro_plus.py
"""
Weather Intelligence Dashboard — Extended Advanced Version
Adds:
- More filters (pressure, precipitation, AQI, clouds, month/hour)
- Aggregation control (mean/median/max)
- Multi-metric selector for visualization
- Time-based heatmap and pairplot
- Summary statistics + Reset filters + Theme selector
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from datetime import datetime
import math

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# --- CONFIG ---
st.set_page_config(page_title="🌍 Weather Intelligence Pro+", layout="wide")
st.title("🌦 Weather Intelligence — Pro+ Dashboard")

st.markdown("Enhanced version with deeper filters, analysis, and flexible visualization.")

# -------------------------------
# Data Load
# -------------------------------
st.sidebar.header("1️⃣ Data Setup")
uploaded = st.sidebar.file_uploader("Upload Weather CSV", type=["csv"])

@st.cache_data
def load_csv(path="/mnt/data/GlobalWeatherRepository_cleaned.csv"):
    return pd.read_csv(path, low_memory=False)

if uploaded:
    df = pd.read_csv(uploaded, low_memory=False)
else:
    try:
        df = load_csv()
    except:
        st.error("No dataset available. Please upload one.")
        st.stop()

df.columns = [c.strip() for c in df.columns]
cols_lower = {c.lower(): c for c in df.columns}

def find_col(*names):
    for n in names:
        if n.lower() in cols_lower:
            return cols_lower[n.lower()]
    return None

date_col = find_col("date", "datetime", "timestamp")
temp_col = find_col("temperature", "temp")
humidity_col = find_col("humidity", "hum")
wind_col = find_col("windspeed", "wind")
pressure_col = find_col("pressure", "atm_pressure")
precip_col = find_col("precipitation", "rain", "snow")
aqi_col = find_col("airquality", "aqi")
cloud_col = find_col("cloud", "cloudiness")
country_col = find_col("country")
city_col = find_col("city", "location")
weather_col = find_col("weather", "conditions")

# Datetime conversion
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

for col in [temp_col, humidity_col, wind_col, pressure_col, precip_col, aqi_col, cloud_col]:
    if col:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("2️⃣ Advanced Filters")

# Reset Filters Button
if st.sidebar.button("🔄 Reset All Filters"):
    st.experimental_rerun()

# Country / City
selected_countries = st.sidebar.multiselect("🌍 Country", sorted(df[country_col].dropna().unique()) if country_col else [])
if selected_countries and city_col:
    cities = df[df[country_col].isin(selected_countries)][city_col].dropna().unique()
else:
    cities = df[city_col].dropna().unique() if city_col else []
selected_cities = st.sidebar.multiselect("🏙 City", sorted(cities))

# Weather
selected_weather = st.sidebar.multiselect(
    "☁ Weather Type", sorted(df[weather_col].dropna().unique()) if weather_col else []
)

# Ranges for numeric filters
def range_slider(label, col):
    if col and col in df:
        return st.sidebar.slider(label,
            float(df[col].min()), float(df[col].max()),
            (float(df[col].min()), float(df[col].max()))
        )
    return None

temp_range = range_slider("🌡 Temperature (°C)", temp_col)
hum_range = range_slider("💧 Humidity (%)", humidity_col)
wind_range = range_slider("🌬 Wind Speed", wind_col)
press_range = range_slider("📈 Pressure (hPa)", pressure_col)
precip_range = range_slider("🌧 Precipitation (mm)", precip_col)
aqi_range = range_slider("🫧 Air Quality Index (AQI)", aqi_col)
cloud_range = range_slider("☁ Cloud Cover (%)", cloud_col)

# Date filters
if date_col:
    min_date, max_date = df[date_col].min(), df[date_col].max()
    selected_dates = st.sidebar.date_input("📅 Date Range", [min_date, max_date])
    # Extract month/hour
    df["month"] = df[date_col].dt.month
    df["hour"] = df[date_col].dt.hour
    selected_months = st.sidebar.multiselect("🗓 Month", sorted(df["month"].unique()))
    selected_hours = st.sidebar.multiselect("⏰ Hour", sorted(df["hour"].unique()))
else:
    selected_dates = None
    selected_months = []
    selected_hours = []

# Aggregation
agg_mode = st.sidebar.radio("📊 Aggregation", ["Mean", "Median", "Max"], horizontal=True)

# Theme
theme = st.sidebar.selectbox("🎨 Theme", ["plotly", "seaborn", "ggplot2", "simple_white"])

# -------------------------------
# Apply Filters
# -------------------------------
dff = df.copy()
if country_col and selected_countries:
    dff = dff[dff[country_col].isin(selected_countries)]
if city_col and selected_cities:
    dff = dff[dff[city_col].isin(selected_cities)]
if weather_col and selected_weather:
    dff = dff[dff[weather_col].isin(selected_weather)]
for (col, rng) in [
    (temp_col, temp_range),
    (humidity_col, hum_range),
    (wind_col, wind_range),
    (pressure_col, press_range),
    (precip_col, precip_range),
    (aqi_col, aqi_range),
    (cloud_col, cloud_range)
]:
    if col and rng:
        dff = dff[(dff[col] >= rng[0]) & (dff[col] <= rng[1])]
if date_col and selected_dates and len(selected_dates) == 2:
    start, end = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
    dff = dff[(dff[date_col] >= start) & (dff[date_col] <= end)]
if selected_months:
    dff = dff[dff["month"].isin(selected_months)]
if selected_hours:
    dff = dff[dff["hour"].isin(selected_hours)]

st.write(f"### Showing {len(dff):,} records after all filters")

# -------------------------------
# Metric Selection
# -------------------------------
num_cols = dff.select_dtypes(include=[np.number]).columns.tolist()
metrics = st.multiselect("📈 Select Metrics for Analysis", num_cols, default=[temp_col] if temp_col else [])
if not metrics:
    st.warning("Select at least one metric to visualize.")
    st.stop()

# Aggregation
if agg_mode == "Mean":
    agg_func = "mean"
elif agg_mode == "Median":
    agg_func = "median"
else:
    agg_func = "max"

# -------------------------------
# Tabs
# -------------------------------
tabs = st.tabs(["Overview", "Charts", "Time Heatmap", "Pairplot", "Anomalies", "Forecast", "Summary"])

# --- Overview ---
with tabs[0]:
    st.header("📊 Overview")
    if country_col:
        counts = dff[country_col].value_counts().nlargest(15)
        st.plotly_chart(px.bar(counts, title="Top Countries by Records", template=theme), use_container_width=True)
    for m in metrics[:3]:
        st.plotly_chart(px.histogram(dff, x=m, nbins=40, title=f"Distribution of {m}", template=theme), use_container_width=True)

# --- Charts ---
with tabs[1]:
    st.header("📈 Charts")
    if date_col:
        grouped = dff.groupby(pd.Grouper(key=date_col, freq="D"))[metrics].agg(agg_func).reset_index()
        for m in metrics:
            st.plotly_chart(px.line(grouped, x=date_col, y=m, title=f"Daily {agg_mode} of {m}", template=theme), use_container_width=True)
    if len(metrics) >= 2:
        st.plotly_chart(px.scatter_matrix(dff, dimensions=metrics, color=country_col, title="Metric Relationship Matrix", template=theme), use_container_width=True)

# --- Time Heatmap ---
with tabs[2]:
    st.header("🕓 Time-Based Heatmap")
    if date_col:
        pivot = dff.groupby(["month", "hour"])[metrics[0]].agg(agg_func).unstack()
        st.plotly_chart(px.imshow(pivot, aspect="auto", title=f"{metrics[0]} by Month & Hour", template=theme, color_continuous_scale="RdBu_r"), use_container_width=True)

# --- Pairplot ---
with tabs[3]:
    st.header("📉 Pairwise Correlation Matrix")
    if len(metrics) > 1:
        corr = dff[metrics].corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="Viridis", title="Metric Correlation", template=theme), use_container_width=True)

# --- Anomalies ---
with tabs[4]:
    st.header("🚨 Anomalies")
    num_data = dff.select_dtypes(include=[np.number]).dropna()
    if num_data.shape[0] > 50:
        model = IsolationForest(contamination=0.02, random_state=42)
        preds = model.fit_predict(num_data)
        dff["anomaly"] = preds == -1
        st.success(f"Detected {dff['anomaly'].sum()} anomalies ({(dff['anomaly'].mean()*100):.2f}%)")
        st.dataframe(dff[dff["anomaly"]].head(10))
    else:
        st.info("Not enough data for anomaly detection.")

# --- Forecast ---
with tabs[5]:
    st.header("🔮 Forecast (Prophet)")
    if PROPHET_AVAILABLE and date_col and temp_col:
        data = dff[[date_col, temp_col]].rename(columns={date_col: "ds", temp_col: "y"}).dropna()
        if len(data) > 30:
            m = Prophet()
            m.fit(data)
            future = m.make_future_dataframe(periods=7)
            forecast = m.predict(future)
            st.plotly_chart(px.line(forecast, x="ds", y="yhat", title="7-Day Forecast", template=theme), use_container_width=True)
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7))
        else:
            st.info("Not enough data to forecast.")
    else:
        st.warning("Prophet not available or temperature column missing.")

# --- Summary ---
with tabs[6]:
    st.header("📋 Summary Statistics")
    st.dataframe(dff[metrics].describe().T)

st.success("✅ Weather Dashboard Pro+ ready — with all filters, metrics, and visual power!")