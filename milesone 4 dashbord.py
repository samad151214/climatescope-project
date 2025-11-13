# weather_dashboard_ultra_v2.py
"""
Weather Intelligence — Ultra v2
- Separate scatter charts (no scatter matrix)
- All KPIs working dynamically
- Fixed-threshold extreme events (confirmed thresholds)
- Anomaly detection, forecasting (Prophet optional), maps, downloads
- Enhanced with real-time metrics, weather alerts, and better visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# Optional Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="🌍 Weather Intelligence — Ultra v2", layout="wide")
st.title("🌦 Weather Intelligence — Ultra v2")
st.markdown("KPIs, separate scatter charts, fixed-threshold extremes, anomaly detection, forecasting, maps, and exports.")

# ---------------------------
# Data loading
# ---------------------------
st.sidebar.header("1️⃣ Data Setup")
uploaded = st.sidebar.file_uploader("Upload Weather CSV", type=["csv"])

@st.cache_data
def load_csv(path=None):
    if path:
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()

if uploaded:
    try:
        df = pd.read_csv(uploaded, low_memory=False)
        st.sidebar.success(f"✅ Uploaded: {uploaded.name} ({len(df):,} rows)")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    # Try to use the provided file content
    try:
        from io import StringIO
        file_content = """country,location_name,latitude,longitude,timezone,last_updated_epoch,last_updated,temperature_celsius,temperature_fahrenheit,condition_text,wind_mph,wind_kph,wind_degree,wind_direction,pressure_mb,pressure_in,precip_mm,precip_in,humidity,cloud,feels_like_celsius,feels_like_fahrenheit,visibility_km,visibility_miles,uv_index,gust_mph,gust_kph,air_quality_Carbon_Monoxide,air_quality_Ozone,air_quality_Nitrogen_dioxide,air_quality_Sulphur_dioxide,air_quality_PM2.5,air_quality_PM10,air_quality_us-epa-index,air_quality_gb-defra-index,sunrise,sunset,moonrise,moonset,moon_phase,moon_illumination
Afghanistan,Kabul,34.52,69.18,Asia/Kabul,1693301400,2023-08-29 14:00,28.8,83.8,Sunny,7.2,11.5,74,ENE,1004.0,29.64,0.0,0.0,19,0,26.7,80.1,10.0,6.0,7.0,8.3,13.3,647.5,130.2,1.2,0.4,7.9,11.1,1,1,05:24 AM,06:24 PM,05:39 PM,02:48 AM,Waxing Gibbous,93
Albania,Tirana,41.33,19.82,Europe/Tirane,1693301400,2023-08-29 11:30,27.0,80.6,Partly cloudy,3.8,6.1,210,SSW,1006.0,29.71,0.0,0.0,54,75,28.0,82.3,10.0,6.0,6.0,7.4,11.9,433.9,104.4,3.6,1.8,28.2,29.6,2,3,06:04 AM,07:19 PM,06:50 PM,03:25 AM,Waxing Gibbous,93"""
        df = pd.read_csv(StringIO(file_content))
        st.sidebar.info("📁 Using sample data")
    except:
        st.sidebar.info("No data loaded. Upload a CSV to begin.")
        df = pd.DataFrame()

if df.empty:
    st.warning("No data loaded yet. Upload a CSV in the sidebar to continue.")
    st.stop()

# Clean column names
df.columns = [c.strip() for c in df.columns]
cols_lower = {c.lower(): c for c in df.columns}

def find_col(*names):
    for n in names:
        if n and n.lower() in cols_lower:
            return cols_lower[n.lower()]
    return None

# Detect columns with more comprehensive search
date_col = find_col("last_updated", "date", "datetime", "timestamp", "time")
temp_col = find_col("temperature_celsius", "temperature", "temp", "t")
humidity_col = find_col("humidity", "hum", "rh")
wind_col = find_col("wind_kph", "wind_mph", "windspeed", "wind", "ws")
pressure_col = find_col("pressure_mb", "pressure", "atm_pressure", "barometer")
precip_col = find_col("precip_mm", "precipitation", "rain", "snow", "precip")
aqi_col = find_col("air_quality_us-epa-index", "airquality", "aqi")
cloud_col = find_col("cloud", "cloudiness", "cloud_cover")
country_col = find_col("country", "country_name")
city_col = find_col("location_name", "city", "location", "town")
weather_col = find_col("condition_text", "weather", "conditions", "description")
lat_col = find_col("latitude", "lat")
lon_col = find_col("longitude", "lon", "lng")

# Convert types
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
for c in [temp_col, humidity_col, wind_col, pressure_col, precip_col, aqi_col, cloud_col, lat_col, lon_col]:
    if c and c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Derived time features
if date_col:
    df["month"] = df[date_col].dt.month
    df["hour"] = df[date_col].dt.hour
    df["day_of_week"] = df[date_col].dt.day_name()
    df["date_only"] = df[date_col].dt.date
else:
    df["month"] = np.nan
    df["hour"] = np.nan
    df["day_of_week"] = np.nan
    df["date_only"] = np.nan

# Create AQI category if AQI data exists
if aqi_col and aqi_col in df.columns:
    def categorize_aqi(aqi):
        if pd.isna(aqi):
            return "Unknown"
        elif aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    df["aqi_category"] = df[aqi_col].apply(categorize_aqi)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("2️⃣ Filters & Controls")
if st.sidebar.button("🔄 Reset All Filters"):
    st.experimental_rerun()

# Country / City filters
countries = sorted(df[country_col].dropna().unique()) if country_col and not df[country_col].dropna().empty else []
selected_countries = st.sidebar.multiselect("🌍 Country", countries, default=countries[:3] if len(countries) > 3 else countries)
if selected_countries and city_col:
    cities = sorted(df[df[country_col].isin(selected_countries)][city_col].dropna().unique())
else:
    cities = sorted(df[city_col].dropna().unique()) if city_col else []
selected_cities = st.sidebar.multiselect("🏙 City", cities, default=cities[:5] if len(cities) > 5 else cities)

# Weather type
weathers = sorted(df[weather_col].dropna().unique()) if weather_col else []
selected_weather = st.sidebar.multiselect("☁ Weather Type", weathers)

# Date range
if date_col:
    min_date, max_date = df[date_col].min().date(), df[date_col].max().date()
    selected_dates = st.sidebar.date_input("📅 Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
else:
    selected_dates = None

# Numeric range helper
def numeric_range_widget(label, col):
    if col and col in df and not df[col].dropna().empty:
        lo = float(df[col].min())
        hi = float(df[col].max())
        return st.sidebar.slider(label, lo, hi, (lo, hi))
    return None

temp_range = numeric_range_widget("🌡 Temperature range (°C)", temp_col)
hum_range = numeric_range_widget("💧 Humidity range (%)", humidity_col)
wind_range = numeric_range_widget("🌬 Wind range (kph)", wind_col)
precip_range = numeric_range_widget("🌧 Precipitation range (mm)", precip_col)

# Aggregation and theme
agg_mode = st.sidebar.selectbox("📊 Aggregation", ["Mean", "Median", "Max", "Min"], index=0)
template = st.sidebar.selectbox("🎨 Plotly Template", ["plotly", "plotly_white", "simple_white", "ggplot2", "seaborn"], index=0)

# Fixed thresholds (confirmed)
st.sidebar.markdown("### ⚠️ Fixed Thresholds for Extreme Events")
TEMP_HIGH = st.sidebar.number_input("Temp HIGH threshold (°C)", value=35.0, step=0.1)
TEMP_LOW = st.sidebar.number_input("Temp LOW threshold (°C)", value=5.0, step=0.1)
WIND_HIGH = st.sidebar.number_input("Wind HIGH threshold (kph)", value=50.0, step=0.1)
PRECIP_HIGH = st.sidebar.number_input("Precipitation HIGH threshold (mm)", value=20.0, step=0.1)
AQI_HIGH = st.sidebar.number_input("AQI HIGH threshold", value=100.0, step=1.0)

# Anomaly detection
st.sidebar.markdown("### 🔎 Anomaly Detection")
run_anomaly = st.sidebar.checkbox("Enable IsolationForest anomalies", value=True)
anomaly_contamination = st.sidebar.slider("Contamination (outlier fraction)", 0.001, 0.2, 0.05, step=0.001)

# Prophet forecast
st.sidebar.markdown("### 🔮 Forecast (Prophet)")
use_prophet = False
if PROPHET_AVAILABLE:
    use_prophet = st.sidebar.checkbox("Use Prophet forecasting", value=False)
    forecast_days = st.sidebar.number_input("Forecast horizon (days)", min_value=1, max_value=365, value=7)
else:
    st.sidebar.info("Prophet not installed for forecasting")

# Metric selection
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
default_metric = temp_col if temp_col else (num_cols[0] if num_cols else None)
metrics = st.sidebar.multiselect("📈 Metrics to analyze (choose 1 or more)", num_cols, 
                               default=[default_metric] if default_metric else [],
                               max_selections=6)

# New Features
st.sidebar.header("3️⃣ New Features")
enable_weather_alerts = st.sidebar.checkbox("Enable Weather Alerts", value=True)
enable_comfort_index = st.sidebar.checkbox("Enable Comfort Index", value=True)
enable_trend_analysis = st.sidebar.checkbox("Enable Trend Analysis", value=True)

# ---------------------------
# Apply filters
# ---------------------------
dff = df.copy()
if country_col and selected_countries:
    dff = dff[dff[country_col].isin(selected_countries)]
if city_col and selected_cities:
    dff = dff[dff[city_col].isin(selected_cities)]
if weather_col and selected_weather:
    dff = dff[dff[weather_col].isin(selected_weather)]

for col, rng in [
    (temp_col, temp_range),
    (humidity_col, hum_range),
    (wind_col, wind_range),
    (precip_col, precip_range)
]:
    if col and rng:
        dff = dff[(dff[col] >= rng[0]) & (dff[col] <= rng[1])]

if date_col and selected_dates and len(selected_dates) == 2:
    start, end = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
    dff = dff[(dff[date_col] >= start) & (dff[date_col] <= end)]

st.write(f"### 📊 Showing {len(dff):,} records after filters")

# ---------------------------
# NEW: Weather Alerts System
# ---------------------------
def generate_weather_alerts(data):
    alerts = []
    
    # Temperature alerts
    if temp_col and temp_col in data.columns:
        high_temp = data[data[temp_col] >= TEMP_HIGH]
        if len(high_temp) > 0:
            alerts.append(f"🔥 {len(high_temp)} records with temperature ≥ {TEMP_HIGH}°C")
        
        low_temp = data[data[temp_col] <= TEMP_LOW]
        if len(low_temp) > 0:
            alerts.append(f"❄️ {len(low_temp)} records with temperature ≤ {TEMP_LOW}°C")
    
    # Wind alerts
    if wind_col and wind_col in data.columns:
        high_wind = data[data[wind_col] >= WIND_HIGH]
        if len(high_wind) > 0:
            alerts.append(f"💨 {len(high_wind)} records with wind speed ≥ {WIND_HIGH} kph")
    
    # Precipitation alerts
    if precip_col and precip_col in data.columns:
        high_precip = data[data[precip_col] >= PRECIP_HIGH]
        if len(high_precip) > 0:
            alerts.append(f"🌧 {len(high_precip)} records with precipitation ≥ {PRECIP_HIGH} mm")
    
    # AQI alerts
    if aqi_col and aqi_col in data.columns:
        high_aqi = data[data[aqi_col] >= AQI_HIGH]
        if len(high_aqi) > 0:
            alerts.append(f"😷 {len(high_aqi)} records with AQI ≥ {AQI_HIGH}")
    
    return alerts

# ---------------------------
# NEW: Comfort Index Calculation
# ---------------------------
def calculate_comfort_index(temp, humidity):
    """Calculate temperature-humidity comfort index (simplified)"""
    if pd.isna(temp) or pd.isna(humidity):
        return None
    
    # Simple comfort calculation based on temp and humidity
    if temp < 10:
        base_comfort = 0.3
    elif temp < 20:
        base_comfort = 0.7
    elif temp < 25:
        base_comfort = 0.9
    elif temp < 30:
        base_comfort = 0.6
    else:
        base_comfort = 0.2
    
    # Adjust for humidity
    if humidity > 80:
        humidity_penalty = 0.3
    elif humidity > 60:
        humidity_penalty = 0.1
    else:
        humidity_penalty = 0
    
    comfort = max(0, base_comfort - humidity_penalty)
    return round(comfort * 100, 1)

if enable_comfort_index and temp_col and humidity_col:
    dff["comfort_index"] = dff.apply(
        lambda row: calculate_comfort_index(row[temp_col], row[humidity_col]), axis=1
    )

# ---------------------------
# Enhanced KPI Row
# ---------------------------
st.header("📈 Real-time Weather Metrics")

# Display weather alerts
if enable_weather_alerts:
    alerts = generate_weather_alerts(dff)
    if alerts:
        st.warning("🚨 **Weather Alerts**")
        for alert in alerts:
            st.write(f"• {alert}")

kpi_cols = st.columns(6)

def safe_stat(series, func):
    if series is None or series.dropna().empty:
        return None
    return func(series)

def safe_metric(label, value, delta=None, help_text=None):
    if value is not None:
        if isinstance(value, float):
            value = round(value, 2)
        st.metric(label, value, delta=delta, help=help_text)
    else:
        st.metric(label, "N/A")

# Records
with kpi_cols[0]:
    st.metric("📋 Records", f"{len(dff):,}")

# Temperature KPI
with kpi_cols[1]:
    if temp_col and temp_col in dff.columns and not dff[temp_col].dropna().empty:
        avg_temp = safe_stat(dff[temp_col], np.mean)
        min_temp = safe_stat(dff[temp_col], np.min)
        max_temp = safe_stat(dff[temp_col], np.max)
        delta_temp = round(max_temp - min_temp, 1) if all(x is not None for x in [max_temp, min_temp]) else None
        safe_metric("🌡 Avg Temp (°C)", avg_temp, delta=f"Δ{delta_temp}°C" if delta_temp else None)

# Humidity KPI
with kpi_cols[2]:
    if humidity_col and humidity_col in dff.columns and not dff[humidity_col].dropna().empty:
        avg_hum = safe_stat(dff[humidity_col], np.mean)
        safe_metric("💧 Avg Humidity (%)", avg_hum)

# Wind KPI
with kpi_cols[3]:
    if wind_col and wind_col in dff.columns and not dff[wind_col].dropna().empty:
        avg_wind = safe_stat(dff[wind_col], np.mean)
        max_wind = safe_stat(dff[wind_col], np.max)
        safe_metric("🌬 Avg Wind (kph)", avg_wind, delta=f"Max: {max_wind}" if max_wind else None)

# Precipitation KPI
with kpi_cols[4]:
    if precip_col and precip_col in dff.columns and not dff[precip_col].dropna().empty:
        total_precip = safe_stat(dff[precip_col], np.sum)
        max_precip = safe_stat(dff[precip_col], np.max)
        safe_metric("🌧 Total Precip (mm)", total_precip, delta=f"Max: {max_precip}" if max_precip else None)

# Comfort Index or AQI KPI
with kpi_cols[5]:
    if enable_comfort_index and "comfort_index" in dff.columns and not dff["comfort_index"].dropna().empty:
        avg_comfort = safe_stat(dff["comfort_index"], np.mean)
        comfort_level = "Excellent" if avg_comfort > 80 else "Good" if avg_comfort > 60 else "Fair" if avg_comfort > 40 else "Poor"
        safe_metric("😊 Comfort Index", avg_comfort, delta=comfort_level)
    elif aqi_col and aqi_col in dff.columns and not dff[aqi_col].dropna().empty:
        avg_aqi = safe_stat(dff[aqi_col], np.mean)
        aqi_level = "Good" if avg_aqi <= 50 else "Moderate" if avg_aqi <= 100 else "Poor"
        safe_metric("😷 Avg AQI", avg_aqi, delta=aqi_level)

# ---------------------------
# NEW: Quick Insights Row
# ---------------------------
st.subheader("💡 Quick Insights")
insight_cols = st.columns(4)

with insight_cols[0]:
    if country_col and not dff[country_col].dropna().empty:
        top_country = dff[country_col].mode().iloc[0] if not dff[country_col].mode().empty else "N/A"
        st.metric("📍 Most Common Country", top_country)

with insight_cols[1]:
    if weather_col and not dff[weather_col].dropna().empty:
        common_weather = dff[weather_col].mode().iloc[0] if not dff[weather_col].mode().empty else "N/A"
        st.metric("☁ Most Common Weather", common_weather)

with insight_cols[2]:
    if date_col and not dff[date_col].dropna().empty:
        latest_date = dff[date_col].max().strftime('%Y-%m-%d') if pd.notna(dff[date_col].max()) else "N/A"
        st.metric("📅 Latest Data", latest_date)

with insight_cols[3]:
    if temp_col and humidity_col and not dff[temp_col].dropna().empty and not dff[humidity_col].dropna().empty:
        temp_hum_corr = dff[temp_col].corr(dff[humidity_col])
        if not pd.isna(temp_hum_corr):
            corr_strength = "Strong" if abs(temp_hum_corr) > 0.7 else "Moderate" if abs(temp_hum_corr) > 0.3 else "Weak"
            st.metric("📊 Temp-Humidity Correlation", f"{temp_hum_corr:.2f}", delta=corr_strength)

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(["Overview", "Charts", "Time Analysis", "Box & Violin", "Maps", "Anomalies", "Forecast", "Extreme Events", "Export"])

# Overview Tab
with tabs[0]:
    st.header("📊 Data Overview")
    
    # Data summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Summary")
        summary_data = {
            "Total Records": len(dff),
            "Countries": dff[country_col].nunique() if country_col else 0,
            "Cities": dff[city_col].nunique() if city_col else 0,
            "Date Range": f"{dff[date_col].min().date()} to {dff[date_col].max().date()}" if date_col else "N/A",
            "Weather Types": dff[weather_col].nunique() if weather_col else 0
        }
        for key, value in summary_data.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.subheader("Data Quality")
        quality_data = {
            "Complete Records": dff.notna().all(axis=1).sum(),
            "Missing Values": dff.isna().sum().sum(),
            "Duplicate Records": dff.duplicated().sum()
        }
        for key, value in quality_data.items():
            st.write(f"**{key}:** {value}")
    
    # Top countries chart
    if country_col:
        country_counts = dff[country_col].value_counts().nlargest(10)
        if not country_counts.empty:
            fig = px.bar(country_counts, x=country_counts.index, y=country_counts.values,
                         labels={'x': 'Country', 'y': 'Records'}, 
                         title="Top 10 Countries by Records", template=template)
            st.plotly_chart(fig, use_container_width=True)
    
    # Weather distribution
    if weather_col:
        weather_counts = dff[weather_col].value_counts().nlargest(8)
        if not weather_counts.empty:
            fig = px.pie(weather_counts, values=weather_counts.values, names=weather_counts.index,
                         title="Weather Condition Distribution", template=template)
            st.plotly_chart(fig, use_container_width=True)

# Charts Tab
with tabs[1]:
    st.header("📈 Interactive Charts")
    
    # Time series charts
    if date_col and metrics:
        st.subheader("Time Series Analysis")
        time_metric = st.selectbox("Select metric for time series", metrics, key="time_metric")
        
        if st.checkbox("Show daily aggregation", value=True):
            grouped = dff.groupby(pd.Grouper(key=date_col, freq="D"))[time_metric].agg(agg_mode.lower()).reset_index()
            fig = px.line(grouped, x=date_col, y=time_metric, 
                         title=f"Daily {agg_mode} {time_metric} Over Time", template=template)
        else:
            fig = px.scatter(dff, x=date_col, y=time_metric, color=country_col if country_col else None,
                           title=f"{time_metric} Over Time", template=template)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    if len(metrics) >= 2:
        st.subheader("Correlation Analysis")
        corr_matrix = dff[metrics].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                       title="Correlation Matrix", template=template)
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots
    if len(metrics) >= 2:
        st.subheader("Scatter Plots")
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("X-axis", metrics, index=0, key="scatter_x")
        with col2:
            y_axis = st.selectbox("Y-axis", metrics, index=min(1, len(metrics)-1), key="scatter_y")
        
        color_by = st.selectbox("Color by", [None, country_col, city_col, weather_col] + 
                               (["aqi_category"] if "aqi_category" in dff.columns else []), key="scatter_color")
        
        fig = px.scatter(dff, x=x_axis, y=y_axis, color=color_by, 
                        hover_data=[city_col] if city_col else None,
                        title=f"{x_axis} vs {y_axis}", template=template)
        st.plotly_chart(fig, use_container_width=True)

# Time Analysis Tab
with tabs[2]:
    st.header("🕓 Time Pattern Analysis")
    
    if date_col and metrics:
        # Hourly patterns
        st.subheader("Hourly Patterns")
        hour_metric = st.selectbox("Select metric for hourly analysis", metrics, key="hour_metric")
        
        hourly_avg = dff.groupby("hour")[hour_metric].mean().reset_index()
        fig = px.line(hourly_avg, x="hour", y=hour_metric, 
                     title=f"Average {hour_metric} by Hour of Day", template=template)
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly patterns
        st.subheader("Weekly Patterns")
        if "day_of_week" in dff.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_avg = dff.groupby("day_of_week")[hour_metric].mean().reindex(day_order).reset_index()
            fig = px.line(weekly_avg, x="day_of_week", y=hour_metric,
                         title=f"Average {hour_metric} by Day of Week", template=template)
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        st.subheader("Heatmap: Month vs Hour")
        try:
            pivot_data = dff.groupby(["month", "hour"])[hour_metric].mean().unstack()
            fig = px.imshow(pivot_data, aspect="auto", 
                          title=f"{hour_metric} Heatmap (Month × Hour)", template=template)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Could not generate heatmap: {e}")

# Box & Violin Tab
with tabs[3]:
    st.header("📦 Distribution Analysis")
    
    if metrics:
        box_metric = st.selectbox("Metric for distribution analysis", metrics, index=0, key="box_metric")
        group_by = st.selectbox("Group by (optional)", [None, country_col, city_col, weather_col], key="group_by")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot
            if group_by and group_by in dff.columns:
                fig = px.box(dff, x=group_by, y=box_metric, 
                           title=f"Box Plot of {box_metric} by {group_by}", template=template)
            else:
                fig = px.box(dff, y=box_metric, title=f"Box Plot of {box_metric}", template=template)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Violin plot
            if group_by and group_by in dff.columns:
                fig = px.violin(dff, x=group_by, y=box_metric, box=True, points="all",
                              title=f"Violin Plot of {box_metric} by {group_by}", template=template)
            else:
                fig = px.violin(dff, y=box_metric, box=True, points="all",
                              title=f"Violin Plot of {box_metric}", template=template)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric metrics selected for distribution analysis.")

# Maps Tab
with tabs[4]:
    st.header("🗺 Geographic Analysis")
    
    if lat_col and lon_col:
        display_map = dff.dropna(subset=[lat_col, lon_col])
        if not display_map.empty:
            st.subheader("Interactive Map")
            map_metric = st.selectbox("Map metric (size/color)", [None] + (metrics if metrics else []), key="map_metric")
            
            if map_metric:
                fig = px.scatter_geo(display_map, lat=lat_col, lon=lon_col, 
                                   color=map_metric, size=map_metric,
                                   hover_name=city_col if city_col else None, 
                                   hover_data=[country_col] if country_col else None,
                                   projection="natural earth", 
                                   title=f"Geographic Distribution - {map_metric}",
                                   template=template)
            else:
                fig = px.scatter_geo(display_map, lat=lat_col, lon=lon_col,
                                   hover_name=city_col if city_col else None,
                                   hover_data=[country_col] if country_col else None,
                                   projection="natural earth", 
                                   title="Geographic Distribution",
                                   template=template)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No geographic data available after filtering.")
    else:
        st.info("Latitude/longitude columns not detected.")

    # Choropleth map
    if country_col:
        st.subheader("Choropleth Map")
        choropleth_metric = st.selectbox("Select metric for choropleth", 
                                       ["records"] + (metrics if metrics else []), 
                                       key="choropleth_metric")
        
        if choropleth_metric == "records":
            country_data = dff.groupby(country_col).size().reset_index(name="records")
            color_col = "records"
            title = "Records by Country"
        else:
            country_data = dff.groupby(country_col)[choropleth_metric].mean().reset_index()
            color_col = choropleth_metric
            title = f"Average {choropleth_metric} by Country"
        
        if not country_data.empty:
            try:
                fig = px.choropleth(country_data, locations=country_col, 
                                  locationmode="country names", color=color_col,
                                  hover_name=country_col, title=title, 
                                  template=template)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Choropleth unavailable: {e}")

# Anomalies Tab
with tabs[5]:
    st.header("🚨 Anomaly Detection")
    
    if run_anomaly and metrics:
        numeric_for_anom = dff[metrics].dropna()
        
        if len(numeric_for_anom) < 10:
            st.info("Not enough numeric rows to run anomaly detection.")
        else:
            with st.spinner("Running anomaly detection..."):
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_for_anom)
                
                # Run Isolation Forest
                iso = IsolationForest(contamination=anomaly_contamination, random_state=42)
                preds = iso.fit_predict(scaled_data)
                
                # Create results
                anom_df = numeric_for_anom.copy()
                anom_df["_anomaly"] = preds == -1
                anom_df["_anomaly_score"] = iso.decision_function(scaled_data)
                
                anom_count = anom_df["_anomaly"].sum()
                st.success(f"Detected {anom_count} anomalies ({(anom_count/len(anom_df))*100:.2f}%)")
                
                # Show anomalies with context
                ctx_cols = [c for c in [date_col, country_col, city_col, weather_col] if c in dff.columns]
                if ctx_cols:
                    merged = pd.concat([dff.loc[anom_df.index, ctx_cols].reset_index(drop=True), 
                                      anom_df.reset_index(drop=True)], axis=1)
                    
                    st.subheader("Detected Anomalies")
                    st.dataframe(merged[merged["_anomaly"] == True].head(100))
                    
                    # Anomaly distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Anomalies by country
                        if country_col:
                            anom_by_country = merged[merged["_anomaly"] == True][country_col].value_counts()
                            if not anom_by_country.empty:
                                fig = px.bar(anom_by_country, x=anom_by_country.index, y=anom_by_country.values,
                                           title="Anomalies by Country", template=template)
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Anomaly scores distribution
                        fig = px.histogram(anom_df, x="_anomaly_score", color="_anomaly",
                                         title="Anomaly Scores Distribution", template=template)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(anom_df[anom_df["_anomaly"] == True].head(100))
    else:
        st.info("Anomaly detection disabled or no metrics selected.")

# Forecast Tab
with tabs[6]:
    st.header("🔮 Weather Forecasting")
    
    if use_prophet and PROPHET_AVAILABLE and date_col and temp_col and temp_col in dff.columns:
        data = dff[[date_col, temp_col]].rename(columns={date_col: "ds", temp_col: "y"}).dropna()
        
        if len(data) > 30:
            with st.spinner("Training Prophet model..."):
                m = Prophet()
                m.fit(data)
                
                future = m.make_future_dataframe(periods=int(forecast_days))
                forecast = m.predict(future)
                
                # Plot forecast
                fig = m.plot(forecast)
                st.pyplot(fig)
                
                # Plot components
                st.subheader("Forecast Components")
                comp_fig = m.plot_components(forecast)
                st.pyplot(comp_fig)
                
                # Show forecast data
                st.subheader("Forecast Data")
                forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(int(forecast_days))
                forecast_display.columns = ['Date', 'Predicted', 'Lower Bound', 'Upper Bound']
                st.dataframe(forecast_display)
                
        else:
            st.info("Not enough historical data (>30 rows) for reliable Prophet forecast.")
    else:
        if use_prophet and not PROPHET_AVAILABLE:
            st.warning("Prophet library not installed. Install with: pip install prophet")
        else:
            st.info("Enable Prophet in sidebar and ensure date & temperature columns exist for forecasting.")

# Extreme Events Tab
with tabs[7]:
    st.header("⚡ Extreme Weather Events")
    
    extreme_events = []
    
    # Temperature extremes
    if temp_col:
        high_temp = dff[dff[temp_col] >= TEMP_HIGH]
        low_temp = dff[dff[temp_col] <= TEMP_LOW]
        
        if not high_temp.empty:
            extreme_events.append(("🔥 Heat Events", high_temp, temp_col, f"≥ {TEMP_HIGH}°C"))
        if not low_temp.empty:
            extreme_events.append(("❄️ Cold Events", low_temp, temp_col, f"≤ {TEMP_LOW}°C"))
    
    # Wind extremes
    if wind_col:
        high_wind = dff[dff[wind_col] >= WIND_HIGH]
        if not high_wind.empty:
            extreme_events.append(("💨 High Wind Events", high_wind, wind_col, f"≥ {WIND_HIGH} kph"))
    
    # Precipitation extremes
    if precip_col:
        high_precip = dff[dff[precip_col] >= PRECIP_HIGH]
        if not high_precip.empty:
            extreme_events.append(("🌧 Heavy Precipitation", high_precip, precip_col, f"≥ {PRECIP_HIGH} mm"))
    
    # AQI extremes
    if aqi_col:
        high_aqi = dff[dff[aqi_col] >= AQI_HIGH]
        if not high_aqi.empty:
            extreme_events.append(("😷 Poor Air Quality", high_aqi, aqi_col, f"≥ {AQI_HIGH}"))
    
    if extreme_events:
        for event_name, event_data, metric_col, threshold in extreme_events:
            st.subheader(f"{event_name} ({len(event_data)} events)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Summary statistics
                st.write(f"**Summary (Threshold: {threshold})**")
                stats = {
                    "Average": event_data[metric_col].mean(),
                    "Maximum": event_data[metric_col].max(),
                    "Minimum": event_data[metric_col].min()
                }
                for stat_name, stat_value in stats.items():
                    st.write(f"{stat_name}: {stat_value:.2f}")
            
            with col2:
                # Location summary
                if country_col:
                    top_countries = event_data[country_col].value_counts().head(3)
                    st.write("**Most Affected Countries:**")
                    for country, count in top_countries.items():
                        st.write(f"- {country}: {count} events")
            
            # Histogram
            fig = px.histogram(event_data, x=metric_col, nbins=20,
                             title=f"Distribution of {event_name}", template=template)
            st.plotly_chart(fig, use_container_width=True)
        
        # Combined extreme events map
        st.subheader("🌍 Extreme Events Map")
        all_extremes = pd.concat([data.assign(event_type=name) for name, data, _, _ in extreme_events])
        
        if lat_col and lon_col:
            map_fig = px.scatter_geo(all_extremes, lat=lat_col, lon=lon_col, color="event_type",
                                   hover_name=city_col if city_col else None,
                                   title="Geographic Distribution of Extreme Events",
                                   template=template)
            st.plotly_chart(map_fig, use_container_width=True)
    else:
        st.info("No extreme events detected with current thresholds.")

# Export Tab
with tabs[8]:
    st.header("📋 Data Summary & Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Summary")
        st.dataframe(dff.describe().T if not dff.empty else pd.DataFrame())
    
    with col2:
        st.subheader("Data Quality Report")
        quality_report = pd.DataFrame({
            'Column': dff.columns,
            'Non-Null Count': dff.count(),
            'Null Count': dff.isnull().sum(),
            'Data Type': dff.dtypes
        })
        st.dataframe(quality_report)
    
    st.subheader("📥 Export Options")
    
    # Export filtered data
    csv_data = dff.to_csv(index=False)
    st.download_button(
        "⬇️ Download Filtered CSV",
        data=csv_data,
        file_name="filtered_weather_data.csv",
        mime="text/csv"
    )
    
    # Export summary statistics
    if metrics:
        summary_stats = dff[metrics].describe().T
        csv_summary = summary_stats.to_csv()
        st.download_button(
            "⬇️ Download Summary Statistics",
            data=csv_summary,
            file_name="weather_summary_statistics.csv",
            mime="text/csv"
        )
    
    st.subheader("🔍 Data Preview (First 100 Rows)")
    st.dataframe(dff.head(100))
