import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AirSight AI", layout="wide")

st.title("🌍 AirSight AI")
st.subheader("Urban Air Quality Intelligence Platform")

df = pd.read_csv("data/aqi_data.csv")

# KPI Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Average AQI", int(df["aqi"].mean()))
col2.metric("Highest AQI", int(df["aqi"].max()))
col3.metric("Hotspots", len(df[df["aqi"] > 200]))

# Heatmap Map
st.header("City Pollution Map")

m = folium.Map(
    location=[20.30, 85.82],
    zoom_start=12
)

for _, row in df.iterrows():
    color = "green"

    if row["aqi"] > 300:
        color = "red"
    elif row["aqi"] > 200:
        color = "orange"

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=10,
        color=color,
        fill=True,
        popup=f"{row['ward']} AQI:{row['aqi']}"
    ).add_to(m)

folium_static(m)

# AQI Forecast
st.header("24 Hour AQI Forecast")

X = df[["traffic","construction","industry","temp","humidity"]]
y = df["aqi"]

model = RandomForestRegressor()
model.fit(X, y)

future = X.copy()
future["traffic"] += 10

forecast = model.predict(future)

forecast_df = pd.DataFrame({
    "Ward": df["ward"],
    "Forecast AQI": forecast
})

fig = px.bar(
    forecast_df,
    x="Ward",
    y="Forecast AQI",
    title="Predicted AQI"
)

st.plotly_chart(fig)

# Source Attribution
st.header("Pollution Source Attribution")

def source(row):
    factors = {
        "Traffic": row["traffic"],
        "Construction": row["construction"],
        "Industry": row["industry"]
    }

    return max(factors, key=factors.get)

df["Main Source"] = df.apply(source, axis=1)

st.dataframe(
    df[["ward","aqi","Main Source"]]
)

# Enforcement Ranking
st.header("Enforcement Priority")

priority = df.sort_values(
    by=["aqi"],
    ascending=False
)

st.dataframe(
    priority[["ward","aqi","Main Source"]]
)

# Citizen Advisory
st.header("Health Advisory")

avg_aqi = df["aqi"].mean()

if avg_aqi > 300:
    st.error(
        "Severe Pollution: Avoid outdoor activity."
    )
elif avg_aqi > 200:
    st.warning(
        "Poor Air Quality: Wear N95 masks."
    )
else:
    st.success(
        "Air Quality Acceptable."
    )