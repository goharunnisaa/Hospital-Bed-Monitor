import pandas as pd
import pickle
import streamlit as st
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Hospital Bed Monitor", layout="wide")

# ========= WHITE + NEON GREEN TEXT + DARK BG =========
st.markdown("""
<style>
html, body, [class*="block-container"], .stMarkdown, .stSelectbox, label, p, span, h1, h2, h3, div {
    color: #FFFFFF !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #1a1a1a !important;
}

[data-testid="stHeader"] {
    background-color: #000000 !important;
}

/* KPI Number Color */
span[data-testid="stMetricValue"] {
    color: #00FF99 !important;
    font-size: 32px !important;
    font-weight: 900 !important;
}

/* KPI Label Color */
span[data-testid="stMetricLabel"] {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ========= DATA LOADING =========
df = pd.read_csv("data/hospital_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

with open("models.pkl", "rb") as f:
    model = pickle.load(f)

# ========= HEADER =========
st.title("🏥 Hospital Bed Occupancy Monitoring & Prediction")

# ========= SELECT HOSPITAL =========
hospital = st.selectbox("🏨 Select Hospital", df["Hospital"].unique())
hospital_df = df[df["Hospital"] == hospital]

# ========= CALCULATE CURRENT STATUS =========
latest = hospital_df.iloc[-1]
beds_occupied = int(latest["Bed_Used"])
total_beds = int(latest["Total_Beds"])
beds_available = total_beds - beds_occupied

# More Safe status (according to you 😄)
status_txt = "Safe" if beds_available > 40 else "Warning" if beds_available > 10 else "Critical"
status_color = "green" if status_txt == "Safe" else "orange" if status_txt == "Warning" else "red"

# ========= KPI CARDS =========
c1,c2,c3 = st.columns(3)
c1.metric("Total Beds", total_beds)
c2.metric("Occupied", beds_occupied)
c3.metric("Available", beds_available)

# Status indicators
icon = "🟢" if status_txt == "Safe" else "🟠" if status_txt == "Warning" else "🔴"

text_color = "#FFFFFF"  # Always white text for readability
underline_color = "green" if status_txt == "Safe" else "orange" if status_txt == "Warning" else "red"

st.markdown(
    f"<span style='font-size:22px; font-weight:900;'>{icon} "
    f"<span style='color:{text_color}; text-decoration: underline; text-decoration-color:{underline_color};'>"
    f"{status_txt}</span></span>",
    unsafe_allow_html=True
)


# ========= FORECAST =========
future_days = 7
last_row = hospital_df.iloc[-1]
last_day = last_row["Date"].dayofyear

future_features = []
future_dates = []

for i in range(1, future_days + 1):
    future_date = last_row["Date"] + pd.Timedelta(days=i)
    future_features.append([future_date.dayofyear, future_date.weekday()])
    future_dates.append(future_date)

predicted = model.predict(np.array(future_features))
pred_df = pd.DataFrame({"Date": future_dates, "Predicted Bed Usage": predicted})

# ========= CHART =========
st.subheader("📈 Next 7-Day Forecast")

fig = px.line(
    pred_df, 
    x="Date", 
    y="Predicted Bed Usage", 
    markers=True,
    line_shape="spline",
    color_discrete_sequence=["#FFFF00"]  # Bright Yellow
)

fig.update_layout(
    plot_bgcolor="#1a1a1a",
    paper_bgcolor="#1a1a1a",
    font=dict(color="white"),
    height=380,
)

st.plotly_chart(fig, use_container_width=True)

# ========= SHORTAGE ALERTS =========
st.subheader("🚨 Shortage Alerts")
shortage = pred_df[pred_df["Predicted Bed Usage"] > total_beds]

if not shortage.empty:
    st.error("⚠ Shortage expected! Take action.")
    for _, r in shortage.iterrows():
        st.write(f"➡ {r['Date'].date()} → Need **{int(r['Predicted Bed Usage']-total_beds)}** more beds")
else:
    st.success("✔ No shortage this week")

# ========= DOWNLOAD =========
st.download_button(
    "⬇ Download Report",
    pred_df.to_csv(index=False),
    "forecast.csv"
)
