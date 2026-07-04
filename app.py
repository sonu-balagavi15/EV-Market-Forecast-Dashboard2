import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="EV Market Forecast Dashboard",
    page_icon="🚗",
    layout="wide"
)

# -------------------- Title --------------------
st.title("🚗 EV Market Forecast Dashboard")
st.markdown("### Analyze Electric Vehicle Sales and Predict Future Growth")

st.markdown("---")

# -------------------- Load Data --------------------
df = pd.read_csv("data/ev_sales.csv")

# -------------------- Sidebar --------------------
st.sidebar.title("⚙️ Dashboard Filters")

country = st.sidebar.selectbox(
    "🌍 Select Country",
    df["Country"].unique()
)

filtered_df = df[df["Country"] == country]

# -------------------- KPI Cards --------------------
total_sales = filtered_df["EV_Sales"].sum()
average_sales = int(filtered_df["EV_Sales"].mean())
highest_sales = filtered_df["EV_Sales"].max()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🚗 Total EV Sales", f"{total_sales:,}")

with col2:
    st.metric("📊 Average Sales", f"{average_sales:,}")

with col3:
    st.metric("🏆 Highest Annual Sales", f"{highest_sales:,}")

st.markdown("---")

# -------------------- Data Table --------------------
st.subheader(f"📋 {country} EV Sales Data")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# -------------------- Charts --------------------
left, right = st.columns(2)

with left:
    line_chart = px.line(
        filtered_df,
        x="Year",
        y="EV_Sales",
        markers=True,
        title=f"{country} EV Sales Trend"
    )

    line_chart.update_layout(
        xaxis_title="Year",
        yaxis_title="EV Sales"
    )

    st.plotly_chart(line_chart, use_container_width=True)

with right:
    bar_chart = px.bar(
        filtered_df,
        x="Year",
        y="EV_Sales",
        color="EV_Sales",
        text_auto=True,
        title="Year-wise EV Sales"
    )

    st.plotly_chart(bar_chart, use_container_width=True)

st.markdown("---")

# -------------------- Forecast --------------------
st.subheader("🔮 EV Sales Forecast (Next 5 Years)")

X = filtered_df[["Year"]]
y = filtered_df["EV_Sales"]

model = LinearRegression()
model.fit(X, y)

future_years = np.array([2024, 2025, 2026, 2027, 2028]).reshape(-1, 1)

predictions = model.predict(future_years)

forecast_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Predicted EV Sales": predictions.astype(int)
})

st.dataframe(
    forecast_df,
    use_container_width=True
)

forecast_chart = px.line(
    forecast_df,
    x="Year",
    y="Predicted EV Sales",
    markers=True,
    title="Forecasted EV Sales"
)

forecast_chart.update_layout(
    xaxis_title="Year",
    yaxis_title="Predicted Sales"
)

st.plotly_chart(
    forecast_chart,
    use_container_width=True
)

# -------------------- Download --------------------
csv = forecast_df.to_csv(index=False)

st.download_button(
    label="📥 Download Forecast CSV",
    data=csv,
    file_name="EV_Forecast.csv",
    mime="text/csv"
)

st.markdown("---")

# -------------------- Footer --------------------
st.markdown(
    """
    <center>
    <h4>🚗 EV Market Forecast Dashboard</h4>
    <p>Developed using <b>Python • Streamlit • Plotly • Scikit-Learn</b></p>
    </center>
    """,
    unsafe_allow_html=True
)