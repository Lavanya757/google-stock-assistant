import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Title
st.title("📈 Google Stock Data Assistant")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("google_stock_data.csv", parse_dates=['Date'])
    df.sort_values("Date", inplace=True)
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Filter")
start_date = st.sidebar.date_input("Start Date", df['Date'].min().date())
end_date = st.sidebar.date_input("End Date", df['Date'].max().date())

filtered_df = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)]

# Show data
st.subheader("📊 Filtered Stock Data")
st.write(filtered_df)

# Summary stats
st.subheader("📌 Summary Statistics")
st.write(filtered_df.describe())

# Plotting
st.subheader("📉 Price Over Time")
fig, ax = plt.subplots()
ax.plot(filtered_df["Date"], filtered_df["Close"], label='Close Price')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title("Google Stock Close Price Over Time")
ax.legend()
st.pyplot(fig)

# Optional: Simple Prediction (Next day trend using linear regression)
if st.checkbox("🤖 Predict Next Day's Close Price (Basic Linear Regression)"):
    df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
    X = df['Date_ordinal'].values.reshape(-1, 1)
    y = df['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    next_day = df['Date'].max() + pd.Timedelta(days=1)
    next_day_ordinal = np.array([[next_day.toordinal()]])
    predicted_price = model.predict(next_day_ordinal)[0]

    st.success(f"📌 Predicted close price for {next_day.date()}: ₹{predicted_price:.2f}")

