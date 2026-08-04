import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Stock Prediction",
    page_icon="📈",
    layout="wide"
)


st.title("📈 AI Stock Prediction Dashboard")


# =========================
# STOCK SELECT
# =========================

stock = st.sidebar.selectbox(
    "Select Stock",
    [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "AAPL"
    ]
)


# =========================
# FETCH DATA
# =========================

@st.cache_data
def load_data(symbol):

    data = yf.download(
        symbol,
        period="1y",
        auto_adjust=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.dropna(inplace=True)

    return data


data = load_data(stock)


if data.empty:
    st.error("No data found")
    st.stop()


# =========================
# INDICATORS
# =========================


# DEMA

ema1 = data["Close"].ewm(span=20).mean()
ema2 = ema1.ewm(span=20).mean()

data["DEMA"] = (2 * ema1) - ema2



# RSI

delta = data["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)


avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()


rs = avg_gain / avg_loss

data["RSI"] = 100 - (100/(1+rs))



# Momentum

data["Momentum"] = (
    data["Close"] -
    data["Close"].shift(10)
)



# Bollinger Bands

middle = data["Close"].rolling(20).mean()

std = data["Close"].rolling(20).std()


data["BB_Upper"] = middle + (2*std)

data["BB_Lower"] = middle - (2*std)


data.dropna(inplace=True)



# =========================
# DATA TABLE
# =========================

st.subheader("📋 Stock Data")

st.dataframe(
    data.tail()
)



# =========================
# CHART
# =========================

st.subheader("📈 Technical Analysis")


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Close"],
        name="Close Price"
    )
)


fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["DEMA"],
        name="DEMA"
    )
)


fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["BB_Upper"],
        name="Upper Band"
    )
)


fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["BB_Lower"],
        name="Lower Band"
    )
)



fig.update_layout(
    height=500,
    title="Stock Price + Indicators"
)


st.plotly_chart(
    fig,
    use_container_width=True
)



# =========================
# RSI MOMENTUM
# =========================


col1,col2 = st.columns(2)


with col1:

    st.subheader("RSI")

    st.line_chart(
        data["RSI"]
    )


with col2:

    st.subheader("Momentum")

    st.line_chart(
        data["Momentum"]
    )



# =========================
# AI STYLE SIGNAL
# =========================

current_price = float(
    data["Close"].iloc[-1]
)


dema = float(
    data["DEMA"].iloc[-1]
)


rsi = float(
    data["RSI"].iloc[-1]
)


if current_price > dema and rsi < 70:

    signal = "BUY 🟢"


elif current_price < dema:

    signal = "SELL 🔴"


else:

    signal = "HOLD 🟡"



st.subheader("🤖 AI Trading Signal")


c1,c2 = st.columns(2)


with c1:

    st.metric(
        "Current Price",
        f"₹ {current_price:.2f}"
    )


with c2:

    st.metric(
        "Signal",
        signal
    )



st.success(
    "🚀 AI Stock Dashboard Running Successfully"
)