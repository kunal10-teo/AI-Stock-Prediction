import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import pandas_ta as ta

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

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
        "HDFCBANK.NS"
    ]
)


# =========================
# FETCH DATA
# =========================

data = yf.download(
    stock,
    period="1y",
    auto_adjust=False
)


if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)


data.dropna(inplace=True)



# =========================
# INDICATORS (Without pandas_ta)
# =========================

# DEMA
ema1 = data["Close"].ewm(span=20, adjust=False).mean()
ema2 = ema1.ewm(span=20, adjust=False).mean()
data["DEMA"] = 2 * ema1 - ema2

# Momentum
data["Momentum"] = data["Close"] - data["Close"].shift(10)

# Bollinger Bands
rolling_mean = data["Close"].rolling(20).mean()
rolling_std = data["Close"].rolling(20).std()

data["BB_Upper"] = rolling_mean + (2 * rolling_std)
data["BB_Lower"] = rolling_mean - (2 * rolling_std)

# RSI
delta = data["Close"].diff()

gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

rs = gain / loss
data["RSI"] = 100 - (100 / (1 + rs))

data.dropna(inplace=True)


# =========================
# DATA TABLE
# =========================

st.subheader("📋 Recent Stock Data")

st.dataframe(
    data.tail()
)



# =========================
# PLOTLY CHART
# =========================

st.subheader("📈 Stock Analysis Chart")


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
        name="BB Upper"
    )
)


fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["BB_Lower"],
        name="BB Lower"
    )
)



fig.update_layout(
    height=500,
    title="Price + Bollinger Bands + DEMA"
)


st.plotly_chart(
    fig,
    use_container_width=True
)



# =========================
# RSI & MOMENTUM
# =========================


col1,col2 = st.columns(2)


with col1:

    st.subheader("📊 RSI")

    st.line_chart(
        data["RSI"]
    )


with col2:

    st.subheader("⚡ Momentum")

    st.line_chart(
        data["Momentum"]
    )



# =========================
# AI PREDICTION
# =========================

st.subheader("🤖 AI Prediction")


MODEL_PATH = os.path.join(
    os.getcwd(),
    "models",
    "real_lstm_model.keras"
)



if os.path.exists(MODEL_PATH):

    model = load_model(
        MODEL_PATH
    )


    close_data = data[["Close"]].values


    scaler = MinMaxScaler()


    scaled = scaler.fit_transform(
        close_data
    )


    last_60 = scaled[-60:]


    X_test = np.reshape(
        last_60,
        (1,60,1)
    )


    prediction = model.predict(
        X_test
    )


    predicted_price = scaler.inverse_transform(
        prediction
    )


    current_price = float(
        data["Close"].iloc[-1]
    )


    predicted = float(
        predicted_price[0][0]
    )



    c1,c2,c3 = st.columns(3)



    with c1:

        st.metric(
            "Current Price",
            f"₹ {current_price:.2f}"
        )


    with c2:

        st.metric(
            "Predicted Price",
            f"₹ {predicted:.2f}"
        )


    # =========================
    # SIGNAL ENGINE
    # =========================


    rsi_value = float(
        data["RSI"].iloc[-1]
    )


    momentum_value = float(
        data["Momentum"].iloc[-1]
    )



    if (
        predicted > current_price
        and momentum_value > 0
        and rsi_value < 70
    ):

        signal="BUY 🟢"


    elif (
        predicted < current_price
        and momentum_value < 0
        and rsi_value < 60
    ):

        signal="SELL 🔴"


    else:

        signal="HOLD 🟡"



    with c3:

        st.metric(
            "Signal",
            signal
        )



    # Confidence

    confidence = min(
        round(
            (abs(predicted-current_price)/current_price)*100+70,
            2
        ),
        95
    )


    st.success(
        f"AI Trading Signal: {signal}"
    )


    st.info(
        f"AI Confidence: {confidence}%"
    )


else:

    st.error(
        "❌ Model Not Found"
    )



st.success(
    "🚀 AI Dashboard Running Successfully"
)