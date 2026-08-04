import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
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
# INDICATORS
# =========================

data["DEMA"] = ta.dema(
    data["Close"],
    length=20
)


data["Momentum"] = ta.mom(
    data["Close"],
    length=10
)


bb = ta.bbands(
    data["Close"],
    length=20
)


if bb is not None:
    data["BB_Upper"] = bb.iloc[:,0]
    data["BB_Lower"] = bb.iloc[:,2]

else:
    data["BB_Upper"] = data["Close"]
    data["BB_Lower"] = data["Close"]



data["RSI"] = ta.rsi(
    data["Close"],
    length=14
)


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