import os
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# ================== CONFIG ==================
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Price Predictor App (Keras)")

# ✅ USE RAW STRING FOR WINDOWS PATHS
MODEL_PATH = r"C:\Users\thean\Desktop\DATA FOR MODEL TRAINING\Latest_stock_price_model.keras"
# If you saved as .h5 instead, point to that:
# MODEL_PATH = r"C:\Users\thean\Desktop\DATA FOR MODEL TRAINING\Latest_stock_price_model.h5"

# Sanity: show where we're running and confirm the file exists
st.sidebar.write("Working dir:", os.getcwd())
if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found:\n"
        f"{MODEL_PATH}\n\n"
        "Make sure you downloaded the Keras model from Colab (.keras/.h5) "
        "to exactly this path, or update MODEL_PATH above."
    )
    st.stop()

# Load model once
model = load_model(MODEL_PATH)

# ================== INPUTS ==================
stock = st.text_input("Enter Stock Symbol", "GOOG")

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# ================== DATA ==================
with st.spinner("Downloading data..."):
    data = yf.download(stock, start=start, end=end, auto_adjust=True, progress=False)

if data.empty:
    st.error("No data returned. Check the ticker symbol.")
    st.stop()

google_data = data.rename(columns=str.capitalize)  # 'Close', 'Volume', ...
st.subheader("Stock Data (last 10 rows)")
st.write(google_data.tail(10))

# ================== PLOTS ==================
def plot_graph(figsize, series_to_plot, full_df, extra_dataset=None, label_main="MA", label_extra=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(series_to_plot, label=label_main)
    plt.plot(full_df["Close"], label="Close")
    if extra_dataset is not None:
        plt.plot(extra_dataset, label=label_extra or "Extra")
    plt.legend()
    plt.grid(True)
    return fig

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data, None, "MA 250"))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'], google_data, None, "MA 200"))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, None, "MA 100"))

st.subheader('Close vs MA(100) and MA(250)')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data,
                     google_data['MA_for_250_days'], "MA 100", "MA 250"))

# ================== LSTM-STYLE PREP ==================
if "Close" not in google_data.columns:
    st.error("Expected 'Close' column not found in data.")
    st.stop()

splitting_len = int(len(google_data) * 0.7)
test_close = google_data[['Close']].iloc[splitting_len:].copy()

if len(test_close) < 150:
    st.warning("Not enough data after split for 100-step sequences. Try a longer date range or a different ticker.")
    st.stop()

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(test_close[['Close']])

# Build sequences of 100 timesteps
X_list, y_list = [], []
for i in range(100, len(scaled_data)):
    X_list.append(scaled_data[i-100:i])
    y_list.append(scaled_data[i])

x_data = np.array(X_list)  # (samples, 100, 1)
y_data = np.array(y_list)  # (samples, 1)

# ================== PREDICT ==================
with st.spinner("Running model predictions..."):
    preds_scaled = model.predict(x_data, verbose=0)

inv_preds = scaler.inverse_transform(preds_scaled)
inv_y     = scaler.inverse_transform(y_data)

# Align to original index
plot_index = google_data.index[splitting_len + 100:]

plot_df = pd.DataFrame({
    "original_test_data": inv_y.reshape(-1),
    "predictions": inv_preds.reshape(-1)
}, index=plot_index)

st.subheader("Original vs Predicted")
st.dataframe(plot_df.tail(10))

st.subheader('Original Close Price vs Predicted Close Price')
fig2 = plt.figure(figsize=(15,6))
# Original series
orig_series = pd.concat(
    [google_data['Close'][:splitting_len+100], plot_df['original_test_data']],
    axis=0
)
# Pred series aligned
pred_series = pd.concat(
    [pd.Series([np.nan] * (splitting_len+100), index=google_data.index[:splitting_len+100]),
     plot_df['predictions']],
    axis=0
)
plt.plot(orig_series, label="Original")
plt.plot(pred_series, label="Predicted")
plt.legend(); plt.grid(True)
st.pyplot(fig2)

st.success("Done. Reminder: predictions are statistical estimates, not guarantees.")
