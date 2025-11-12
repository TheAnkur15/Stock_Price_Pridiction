# 📈 Stock Price Predictor App (Keras + Streamlit)

## 🧠 Overview
This project is a **deep learning–based stock market analyzer and predictor** built using **TensorFlow Keras** and deployed through **Streamlit**.  
It visualizes stock performance, moving averages, and uses an **LSTM model** trained on historical closing prices to forecast future price trends.

---

## ⚙️ Tech Stack

| Category | Tools Used |
|-----------|-------------|
| Programming Language | Python 3.10+ |
| Framework | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib |
| Data Source | Yahoo Finance (`yfinance` API) |
| Scaling | Scikit-learn `MinMaxScaler` |

---

## 🧩 Features
- ✅ Interactive web interface using Streamlit  
- ✅ Real-time stock data fetched from Yahoo Finance  
- ✅ Visualizes multiple moving averages (100, 200, 250 days)  
- ✅ LSTM-based prediction of closing prices  
- ✅ Displays actual vs predicted data comparison  
- ✅ Easy to extend to any stock symbol (e.g., AAPL, GOOG, TSLA)  

---

## 📦 Folder Structure

STOCK_MARKET/
│
├── web_stock_price_pridicter.py       # Streamlit app
├── Latest_stock_price_model.keras     # Trained model
├── README.md                          # Documentation
├── requirements.txt                   # Dependencies
├── .venv/                             # (Virtual environment)
└── .ipynb_checkpoints/                # (Ignore)



---

## 🧠 How It Works
1. Fetches 20 years of stock data from Yahoo Finance.  
2. Computes technical indicators — moving averages of 100, 200, and 250 days.  
3. Scales the closing price values between 0 and 1 for training consistency.  
4. Uses a pre-trained **LSTM neural network model** (saved as `.keras`) to predict future prices.  
5. Inverse-transforms predictions to get real-world price estimates.  
6. Displays original vs predicted prices interactively.

---
## 🚀 Running the App

### 🧩 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt

⚙️ 2️⃣ Run Streamlit

streamlit run web_stock_price_pridicter.py
🌐 3️⃣ Open in Browser

By default, Streamlit will open the app at:
👉 http://localhost:8501

💾 Model Information

The model (Latest_stock_price_model.keras) was trained in Google Colab on closing prices using LSTM layers.
After training, it was saved using:

model.save("Latest_stock_price_model.keras")

📊 Example Output

Moving Average (100, 200, 250 days) visualizations

Predicted vs Actual Closing Price comparison

Interactive DataFrame displaying recent test data and prediction values

🧰 Future Improvements

Integrate multi-feature LSTM using Open, High, Low, and Volume data

Add sentiment analysis using financial news headlines

Implement auto-refresh for live market updates

Deploy on Streamlit Cloud or Render for public access

👨‍💻 Project Info

Project: Stock Price Predictor (Keras + Streamlit)
Version: 1.0
License: MIT

✅ Reviewer Note

This project includes:

Functional Streamlit app

Pre-trained .keras model

Clean and documented folder structure

This README for proper project documentation



