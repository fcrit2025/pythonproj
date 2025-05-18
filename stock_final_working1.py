import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import datetime
import re
import os
import numpy as np
import xgboost as xgb
from transformers import pipeline
from sklearn.model_selection import train_test_split
import shap

# Load Indian stock symbols from a local CSV file
@st.cache_data
def get_indian_stocks():
    file_path = "indian_stocks.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding="utf-8")  # Ensure UTF-8 encoding
        df.columns = df.columns.str.strip()  # Remove any spaces in column names
        
        if "SYMBOL" in df.columns:
            return df["SYMBOL"].dropna().tolist()  # Drop empty rows if any
        else:
            st.error("Error: 'SYMBOL' column not found. Check CSV file.")
            return []
    else:
        st.error("Error: 'indian_stocks.csv' file not found.")
        return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

# Function to fetch stock data
def get_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    return stock.history(start=start, end=end)


# Function to fetch news
NEWS_API_KEY = "563215a35c1a47968f46271e04083ea3" # 563215a35c1a47968f46271e04083ea3   eb59edfd3de9450f9b11df2e69591e30
NEWS_API_URL = "https://newsapi.org/v2/everything"

def get_news(stock_symbol):
    stock_name_mapping = {
        "RELIANCE": "Reliance Industries",
        "TCS": "Tata Consultancy Services",
        "INFY": "Infosys",
        "HDFCBANK": "HDFC Bank",
        "ICICIBANK": "ICICI Bank"
    }
    query = stock_name_mapping.get(stock_symbol, stock_symbol)
    params = {"q": query, "apiKey": NEWS_API_KEY, "language": "en", "sortBy": "publishedAt"}
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching news: {response.json()}")
        return []
    return response.json().get("articles", [])

# Load FinBERT sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(text):
    result = sentiment_pipeline(text[:512])[0]  # Limiting text length for processing
    return result['label'], result['score']

def filter_relevant_news(news_articles, stock_name):
    filtered_articles = []
    for article in news_articles:
        title = article.get('title', '')  # Default to empty string if None
        if title and re.search(stock_name, title, re.IGNORECASE):  
            filtered_articles.append(article)
    return filtered_articles


# Function to train XGBoost model for stock prediction
def train_xgboost_model(df_stock):
    df_stock['Returns'] = df_stock['Close'].pct_change()
    df_stock.dropna(inplace=True)
    
    df_stock['Target'] = df_stock['Returns'].shift(-1)
    df_stock.dropna(inplace=True)
    
    X = df_stock[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df_stock['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    df_stock.loc[y_test.index, 'Predicted Close'] = predictions
    
    # SHAP Explainability
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    return df_stock, model, shap_values, X.columns

# Function to predict future prices
def predict_future_prices(model, last_known_data, days=30):
    future_dates = pd.date_range(start=last_known_data.index[-1], periods=days + 1, freq='D')[1:]
    future_data = pd.DataFrame(index=future_dates, columns=['Predicted Close'])
    
    last_row = last_known_data.iloc[-1]
    for date in future_dates:
        features = np.array([last_row[['Open', 'High', 'Low', 'Close', 'Volume']]])
        future_data.loc[date, 'Predicted Close'] = model.predict(features)[0]
    
    return future_data

# Streamlit UI
st.title("Indian Stock Market Analysis")
st.sidebar.header("Stock Selection")
indian_stocks = get_indian_stocks()
selected_stock = st.sidebar.selectbox("Select Stock", indian_stocks)
start_date = st.sidebar.date_input("Select Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("Select End Date", datetime.date.today())


# Fetch and display stock data
if st.sidebar.button("Fetch Data"):
    df_stock = get_stock_data(f"{selected_stock}.NS", start_date, end_date)
    
    if not df_stock.empty:
        st.subheader(f"Stock Price Trend for {selected_stock}")
        st.line_chart(df_stock["Close"])  # Show only historical close prices

        # Train the XGBoost model and get SHAP values
        df_stock, model, shap_values, feature_names = train_xgboost_model(df_stock)

        # Fetch and display news
        st.subheader(f"Latest News for {selected_stock}")
        news_articles = get_news(selected_stock)
        filtered_news = filter_relevant_news(news_articles, selected_stock)
        sentiment_data = []
        daily_sentiment = {}

        for article in filtered_news:
            sentiment, confidence_score = analyze_sentiment(article["title"] + " " + article.get("description", ""))
            date = article.get("publishedAt", "")[0:10]
            sentiment_data.append([article['title'], confidence_score, sentiment])

            if date in daily_sentiment:
                daily_sentiment[date].append((sentiment, confidence_score))
            else:
                daily_sentiment[date] = [(sentiment, confidence_score)]

            st.write(f"**{article['title']}**")
            st.write(article["description"])
            st.write(f"[Read more]({article['url']})")
            st.write("---")

        # Display table of sentiment analysis
        st.subheader("News Sentiment Analysis")
        df_sentiment = pd.DataFrame(sentiment_data, columns=["Headline", "Confidence Score", "Sentiment"])
        st.table(df_sentiment)

        # Compute average sentiment per day
        avg_daily_sentiment = {date: max(set([s for s, _ in scores]), key=[s for s, _ in scores].count) for date, scores in daily_sentiment.items()}
        df_avg_sentiment = pd.DataFrame(avg_daily_sentiment.items(), columns=["Date", "Average Sentiment"])

        # Compute weighted average sentiment per day
        weighted_sentiment = {}
        for date, scores in daily_sentiment.items():
            total_weight = sum(abs(score) for _, score in scores)
            weighted_avg = sum(score if sentiment == "positive" else -score for sentiment, score in scores) / total_weight if total_weight else 0
            weighted_sentiment[date] = weighted_avg

        df_weighted_sentiment = pd.DataFrame(weighted_sentiment.items(), columns=["Date", "Weighted Sentiment"])

        st.subheader("Daily Average Sentiment")
        st.table(df_avg_sentiment)

        # Generate Explainable AI Conclusion
        st.subheader("Explainable AI Conclusion")

        last_shap_values = shap_values.values[-1]  # Get last prediction explanation
        feature_importance = dict(zip(feature_names, last_shap_values))

        most_influential_factor = max(feature_importance, key=feature_importance.get)

        sentiment_impact = "positive" if df_weighted_sentiment["Weighted Sentiment"].iloc[-1] > 0 else "negative"

        conclusion = f"""
        The stock price prediction for {selected_stock} is primarily influenced by **{most_influential_factor}**.
        The weighted sentiment for the latest available data is **{sentiment_impact}**, which suggests that sentiment has a **{sentiment_impact} impact** on stock movement.
        """

        st.write(conclusion)
