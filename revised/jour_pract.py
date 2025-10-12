import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import re
import os
import xgboost as xgb
import plotly.graph_objects as go
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Custom Indian holiday calendar
class IndiaHolidayCalendar:
    @staticmethod
    def get_holidays():
        return ['2024-01-26', '2024-08-15', '2024-10-02', '2024-10-24', '2024-03-25']

# Load FinBERT sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

sentiment_pipeline = load_sentiment_model()

# Load Indian stock symbols
@st.cache_data
def get_indian_stocks():
    return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "WIPRO", "HINDUNILVR", "ITC", "BAJFINANCE"]

# Fetch stock data
def get_stock_data(ticker, start, end):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end)
        if not data.empty:
            data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
            data.index = data.index.tz_localize(None)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

# Fetch stock info
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        def format_value(value, format_str):
            if value == "N/A" or value is None:
                return "N/A"
            return format_str.format(value)
        
        return {
            "Market Cap": format_value(info.get("marketCap"), "{:,} INR"),
            "P/E Ratio": format_value(info.get("trailingPE"), "{:.2f}"),
            "Current Price": format_value(info.get("currentPrice"), "{:.2f} INR"),
            "Book Value": format_value(info.get("bookValue"), "{:.2f} INR"),
            "ROE": format_value(info.get("returnOnEquity"), "{:.2f}%"),
            "Dividend Yield": format_value(info.get("dividendYield"), "{:.2f}%"),
        }
    except:
        return {}

# News API
NEWS_API_KEY = "563215a35c1a47968f46271e04083ea3"
NEWS_API_URL = "https://newsapi.org/v2/everything"

def get_news(stock_symbol):
    stock_name_mapping = {
        "RELIANCE": "Reliance Industries",
        "TCS": "Tata Consultancy Services",
        "INFY": "Infosys",
        "HDFCBANK": "HDFC Bank",
        "ICICIBANK": "ICICI Bank",
        "SBIN": "State Bank of India",
        "WIPRO": "Wipro",
        "HINDUNILVR": "Hindustan Unilever",
        "ITC": "ITC Limited",
        "BAJFINANCE": "Bajaj Finance"
    }
    query = stock_name_mapping.get(stock_symbol, stock_symbol)
    params = {"q": query, "apiKey": NEWS_API_KEY, "language": "en", "sortBy": "publishedAt", "pageSize": 10}
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("articles", [])
        return []
    except:
        return []

# Sentiment analysis
def analyze_sentiment(text):
    if not text or pd.isna(text):
        return "neutral", 0.0
    try:
        result = sentiment_pipeline(text[:512])[0]
        return result['label'], result['score']
    except:
        return "neutral", 0.0

# Feature engineering
def create_advanced_features(df):
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['5D_MA'] = df['Close'].rolling(5).mean()
    df['20D_MA'] = df['Close'].rolling(20).mean()
    df['MA_Ratio'] = df['5D_MA'] / df['20D_MA']
    df['5D_Volatility'] = df['Returns'].rolling(5).std()
    df['Volume_MA5'] = df['Volume'].rolling(5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    return df.dropna()

# Dynamic Uncertainty-Weighted Fusion (Core Innovation from Paper)
class DynamicUncertaintyFusion:
    def __init__(self, lookback_window=10, temperature=1.0):
        self.lookback_window = lookback_window
        self.temperature = temperature
        self.uncertainty_history = {}
        
    def calculate_uncertainty(self, source_name, predictions, actuals):
        """Calculate time-varying uncertainty using recent prediction errors"""
        if len(predictions) < self.lookback_window:
            return 1.0  # High uncertainty if insufficient data
            
        errors = np.abs(np.array(predictions[-self.lookback_window:]) - 
                       np.array(actuals[-self.lookback_window:]))
        uncertainty = np.var(errors)  # Variance of recent errors
        return uncertainty + 1e-6  # Avoid division by zero
    
    def compute_dynamic_weights(self, uncertainties):
        """Compute Bayesian softmax weights based on uncertainties"""
        if not uncertainties:
            return {}
            
        # Convert to numpy array for numerical stability
        unc_array = np.array(list(uncertainties.values()))
        
        # Apply softmax on negative uncertainties (paper's formulation)
        weights = np.exp(-unc_array**2 / self.temperature)
        weights = weights / np.sum(weights)
        
        return dict(zip(uncertainties.keys(), weights))

# Create source-specific models
def create_source_specific_models():
    """Create expert models for different data sources as per paper"""
    
    # Market Data Model (GRU-based)
    market_model = Sequential([
        Input(shape=(1, 10)),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])
    market_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Sentiment Model (Simpler architecture)
    sentiment_model = Sequential([
        Input(shape=(1, 5)),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])
    sentiment_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return {
        'market': market_model,
        'sentiment': sentiment_model
    }

def create_hybrid_model_with_uncertainty(df_stock, sentiment_features):
    """Enhanced hybrid model with dynamic uncertainty-weighted fusion"""
    
    # Prepare data
    df_processed = create_advanced_features(df_stock)
    
    # Add sentiment features if available
    if sentiment_features:
        sentiment_df = pd.DataFrame(list(sentiment_features.items()), 
                                  columns=["Date", "Sentiment_Score"])
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        df_processed = df_processed.reset_index().merge(
            sentiment_df, left_on='Date', right_on='Date', how='left'
        ).set_index('Date')
        df_processed['Sentiment_Score'] = df_processed['Sentiment_Score'].fillna(0)
    else:
        df_processed['Sentiment_Score'] = 0
    
    # Create target (next day return)
    df_processed['Target'] = df_processed['Close'].pct_change().shift(-1)
    df_processed = df_processed.dropna()
    
    if len(df_processed) < 50:
        st.error("Insufficient data for model training")
        return None, None, None, None
    
    # Feature sets for different sources
    market_features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                      '5D_MA', 'MA_Ratio', '5D_Volatility', 'Volume_Ratio', 'Returns']
    
    sentiment_features_list = ['Sentiment_Score', 'Returns', '5D_Volatility']
    
    # Prepare data
    X_market = df_processed[market_features].values
    X_sentiment = df_processed[sentiment_features_list].values
    y = df_processed['Target'].values
    
    # Train-test split
    split_idx = int(0.8 * len(X_market))
    X_market_train, X_market_test = X_market[:split_idx], X_market[split_idx:]
    X_sentiment_train, X_sentiment_test = X_sentiment[:split_idx], X_sentiment[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    market_scaler = MinMaxScaler()
    sentiment_scaler = MinMaxScaler()
    
    X_market_train_scaled = market_scaler.fit_transform(X_market_train)
    X_market_test_scaled = market_scaler.transform(X_market_test)
    
    X_sentiment_train_scaled = sentiment_scaler.fit_transform(X_sentiment_train)
    X_sentiment_test_scaled = sentiment_scaler.transform(X_sentiment_test)
    
    # Reshape for GRU
    X_market_train_3d = X_market_train_scaled.reshape(X_market_train_scaled.shape[0], 1, X_market_train_scaled.shape[1])
    X_market_test_3d = X_market_test_scaled.reshape(X_market_test_scaled.shape[0], 1, X_market_test_scaled.shape[1])
    
    X_sentiment_train_3d = X_sentiment_train_scaled.reshape(X_sentiment_train_scaled.shape[0], 1, X_sentiment_train_scaled.shape[1])
    X_sentiment_test_3d = X_sentiment_test_scaled.reshape(X_sentiment_test_scaled.shape[0], 1, X_sentiment_test_scaled.shape[1])
    
    # Create and train models
    models = create_source_specific_models()
    
    # Train market model
    market_history = models['market'].fit(
        X_market_train_3d, y_train, 
        epochs=30, batch_size=16, verbose=0,
        validation_data=(X_market_test_3d, y_test)
    )
    
    # Train sentiment model
    sentiment_history = models['sentiment'].fit(
        X_sentiment_train_3d, y_train,
        epochs=30, batch_size=16, verbose=0,
        validation_data=(X_sentiment_test_3d, y_test)
    )
    
    # Get predictions
    market_pred = models['market'].predict(X_market_test_3d).flatten()
    sentiment_pred = models['sentiment'].predict(X_sentiment_test_3d).flatten()
    
    # Initialize dynamic fusion
    fusion_model = DynamicUncertaintyFusion(lookback_window=10)
    
    # Calculate uncertainties
    market_uncertainty = fusion_model.calculate_uncertainty(
        'market', market_pred.tolist(), y_test.tolist()
    )
    sentiment_uncertainty = fusion_model.calculate_uncertainty(
        'sentiment', sentiment_pred.tolist(), y_test.tolist()
    )
    
    uncertainties = {
        'market': market_uncertainty,
        'sentiment': sentiment_uncertainty
    }
    
    # Compute dynamic weights
    weights = fusion_model.compute_dynamic_weights(uncertainties)
    
    # Apply dynamic fusion
    if weights:
        final_pred = (weights['market'] * market_pred + 
                     weights['sentiment'] * sentiment_pred)
    else:
        final_pred = 0.5 * market_pred + 0.5 * sentiment_pred
    
    # Calculate metrics
    hybrid_mae = mean_absolute_error(y_test, final_pred)
    hybrid_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
    accuracy = max(0, 100 - (hybrid_mae * 100))
    
    model_metrics = {
        'market': {'mae': mean_absolute_error(y_test, market_pred), 'rmse': np.sqrt(mean_squared_error(y_test, market_pred))},
        'sentiment': {'mae': mean_absolute_error(y_test, sentiment_pred), 'rmse': np.sqrt(mean_squared_error(y_test, sentiment_pred))},
        'hybrid': {'mae': hybrid_mae, 'rmse': hybrid_rmse, 'accuracy': accuracy},
        'weights': weights,
        'uncertainties': uncertainties
    }
    
    # Store predictions
    df_processed.loc[y_test.index, 'Predicted_Return'] = final_pred
    
    return df_processed, models, model_metrics, [market_scaler, sentiment_scaler]

def generate_price_predictions(models, scalers, last_data, days=10):
    """Generate future price predictions"""
    try:
        predictions = []
        current_data = last_data.copy()
        
        for _ in range(days):
            # Prepare market features
            market_features = current_data[['Open', 'High', 'Low', 'Close', 'Volume', 
                                         '5D_MA', 'MA_Ratio', '5D_Volatility', 'Volume_Ratio', 'Returns']].iloc[-1:].values
            market_features_scaled = scalers[0].transform(market_features)
            market_features_3d = market_features_scaled.reshape(1, 1, market_features_scaled.shape[1])
            
            # Prepare sentiment features (using last available sentiment)
            sentiment_features = current_data[['Sentiment_Score', 'Returns', '5D_Volatility']].iloc[-1:].values
            sentiment_features_scaled = scalers[1].transform(sentiment_features)
            sentiment_features_3d = sentiment_features_scaled.reshape(1, 1, sentiment_features_scaled.shape[1])
            
            # Get predictions
            market_pred = models['market'].predict(market_features_3d)[0][0]
            sentiment_pred = models['sentiment'].predict(sentiment_features_3d)[0][0]
            
            # Simple average (can be enhanced with dynamic weights)
            final_pred = 0.6 * market_pred + 0.4 * sentiment_pred
            
            # Calculate new price
            last_close = current_data['Close'].iloc[-1]
            new_price = last_close * (1 + final_pred)
            
            predictions.append(new_price)
            
            # Update data for next prediction (simplified)
            new_row = current_data.iloc[-1:].copy()
            new_row['Close'] = new_price
            new_row['Open'] = new_price * 0.99
            new_row['High'] = new_price * 1.01
            new_row['Low'] = new_price * 0.99
            current_data = pd.concat([current_data, new_row])
        
        return predictions
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return []

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Stock Analysis", layout="wide")
    st.title("📈 Indian Stock Analysis with Dynamic Uncertainty-Weighted Fusion")
    st.markdown("**Bayesian Framework for Time-Varying Source Reliability**")
    
    # Sidebar
    st.sidebar.header("Stock Selection")
    indian_stocks = get_indian_stocks()
    selected_stock = st.sidebar.selectbox("Select Stock", indian_stocks)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.date(2023, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.date.today())
    
    if st.sidebar.button("Analyze Stock", type="primary"):
        analyze_stock(selected_stock, start_date, end_date)

def analyze_stock(stock_symbol, start_date, end_date):
    ticker = f"{stock_symbol}.NS"
    
    with st.spinner("Fetching stock data..."):
        df_stock = get_stock_data(ticker, start_date, end_date)
    
    if df_stock.empty:
        st.error("No data available for the selected stock and date range.")
        return
    
    # Display basic info
    st.subheader(f"📊 Stock Information: {stock_symbol}")
    stock_info = get_stock_info(ticker)
    
    if stock_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", stock_info["Current Price"])
            st.metric("P/E Ratio", stock_info["P/E Ratio"])
        with col2:
            st.metric("Market Cap", stock_info["Market Cap"])
            st.metric("Book Value", stock_info["Book Value"])
        with col3:
            st.metric("ROE", stock_info["ROE"])
            st.metric("Dividend Yield", stock_info["Dividend Yield"])
    
    # Price chart
    st.subheader("📈 Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_stock.index,
        open=df_stock['Open'],
        high=df_stock['High'],
        low=df_stock['Low'],
        close=df_stock['Close'],
        name='Price'
    ))
    fig.update_layout(title=f"{stock_symbol} Price Movement", xaxis_title="Date", yaxis_title="Price (INR)")
    st.plotly_chart(fig, use_container_width=True)
    
    # News and sentiment analysis
    st.subheader("📰 News Sentiment Analysis")
    with st.spinner("Fetching news..."):
        news_articles = get_news(stock_symbol)
    
    daily_sentiment = {}
    if news_articles:
        for article in news_articles:
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title} {description}".strip()
            
            sentiment, confidence = analyze_sentiment(text)
            date = article.get("publishedAt", "")[:10]
            
            if date not in daily_sentiment:
                daily_sentiment[date] = []
            daily_sentiment[date].append((sentiment, confidence))
            
            with st.expander(f"{title}"):
                st.write(f"**Description:** {description}")
                st.write(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")
                st.write(f"**Source:** {article.get('source', {}).get('name', 'Unknown')}")
                st.write(f"**Published:** {article.get('publishedAt', '')}")
    
    # Calculate average daily sentiment
    avg_sentiment_scores = {}
    for date, sentiments in daily_sentiment.items():
        weighted_sum = 0
        total_weight = 0
        for sentiment, confidence in sentiments:
            value = 1 if sentiment == "positive" else (-1 if sentiment == "negative" else 0)
            weighted_sum += value * confidence
            total_weight += confidence
        avg_sentiment_scores[date] = weighted_sum / total_weight if total_weight > 0 else 0
    
    # AI Analysis with Dynamic Uncertainty Fusion
    st.subheader("🤖 AI Stock Analysis with Dynamic Uncertainty-Weighted Fusion")
    
    with st.spinner("Training AI models with dynamic fusion..."):
        result = create_hybrid_model_with_uncertainty(df_stock, avg_sentiment_scores)
    
    if result is None:
        return
        
    df_processed, models, metrics, scalers = result
    
    # Display model performance
    st.subheader("📊 Model Performance with Dynamic Fusion")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Model MAE", f"{metrics['market']['mae']:.4f}")
    with col2:
        st.metric("Sentiment Model MAE", f"{metrics['sentiment']['mae']:.4f}")
    with col3:
        st.metric("Hybrid Model Accuracy", f"{metrics['hybrid']['accuracy']:.1f}%")
    
    # Display dynamic weights and uncertainties
    st.subheader("⚖️ Dynamic Fusion Weights & Uncertainties")
    if metrics.get('weights'):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Source Weights:**")
            for source, weight in metrics['weights'].items():
                st.write(f"- {source.capitalize()}: {weight:.2%}")
        with col2:
            st.write("**Source Uncertainties:**")
            for source, uncertainty in metrics['uncertainties'].items():
                st.write(f"- {source.capitalize()}: {uncertainty:.4f}")
    
    # Generate predictions
    st.subheader("🔮 10-Day Price Forecast")
    last_30_days = df_processed.iloc[-30:].copy()
    
    with st.spinner("Generating price predictions..."):
        future_prices = generate_price_predictions(models, scalers, last_30_days, days=10)
    
    if future_prices:
        future_dates = pd.date_range(start=df_processed.index[-1] + pd.Timedelta(days=1), periods=10, freq='D')
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_prices
        })
        forecast_df['Daily_Change'] = forecast_df['Predicted_Price'].pct_change() * 100
        
        # Display forecast table
        st.dataframe(forecast_df.style.format({
            'Predicted_Price': '₹{:.2f}',
            'Daily_Change': '{:+.2f}%'
        }).applymap(lambda x: 'color: green' if x > 0 else 'color: red', subset=['Daily_Change']))
        
        # Plot forecast
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=df_processed.index[-30:],
            y=df_processed['Close'][-30:],
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue', width=2)
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Predicted_Price'],
            mode='lines+markers',
            name='AI Forecast',
            line=dict(color='red', width=2, dash='dot')
        ))
        fig_forecast.update_layout(
            title=f"{stock_symbol} Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (INR)"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Investment recommendation
        current_price = df_processed['Close'].iloc[-1]
        avg_predicted = np.mean(future_prices)
        price_change_pct = (avg_predicted - current_price) / current_price * 100
        
        st.subheader("💡 Investment Recommendation")
        
        if price_change_pct > 5 and metrics['hybrid']['accuracy'] > 70:
            recommendation = "STRONG BUY 🟢"
            reasoning = "High confidence in significant price appreciation"
        elif price_change_pct > 2 and metrics['hybrid']['accuracy'] > 65:
            recommendation = "BUY 🟢"
            reasoning = "Good confidence in moderate growth"
        elif price_change_pct > -2:
            recommendation = "HOLD 🟡"
            reasoning = "Neutral outlook with minimal expected movement"
        elif price_change_pct > -5:
            recommendation = "HOLD (Caution) 🟡"
            reasoning = "Potential for slight decline"
        else:
            recommendation = "SELL 🔴"
            reasoning = "Expected price decline"
        
        st.info(f"**{recommendation}** - {reasoning}")
        st.write(f"Expected price change: {price_change_pct:+.1f}%")
        st.write(f"Model confidence: {metrics['hybrid']['accuracy']:.1f}%")

if __name__ == "__main__":
    main()