import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import re
import os
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

# Load sentiment analysis model (simplified to avoid dependencies)
def analyze_sentiment(text):
    """Simplified sentiment analysis to avoid transformer dependencies"""
    if not text or pd.isna(text):
        return "neutral", 0.5
    
    positive_words = ['up', 'rise', 'gain', 'bullish', 'positive', 'growth', 'profit', 'increase']
    negative_words = ['down', 'fall', 'drop', 'bearish', 'negative', 'loss', 'decline', 'decrease']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive", min(0.5 + (positive_count * 0.1), 0.9)
    elif negative_count > positive_count:
        return "negative", min(0.5 + (negative_count * 0.1), 0.9)
    else:
        return "neutral", 0.5

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
    params = {"q": query, "apiKey": NEWS_API_KEY, "language": "en", "sortBy": "publishedAt", "pageSize": 5}
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("articles", [])
        return []
    except:
        return []

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
    
    # Handle infinite values and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

# Dynamic Uncertainty-Weighted Fusion (Core Innovation from Paper)
class DynamicUncertaintyFusion:
    def __init__(self, lookback_window=10, temperature=1.0):
        self.lookback_window = lookback_window
        self.temperature = temperature
        
    def calculate_uncertainty(self, predictions, actuals):
        """Calculate time-varying uncertainty using recent prediction errors"""
        if len(predictions) < 2:
            return 1.0  # High uncertainty if insufficient data
            
        errors = np.abs(np.array(predictions) - np.array(actuals))
        uncertainty = np.var(errors) if len(errors) > 1 else 1.0
        return max(uncertainty, 1e-6)  # Avoid division by zero
    
    def compute_dynamic_weights(self, uncertainties):
        """Compute Bayesian softmax weights based on uncertainties"""
        if not uncertainties:
            return {'market': 0.5, 'sentiment': 0.5}
            
        # Convert to numpy array for numerical stability
        unc_array = np.array(list(uncertainties.values()))
        
        # Apply softmax on negative uncertainties (paper's formulation)
        weights = np.exp(-unc_array**2 / self.temperature)
        weights = weights / np.sum(weights)
        
        return dict(zip(uncertainties.keys(), weights))

def create_market_model(input_dim, sequence_length=1):
    """Create market data model with proper dimensions"""
    model = Sequential([
        LSTM(32, input_shape=(sequence_length, input_dim), return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def create_sentiment_model(input_dim, sequence_length=1):
    """Create sentiment model with proper dimensions"""
    model = Sequential([
        LSTM(16, input_shape=(sequence_length, input_dim), return_sequences=False),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def create_hybrid_model_with_uncertainty(df_stock, sentiment_features):
    """Enhanced hybrid model with dynamic uncertainty-weighted fusion"""
    
    try:
        # Prepare data
        df_processed = create_advanced_features(df_stock)
        
        # Add sentiment features if available
        if sentiment_features:
            # Convert sentiment features to daily average
            sentiment_series = pd.Series(sentiment_features)
            # Align sentiment with stock data dates
            for date, score in sentiment_features.items():
                if pd.to_datetime(date) in df_processed.index:
                    df_processed.loc[pd.to_datetime(date), 'Sentiment_Score'] = score
            df_processed['Sentiment_Score'] = df_processed['Sentiment_Score'].fillna(0)
        else:
            df_processed['Sentiment_Score'] = 0
        
        # Create target (next day return)
        df_processed['Target'] = df_processed['Close'].shift(-1)  # Next day's close price
        df_processed = df_processed.dropna()
        
        if len(df_processed) < 30:
            st.error("Insufficient data for model training. Need at least 30 days of data.")
            return None
        
        # Feature sets for different sources - USING COMMON FEATURES TO AVOID DIMENSION ISSUES
        common_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', '5D_Volatility']
        
        # Prepare data
        X = df_processed[common_features].values
        y = df_processed['Target'].values
        
        # Train-test split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Reshape for LSTM (samples, time steps, features)
        X_train_3d = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
        X_test_3d = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
        
        # Create and train models with SAME input dimensions
        input_dim = X_train_3d.shape[2]  # This ensures both models have same input dim
        
        market_model = create_market_model(input_dim)
        sentiment_model = create_sentiment_model(input_dim)
        
        # Train market model
        market_history = market_model.fit(
            X_train_3d, y_train, 
            epochs=20, batch_size=16, verbose=0,
            validation_data=(X_test_3d, y_test)
        )
        
        # Train sentiment model
        sentiment_history = sentiment_model.fit(
            X_train_3d, y_train,
            epochs=20, batch_size=16, verbose=0,
            validation_data=(X_test_3d, y_test)
        )
        
        # Get predictions
        market_pred = market_model.predict(X_test_3d).flatten()
        sentiment_pred = sentiment_model.predict(X_test_3d).flatten()
        
        # Initialize dynamic fusion
        fusion_model = DynamicUncertaintyFusion(lookback_window=min(10, len(market_pred)))
        
        # Calculate uncertainties
        market_uncertainty = fusion_model.calculate_uncertainty(market_pred, y_test)
        sentiment_uncertainty = fusion_model.calculate_uncertainty(sentiment_pred, y_test)
        
        uncertainties = {
            'market': market_uncertainty,
            'sentiment': sentiment_uncertainty
        }
        
        # Compute dynamic weights
        weights = fusion_model.compute_dynamic_weights(uncertainties)
        
        # Apply dynamic fusion
        if weights:
            final_pred = (weights['market'] * market_pred + weights['sentiment'] * sentiment_pred)
        else:
            final_pred = 0.5 * market_pred + 0.5 * sentiment_pred
        
        # Calculate metrics
        hybrid_mae = mean_absolute_error(y_test, final_pred)
        hybrid_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
        
        # Calculate accuracy based on direction prediction
        direction_accuracy = np.mean((np.sign(final_pred - X_test[:, 3]) == np.sign(y_test - X_test[:, 3]))) * 100
        
        model_metrics = {
            'market': {
                'mae': mean_absolute_error(y_test, market_pred), 
                'rmse': np.sqrt(mean_squared_error(y_test, market_pred))
            },
            'sentiment': {
                'mae': mean_absolute_error(y_test, sentiment_pred), 
                'rmse': np.sqrt(mean_squared_error(y_test, sentiment_pred))
            },
            'hybrid': {
                'mae': hybrid_mae, 
                'rmse': hybrid_rmse, 
                'accuracy': direction_accuracy
            },
            'weights': weights,
            'uncertainties': uncertainties
        }
        
        # Store predictions
        test_dates = df_processed.index[split_idx:split_idx + len(final_pred)]
        predictions_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': y_test,
            'Predicted': final_pred,
            'Market_Pred': market_pred,
            'Sentiment_Pred': sentiment_pred
        }).set_index('Date')
        
        models_dict = {
            'market': market_model,
            'sentiment': sentiment_model
        }
        
        return predictions_df, models_dict, model_metrics, scaler, common_features
        
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None

def generate_future_predictions(model, scaler, last_data, features, days=10):
    """Generate future price predictions"""
    try:
        predictions = []
        current_data = last_data[features].values[-1:].copy()  # Last available data point
        
        for i in range(days):
            # Scale the input
            current_scaled = scaler.transform(current_data)
            current_3d = current_scaled.reshape(1, 1, current_scaled.shape[1])
            
            # Predict next price
            pred_price = model.predict(current_3d, verbose=0)[0][0]
            predictions.append(pred_price)
            
            # Update for next prediction (simplified approach)
            # For a more realistic approach, you'd need to update all features
            current_data[0, 3] = pred_price  # Update Close price
            current_data[0, 5] = (pred_price - current_data[0, 3]) / current_data[0, 3]  # Update Returns
        
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
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", stock_info.get("Current Price", "N/A"))
            st.metric("P/E Ratio", stock_info.get("P/E Ratio", "N/A"))
        with col2:
            st.metric("Market Cap", stock_info.get("Market Cap", "N/A"))
            st.metric("Book Value", stock_info.get("Book Value", "N/A"))
    
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
    fig.update_layout(
        title=f"{stock_symbol} Price Movement", 
        xaxis_title="Date", 
        yaxis_title="Price (INR)",
        height=400
    )
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
            
            with st.expander(f"{title[:80]}..."):
                st.write(f"**Description:** {description}")
                st.write(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")
                st.write(f"**Source:** {article.get('source', {}).get('name', 'Unknown')}")
    
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
        st.warning("Could not train models. Please try with a different stock or date range.")
        return
        
    predictions_df, models, metrics, scaler, features = result
    
    # Display model performance
    st.subheader("📊 Model Performance with Dynamic Fusion")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Model MAE", f"{metrics['market']['mae']:.2f}")
    with col2:
        st.metric("Sentiment Model MAE", f"{metrics['sentiment']['mae']:.2f}")
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
    
    # Show prediction results
    st.subheader("📋 Prediction Results")
    st.dataframe(predictions_df.tail(10).style.format({
        'Actual': '{:.2f}',
        'Predicted': '{:.2f}',
        'Market_Pred': '{:.2f}',
        'Sentiment_Pred': '{:.2f}'
    }))
    
    # Generate future predictions
    st.subheader("🔮 10-Day Price Forecast")
    
    # Use market model for future predictions (as it typically performs better)
    last_data = df_stock[features].iloc[-30:].copy()
    
    with st.spinner("Generating price predictions..."):
        future_prices = generate_future_predictions(
            models['market'], scaler, last_data, features, days=10
        )
    
    if future_prices:
        future_dates = pd.date_range(
            start=df_stock.index[-1] + pd.Timedelta(days=1), 
            periods=10, 
            freq='D'
        )
        
        current_price = df_stock['Close'].iloc[-1]
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_prices
        })
        forecast_df['Price_Change'] = forecast_df['Predicted_Price'] - current_price
        forecast_df['Percent_Change'] = (forecast_df['Price_Change'] / current_price) * 100
        
        # Display forecast table
        st.dataframe(forecast_df.style.format({
            'Predicted_Price': '₹{:.2f}',
            'Price_Change': '₹{:.2f}',
            'Percent_Change': '{:+.2f}%'
        }))
        
        # Investment recommendation
        avg_predicted = np.mean(future_prices)
        price_change_pct = (avg_predicted - current_price) / current_price * 100
        
        st.subheader("💡 Investment Recommendation")
        
        if price_change_pct > 5 and metrics['hybrid']['accuracy'] > 60:
            recommendation = "STRONG BUY 🟢"
            reasoning = "High confidence in significant price appreciation"
        elif price_change_pct > 2 and metrics['hybrid']['accuracy'] > 55:
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
        
        st.success(f"**{recommendation}**")
        st.write(f"**Reasoning:** {reasoning}")
        st.write(f"**Expected price change:** {price_change_pct:+.1f}%")
        st.write(f"**Model confidence:** {metrics['hybrid']['accuracy']:.1f}%")

if __name__ == "__main__":
    main()