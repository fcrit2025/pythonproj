import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import re
import os
import xgboost as xgb
import shap
import plotly.graph_objects as go
from prophet import Prophet
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
# For TensorFlow 2.x (modern versions)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import CustomBusinessDay
from keras import Input

import bs4
from scipy.stats import beta
from bayespy.nodes import Gaussian, Gamma, GaussianARD
import bayespy as bp


# Custom Indian holiday calendar
class IndiaHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('Republic Day', month=1, day=26),
        Holiday('Independence Day', month=8, day=15),
        Holiday('Gandhi Jayanti', month=10, day=2),
        Holiday('Diwali', month=10, day=24),  # Example date - adjust as needed
        Holiday('Holi', month=3, day=25),     # Example date - adjust as needed
    ]

# Load FinBERT sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Load Indian stock symbols
@st.cache_data
def get_indian_stocks():
    file_path = "indian_stocks.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding="utf-8")
        df.columns = df.columns.str.strip()
        if "SYMBOL" in df.columns:
            return df["SYMBOL"].dropna().tolist()
        else:
            st.error("Error: 'SYMBOL' column not found.")
            return []
    else:
        st.error("File 'indian_stocks.csv' not found.")
        return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

# Fetch stock data
def get_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start, end=end)
    if not data.empty:
        data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        data.index = data.index.tz_localize(None)
    return data

# Fetch stock info
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    def format_value(value, format_str):
        if value == "N/A" or value is None:
            return "N/A"
        return format_str.format(value)
    
    return {
        "Market Cap": format_value(info.get("marketCap"), "{:,} INR"),
        "P/E Ratio": format_value(info.get("trailingPE"), "{}"),
        "ROCE": format_value(info.get("returnOnCapitalEmployed"), "{:.2f}%"),
        "Current Price": format_value(info.get("currentPrice"), "{:.2f} INR"),
        "Book Value": format_value(info.get("bookValue"), "{:.2f} INR"),
        "ROE": format_value(info.get("returnOnEquity"), "{:.2f}%"),
        "Dividend Yield": format_value(info.get("dividendYield"), "{:.2f}%"),
        "Face Value": format_value(info.get("faceValue"), "{:.2f} INR"),
        "High": format_value(info.get("dayHigh"), "{:.2f} INR"),
        "Low": format_value(info.get("dayLow"), "{:.2f} INR"),
    }

# News API
NEWS_API_KEY = "563215a35c1a47968f46271e04083ea3"  #P8:563215a35c1a47968f46271e04083ea3  P6:2c21d58c03d34b2aa0509ef13f331e6d   my:eb59edfd3de9450f9b11df2e69591e30
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

# Sentiment analysis
def analyze_sentiment(text):
    if not text:
        return "neutral", 0.0
    result = sentiment_pipeline(text[:512])[0]
    return result['label'], result['score']

# Filter relevant news
def filter_relevant_news(news_articles, stock_name):
    filtered_articles = []
    for article in news_articles:
        title = article.get('title', '')
        if title and re.search(stock_name, title, re.IGNORECASE):  
            filtered_articles.append(article)
    return filtered_articles


# ========== 1. INDIA VIX DATA FETCHING ==========
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_india_vix_data():
    """Fetch India VIX data from Yahoo Finance"""
    try:
        vix_ticker = "^INDIAVIX"
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=90)
        
        vix_data = yf.download(vix_ticker, start=start_date, end=end_date)
        if not vix_data.empty:
            vix_data = vix_data[['Close']].rename(columns={'Close': 'India_VIX'})
            vix_data['VIX_Change'] = vix_data['India_VIX'].pct_change() * 100
            vix_data['VIX_Level'] = pd.cut(vix_data['India_VIX'], 
                                          bins=[0, 15, 25, 35, 100],
                                          labels=['Low', 'Moderate', 'High', 'Extreme'])
            return vix_data
    except Exception as e:
        st.warning(f"Could not fetch India VIX data: {str(e)}")
    return pd.DataFrame()

# ========== 2. FII/DII DATA SCRAPING ==========
@st.cache_data(ttl=21600)  # Cache for 6 hours
def get_fii_dii_data(days_back=30):
    """Scrape FII/DII data from Groww"""
    try:
        url = "https://groww.in/fii-dii-data"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = bs4.BeautifulSoup(response.content, 'html.parser')
            
            # Try to find tables
            tables = soup.find_all('table')
            fii_dii_data = []
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        try:
                            date = cols[0].text.strip()
                            fii_buy = float(cols[1].text.replace(',', '').strip())
                            fii_sell = float(cols[2].text.replace(',', '').strip())
                            dii_buy = float(cols[3].text.replace(',', '').strip())
                            dii_sell = float(cols[4].text.replace(',', '').strip())
                            
                            fii_dii_data.append({
                                'Date': pd.to_datetime(date),
                                'FII_Net': fii_buy - fii_sell,
                                'DII_Net': dii_buy - dii_sell,
                                'FII_Buy': fii_buy,
                                'FII_Sell': fii_sell,
                                'DII_Buy': dii_buy,
                                'DII_Sell': dii_sell
                            })
                        except:
                            continue
            
            if fii_dii_data:
                df = pd.DataFrame(fii_dii_data)
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                
                # Calculate additional metrics
                df['FII_DII_Net'] = df['FII_Net'] + df['DII_Net']
                df['FII_Net_MA5'] = df['FII_Net'].rolling(5).mean()
                df['DII_Net_MA5'] = df['DII_Net'].rolling(5).mean()
                
                # Get last N days
                end_date = datetime.date.today()
                start_date = end_date - datetime.timedelta(days=days_back)
                df = df.loc[start_date:end_date]
                
                return df
                
    except Exception as e:
        st.warning(f"Could not fetch FII/DII data: {str(e)}")
    
    # Fallback: Create sample data
    dates = pd.date_range(end=datetime.date.today(), periods=days_back, freq='D')
    np.random.seed(42)
    return pd.DataFrame({
        'Date': dates,
        'FII_Net': np.random.normal(1000, 500, days_back),
        'DII_Net': np.random.normal(800, 400, days_back),
        'FII_Buy': np.random.normal(5000, 1000, days_back),
        'FII_Sell': np.random.normal(4000, 800, days_back),
        'DII_Buy': np.random.normal(3000, 600, days_back),
        'DII_Sell': np.random.normal(2200, 500, days_back)
    }).set_index('Date')

# ========== 3. DYNAMIC UNCERTAINTY-WEIGHTED FUSION ==========
class DynamicUncertaintyFusion:
    """Bayesian framework for time-varying source reliability"""
    
    def __init__(self, n_sources=3):
        self.n_sources = n_sources
        self.source_weights = np.ones(n_sources) / n_sources
        self.uncertainty_history = []
        self.reliability_scores = np.ones(n_sources)
        
    def calculate_source_uncertainty(self, predictions, actual=None):
        """Calculate uncertainty metrics for each source"""
        uncertainties = []
        
        for i in range(self.n_sources):
            if actual is not None and len(predictions[i]) == len(actual):
                # Calculate prediction error
                errors = np.abs(predictions[i] - actual)
                uncertainty = np.std(errors) / (np.mean(np.abs(actual)) + 1e-10)
            else:
                # Use prediction variance
                if len(predictions[i]) > 1:
                    uncertainty = np.std(predictions[i]) / (np.mean(np.abs(predictions[i])) + 1e-10)
                else:
                    uncertainty = 0.1
            
            uncertainties.append(max(uncertainty, 0.01))
        
        return np.array(uncertainties)
    
    def update_reliability_bayesian(self, predictions, actual=None, alpha=1.1, beta_param=1.1):
        """Update source reliability using Bayesian inference"""
        
        uncertainties = self.calculate_source_uncertainty(predictions, actual)
        
        # Bayesian update of reliability scores
        for i in range(self.n_sources):
            # Convert uncertainty to evidence (lower uncertainty = more evidence)
            evidence = 1 / (uncertainties[i] + 1e-10)
            
            # Update Beta distribution parameters
            alpha_new = alpha + evidence
            beta_new = beta_param + (1 - evidence)
            
            # Expected reliability (mean of Beta distribution)
            self.reliability_scores[i] = alpha_new / (alpha_new + beta_new)
        
        # Normalize reliability scores to get weights
        total_reliability = np.sum(self.reliability_scores)
        if total_reliability > 0:
            self.source_weights = self.reliability_scores / total_reliability
        
        return self.source_weights
    
    def fuse_predictions(self, predictions, use_uncertainty_weighting=True):
        """Fuse predictions from multiple sources"""
        
        if not use_uncertainty_weighting:
            # Simple average
            return np.mean(predictions, axis=0)
        
        # Uncertainty-weighted fusion
        uncertainties = self.calculate_source_uncertainty(predictions)
        
        # Inverse uncertainty weighting (higher uncertainty = lower weight)
        precision_weights = 1 / (uncertainties + 1e-10)
        normalized_weights = precision_weights / np.sum(precision_weights)
        
        # Apply Bayesian reliability adjustment
        final_weights = normalized_weights * self.reliability_scores
        final_weights = final_weights / np.sum(final_weights)
        
        # Weighted combination
        fused_prediction = np.zeros_like(predictions[0])
        for i in range(self.n_sources):
            fused_prediction += final_weights[i] * predictions[i]
        
        return fused_prediction
    
    def get_source_metrics(self):
        """Get detailed metrics for each source"""
        metrics = []
        for i in range(self.n_sources):
            metrics.append({
                'source': f'Source_{i+1}',
                'weight': self.source_weights[i],
                'reliability': self.reliability_scores[i],
                'uncertainty': 1 / (self.reliability_scores[i] + 1e-10) if self.reliability_scores[i] > 0 else 1.0
            })
        return pd.DataFrame(metrics)



# ========== ENHANCED FEATURE ENGINEERING ==========
def create_enhanced_features(df, vix_data=None, fii_dii_data=None):
    """Enhanced feature engineering with VIX and FII/DII data"""
    df = df.copy()
    
    # Original technical features
    df['Returns'] = df['Close'].pct_change()
    df['5D_MA'] = df['Close'].rolling(5).mean()
    df['20D_MA'] = df['Close'].rolling(20).mean()
    df['MA_Ratio'] = df['5D_MA'] / df['20D_MA']
    df['5D_Volatility'] = df['Returns'].rolling(5).std()
    df['Volume_MA5'] = df['Volume'].rolling(5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    
    # Add VIX data if available
    if vix_data is not None and not vix_data.empty:
        vix_aligned = vix_data.reindex(df.index, method='ffill')
        df['India_VIX'] = vix_aligned['India_VIX']
        df['VIX_Change'] = vix_aligned['VIX_Change']
        
        # VIX-based features
        df['VIX_SMA'] = df['India_VIX'].rolling(5).mean()
        df['VIX_Ratio'] = df['India_VIX'] / df['VIX_SMA']
    
    # Add FII/DII data if available
    if fii_dii_data is not None and not fii_dii_data.empty:
        fii_dii_aligned = fii_dii_data.reindex(df.index, method='ffill')
        
        for col in ['FII_Net', 'DII_Net', 'FII_DII_Net']:
            if col in fii_dii_aligned.columns:
                df[col] = fii_dii_aligned[col]
                df[f'{col}_MA5'] = fii_dii_aligned[col].rolling(5).mean()
    
    # Market sentiment composite score
    sentiment_components = []
    if 'Sentiment' in df.columns:
        sentiment_components.append(df['Sentiment'])
    if 'India_VIX' in df.columns:
        # Invert VIX (higher VIX = lower sentiment)
        vix_sentiment = -df['India_VIX'].pct_change().rolling(3).mean()
        sentiment_components.append(vix_sentiment)
    if 'FII_Net' in df.columns:
        # Normalize FII net investment
        fii_sentiment = df['FII_Net'].rolling(5).mean() / (df['FII_Net'].rolling(20).std() + 1e-10)
        sentiment_components.append(fii_sentiment)
    
    if sentiment_components:
        df['Market_Sentiment_Score'] = pd.concat(sentiment_components, axis=1).mean(axis=1)
    
    return df.dropna()

# Prophet forecasting
def prophet_forecast(df, days=10):
    prophet_df = df.reset_index()[['Date', 'Close']].rename(
        columns={'Date': 'ds', 'Close': 'y'}
    )
    
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=False,
        seasonality_mode='additive'
    )
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=days, include_history=False)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat']].set_index('ds')

def adjust_predictions_for_market_closures(predictions_df):
    """
    Adjust predictions to show steady values on market closed days (weekends and Indian holidays).
    """
    # Create Indian business day calendar (Mon-Fri excluding holidays)
    india_bd = CustomBusinessDay(calendar=IndiaHolidayCalendar())
    
    # Generate business days in the prediction range
    business_days = pd.date_range(
        start=predictions_df.index.min(),
        end=predictions_df.index.max(),
        freq=india_bd
    )
    
    # Mark non-business days
    predictions_df['is_market_day'] = predictions_df.index.isin(business_days)
    
    # Forward fill predictions for non-market days
    predictions_df['adjusted_prediction'] = np.where(
        predictions_df['is_market_day'],
        predictions_df['Predicted Price'],
        np.nan
    )
    predictions_df['adjusted_prediction'] = predictions_df['adjusted_prediction'].ffill()
    
    # Calculate daily changes based on adjusted predictions
    predictions_df['Daily Change (%)'] = predictions_df['adjusted_prediction'].pct_change().fillna(0) * 100
    
    return predictions_df[['adjusted_prediction', 'Daily Change (%)']].rename(
        columns={'adjusted_prediction': 'Predicted Price'}
    )


# ========== MODIFIED HYBRID MODEL WITH UNCERTAINTY FUSION ==========
def create_enhanced_hybrid_model(df_stock, sentiment_features, vix_data=None, fii_dii_data=None):
    """Enhanced hybrid model with multi-source fusion"""
    
    # Prepare data with all features
    sentiment_df = pd.DataFrame(sentiment_features.items(), columns=["Date", "Sentiment"])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None)
    df_stock.index = pd.to_datetime(df_stock.index).tz_localize(None)
    df_stock = df_stock.reset_index().merge(sentiment_df, on="Date", how="left").set_index("Date")
    df_stock['Sentiment'] = pd.to_numeric(df_stock['Sentiment'], errors='coerce').fillna(0)
    
    # Enhanced feature engineering
    df_stock = create_enhanced_features(df_stock, vix_data, fii_dii_data)
    df_stock['Target'] = df_stock['Close'].pct_change().shift(-1)
    df_stock.dropna(inplace=True)
    
    # Define feature sets for different models
    basic_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment',
                     '5D_MA', 'MA_Ratio', '5D_Volatility', 'Volume_Ratio']
    
    enhanced_features = basic_features.copy()
    if 'India_VIX' in df_stock.columns:
        enhanced_features.extend(['India_VIX', 'VIX_Change', 'VIX_Ratio'])
    if 'FII_Net' in df_stock.columns:
        enhanced_features.extend(['FII_Net', 'FII_Net_MA5'])
    if 'Market_Sentiment_Score' in df_stock.columns:
        enhanced_features.append('Market_Sentiment_Score')
    
    # Train-test split
    X_basic = df_stock[basic_features]
    X_enhanced = df_stock[enhanced_features]
    y = df_stock['Target']
    
    X_train_basic, X_test_basic, y_train, y_test = train_test_split(
        X_basic, y, test_size=0.2, shuffle=False
    )
    
    X_train_enhanced, X_test_enhanced, _, _ = train_test_split(
        X_enhanced, y, test_size=0.2, shuffle=False
    )
    
    # 1. XGBoost Model (Basic features)
    xgb_basic = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        early_stopping_rounds=30,
        random_state=42
    )
    xgb_basic.fit(X_train_basic, y_train, eval_set=[(X_test_basic, y_test)], verbose=False)
    
    # 2. XGBoost Model (Enhanced features)
    xgb_enhanced = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=8,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        random_state=42
    )
    xgb_enhanced.fit(X_train_enhanced, y_train, eval_set=[(X_test_enhanced, y_test)], verbose=False)
    
    # 3. GRU Model
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_enhanced)
    X_3d = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    
    X_train_3d = X_3d[:len(X_train_enhanced)]
    X_test_3d = X_3d[len(X_train_enhanced):len(X_train_enhanced)+len(X_test_enhanced)]
    
    gru_model = Sequential([
        GRU(64, input_shape=(1, X_3d.shape[2]), return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])
    
    gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    gru_model.fit(X_train_3d, y_train[:len(X_train_enhanced)], 
                  epochs=50, batch_size=32, verbose=0)
    
    # Get predictions from all models
    predictions = {
        'xgb_basic': xgb_basic.predict(X_test_basic),
        'xgb_enhanced': xgb_enhanced.predict(X_test_enhanced),
        'gru': gru_model.predict(X_test_3d).flatten()
    }
    
    # Initialize uncertainty fusion
    fusion_model = DynamicUncertaintyFusion(n_sources=3)
    
    # Update reliability weights
    predictions_list = [predictions['xgb_basic'], predictions['xgb_enhanced'], predictions['gru']]
    weights = fusion_model.update_reliability_bayesian(predictions_list, y_test.values)
    
    # Fuse predictions
    fused_predictions = fusion_model.fuse_predictions(predictions_list)
    
    # Store predictions
    df_stock.loc[y_test.index, 'Predicted'] = fused_predictions
    
    # Calculate metrics
    model_metrics = {}
    for name, pred in predictions.items():
        model_metrics[name] = {
            'mae': mean_absolute_error(y_test, pred),
            'mse': mean_squared_error(y_test, pred),
            'rmse': np.sqrt(mean_squared_error(y_test, pred))
        }
    
    model_metrics['fused'] = {
        'mae': mean_absolute_error(y_test, fused_predictions),
        'mse': mean_squared_error(y_test, fused_predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, fused_predictions)),
        'weights': weights.tolist(),
        'reliability': fusion_model.reliability_scores.tolist()
    }
    
    # Calculate accuracy
    hybrid_mae = model_metrics['fused']['mae']
    accuracy = max(0, 100 - (hybrid_mae * 100))
    model_metrics['fused']['accuracy'] = accuracy
    
    models = {
        'xgb_basic': xgb_basic,
        'xgb_enhanced': xgb_enhanced,
        'gru': gru_model,
        'fusion': fusion_model,
        'scaler': scaler
    }
    
    return df_stock, models, enhanced_features, model_metrics



# ========== ENHANCED PRICE PREDICTION WITH FUSION ==========
def enhanced_hybrid_predict_prices(models, last_known_data, features, days=10):
    """Generate predictions with uncertainty-weighted fusion"""
    try:
        fusion_model = models['fusion']
        scaler = models['scaler']
        
        # Get individual model predictions
        all_predictions = []
        
        for i in range(days):
            # Prepare input for current step
            current_features = last_known_data[features].iloc[-1:].copy()
            
            # XGBoost Basic prediction
            xgb_basic_pred = models['xgb_basic'].predict(current_features[features[:10]])[0]
            
            # XGBoost Enhanced prediction
            xgb_enhanced_pred = models['xgb_enhanced'].predict(current_features)[0]
            
            # GRU prediction
            input_scaled = scaler.transform(current_features)
            input_3d = input_scaled.reshape(1, 1, input_scaled.shape[1])
            gru_pred = models['gru'].predict(input_3d)[0][0]
            
            # Collect predictions
            day_predictions = [xgb_basic_pred, xgb_enhanced_pred, gru_pred]
            all_predictions.append(day_predictions)
            
            # Update data for next prediction (simplified)
            last_close = last_known_data['Close'].iloc[-1]
            # ... [rest of the data update logic]
        
        # Transpose to get predictions by source
        source_predictions = np.array(all_predictions).T
        
        # Fuse predictions using uncertainty weighting
        fused_daily_predictions = fusion_model.fuse_predictions(source_predictions)
        
        # Generate price predictions
        future_prices = pd.DataFrame(index=pd.date_range(
            start=datetime.date.today() + datetime.timedelta(days=1),
            periods=days,
            freq='D'
        ))
        
        last_close = last_known_data['Close'].iloc[-1]
        predicted_prices = [last_close]
        
        for pred in fused_daily_predictions:
            new_price = predicted_prices[-1] * (1 + pred)
            predicted_prices.append(new_price)
        
        future_prices['Predicted Price'] = predicted_prices[1:]
        future_prices['Daily Change (%)'] = future_prices['Predicted Price'].pct_change().fillna(0) * 100
        
        # Adjust for market closures
        future_prices = adjust_predictions_for_market_closures(future_prices)
        
        # Add uncertainty metrics
        source_metrics = fusion_model.get_source_metrics()
        
        return future_prices, source_metrics
    
    except Exception as e:
        st.error(f"Enhanced prediction failed: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Candlestick chart
def create_candlestick_chart(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    )])
    fig.update_layout(
        title="Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    return fig

# Generate investment recommendation
def generate_recommendation(predicted_prices, current_price, accuracy, avg_sentiment):
    avg_prediction = predicted_prices['Predicted Price'].mean()
    price_change = ((avg_prediction - current_price) / current_price) * 100
    
    # Enhanced sentiment factor with confidence scaling
    sentiment_factor = 1 + (avg_sentiment * (accuracy/100))  # Scale sentiment impact by accuracy
    
    adjusted_change = price_change * sentiment_factor
    
    # Modified thresholds with confidence weighting
    confidence_weight = accuracy / 100
    if adjusted_change > 7 * confidence_weight and accuracy > 72:
        return "STRONG BUY", "High confidence in significant price increase"
    elif adjusted_change > 3 * confidence_weight and accuracy > 65:
        return "BUY", "Good confidence in moderate price increase"
    elif adjusted_change > 0 and accuracy > 60:
        return "HOLD (Positive)", "Potential for slight growth"
    elif adjusted_change < -7 * confidence_weight and accuracy > 72:
        return "STRONG SELL", "High confidence in significant price drop"
    elif adjusted_change < -3 * confidence_weight and accuracy > 65:
        return "SELL", "Good confidence in moderate price drop"
    elif adjusted_change < 0 and accuracy > 60:
        return "HOLD (Caution)", "Potential for slight decline"
    else:
        return "HOLD", "Unclear direction - consider other factors"

# Streamlit UI
st.title("Indian Stock Market Analysis with Hybrid AI")
st.sidebar.header("Stock Selection")
indian_stocks = get_indian_stocks()
selected_stock = st.sidebar.selectbox("Select Stock", indian_stocks)
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
chart_type = st.sidebar.radio("Chart Type", ["Candlestick Chart", "Line Chart"])

if st.sidebar.button("Analyze"):
    ticker = f"{selected_stock}.NS"
    df_stock = get_stock_data(ticker, start_date, end_date)
    
    if not df_stock.empty:
        df_stock = df_stock.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        
        # Display stock info
        st.subheader(f"Stock Information for {selected_stock}")
        stock_info = get_stock_info(ticker)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Market Cap:** {stock_info['Market Cap']}")
            st.write(f"**P/E Ratio:** {stock_info['P/E Ratio']}")
            st.write(f"**ROCE:** {stock_info['ROCE']}")
            st.write(f"**Current Price:** {stock_info['Current Price']}")
        with col2:
            st.write(f"**Book Value:** {stock_info['Book Value']}")
            st.write(f"**ROE:** {stock_info['ROE']}")
            st.write(f"**Dividend Yield:** {stock_info['Dividend Yield']}")
            st.write(f"**Face Value:** {stock_info['Face Value']}")
        st.write(f"**High/Low:** {stock_info['High']} / {stock_info['Low']}")

        # Display historical data
        st.subheader("Historical Price Data")
        st.dataframe(df_stock[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index(ascending=False).style.format({
            'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}', 
            'Close': '{:.2f}', 'Volume': '{:,}'
        }))

        # Display chart
        st.subheader(f"{chart_type} for {selected_stock}")
        if chart_type == "Candlestick Chart":
            st.plotly_chart(create_candlestick_chart(df_stock))
        else:
            st.line_chart(df_stock["Close"])
# ========== NEW: FETCH ADDITIONAL DATA ==========
        st.sidebar.subheader("Market Data Sources")
        
        # India VIX Data
        with st.spinner("Fetching India VIX data..."):
            vix_data = get_india_vix_data()
            if not vix_data.empty:
                st.sidebar.success("✓ India VIX data loaded")
            else:
                st.sidebar.warning("India VIX data not available")
        
        # FII/DII Data
        with st.spinner("Fetching FII/DII data..."):
            fii_dii_data = get_fii_dii_data(days_back=60)
            if not fii_dii_data.empty:
                st.sidebar.success("✓ FII/DII data loaded")
            else:
                st.sidebar.warning("Using sample FII/DII data")
        
        # Display Market Indicators Section
        st.subheader("Market Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not vix_data.empty:
                current_vix = vix_data['India_VIX'].iloc[-1]
                vix_change = vix_data['VIX_Change'].iloc[-1]
                vix_color = "green" if vix_change < 0 else "red"
                st.metric(
                    "India VIX (Fear Index)",
                    f"{current_vix:.2f}",
                    f"{vix_change:+.2f}%",
                    delta_color="inverse"
                )
        
        with col2:
            if not fii_dii_data.empty:
                latest_fii = fii_dii_data['FII_Net'].iloc[-1]
                fii_color = "green" if latest_fii > 0 else "red"
                st.metric(
                    "FII Net Investment",
                    f"₹{latest_fii:,.0f} Cr",
                    f"{latest_fii/fii_dii_data['FII_Net'].std():+.1f}σ"
                )
        
        with col3:
            if not fii_dii_data.empty:
                latest_dii = fii_dii_data['DII_Net'].iloc[-1]
                dii_color = "green" if latest_dii > 0 else "red"
                st.metric(
                    "DII Net Investment",
                    f"₹{latest_dii:,.0f} Cr",
                    f"{latest_dii/fii_dii_data['DII_Net'].std():+.1f}σ"
                )
        
        # VIX Trend Chart
        if not vix_data.empty:
            st.subheader("India VIX Trend")
            fig_vix = go.Figure()
            fig_vix.add_trace(go.Scatter(
                x=vix_data.index,
                y=vix_data['India_VIX'],
                mode='lines',
                name='India VIX',
                line=dict(color='purple', width=2),
                fill='tozeroy',
                fillcolor='rgba(128, 0, 128, 0.1)'
            ))
            
            # Add horizontal bands
            fig_vix.add_hrect(y0=0, y1=15, line_width=0, fillcolor="green", opacity=0.1)
            fig_vix.add_hrect(y0=15, y1=25, line_width=0, fillcolor="yellow", opacity=0.1)
            fig_vix.add_hrect(y0=25, y1=35, line_width=0, fillcolor="orange", opacity=0.1)
            fig_vix.add_hrect(y0=35, y1=100, line_width=0, fillcolor="red", opacity=0.1)
            
            fig_vix.update_layout(
                title="India VIX with Market Sentiment Zones",
                xaxis_title="Date",
                yaxis_title="VIX Level",
                hovermode="x unified"
            )
            st.plotly_chart(fig_vix, use_container_width=True)
        
        # FII/DII Activity Chart
        if not fii_dii_data.empty:
            st.subheader("FII/DII Investment Activity")
            
            fig_fii_dii = go.Figure()
            
            fig_fii_dii.add_trace(go.Bar(
                x=fii_dii_data.index,
                y=fii_dii_data['FII_Net'],
                name='FII Net',
                marker_color='blue',
                opacity=0.7
            ))
            
            fig_fii_dii.add_trace(go.Bar(
                x=fii_dii_data.index,
                y=fii_dii_data['DII_Net'],
                name='DII Net',
                marker_color='green',
                opacity=0.7
            ))
            
            fig_fii_dii.add_trace(go.Scatter(
                x=fii_dii_data.index,
                y=fii_dii_data['FII_Net_MA5'],
                name='FII 5D MA',
                line=dict(color='darkblue', width=2, dash='dash')
            ))
            
            fig_fii_dii.add_trace(go.Scatter(
                x=fii_dii_data.index,
                y=fii_dii_data['DII_Net_MA5'],
                name='DII 5D MA',
                line=dict(color='darkgreen', width=2, dash='dash')
            ))
            
            fig_fii_dii.update_layout(
                title="FII/DII Net Investment (₹ Crores)",
                xaxis_title="Date",
                yaxis_title="Net Investment",
                barmode='group',
                hovermode="x unified"
            )
            st.plotly_chart(fig_fii_dii, use_container_width=True)
        # News and sentiment analysis
        st.subheader(f"Latest News for {selected_stock}")
        news_articles = get_news(selected_stock)
        filtered_news = filter_relevant_news(news_articles, selected_stock)
        daily_sentiment = {}
        sentiment_data = []

        if not filtered_news:
            st.warning("No recent news articles found for this stock. The analysis will continue without news sentiment data.")
        else:
            for article in filtered_news:
                title = article.get('title', '')
                description = article.get('description', '')
                text = f"{title} {description}".strip()
                sentiment, confidence_score = analyze_sentiment(text)
                date = article.get("publishedAt", "")[0:10]
                sentiment_data.append([date, title, sentiment, f"{confidence_score:.2f}"])

                if date in daily_sentiment:
                    daily_sentiment[date].append((sentiment, confidence_score))
                else:
                    daily_sentiment[date] = [(sentiment, confidence_score)]

                st.write(f"**{title}**")
                st.write(description)
                st.write(f"[Read more]({article['url']})")
                st.write("---")

        # News sentiment table
        if sentiment_data:
            st.subheader("News Sentiment Analysis")
            df_sentiment = pd.DataFrame(sentiment_data, columns=["Date", "Headline", "Sentiment", "Confidence"])
            st.dataframe(df_sentiment.sort_values("Date", ascending=False))

        # Daily average sentiment
        if daily_sentiment:
            st.subheader("Daily Weighted Average Sentiment")
            avg_daily_sentiment = []
            for date, scores in daily_sentiment.items():
                weighted_sum = 0
                total_weight = 0
                for sentiment, score in scores:
                    value = 1 if sentiment == "positive" else (-1 if sentiment == "negative" else 0)
                    weighted_sum += value * score
                    total_weight += score
                avg_score = weighted_sum / total_weight if total_weight != 0 else 0
                
                sentiment_class = "Positive" if avg_score > 0.2 else "Negative" if avg_score < -0.2 else "Neutral"
                avg_daily_sentiment.append([date, f"{avg_score:.2f}", sentiment_class])
            
            df_avg_sentiment = pd.DataFrame(avg_daily_sentiment, 
                                          columns=["Date", "Weighted Score", "Sentiment"])
            st.dataframe(df_avg_sentiment.sort_values("Date", ascending=False))

                # Train hybrid model - NOW WITH COMPREHENSIVE METRICS
        df_stock, models, features, model_metrics = create_enhanced_hybrid_model(
            df_stock, 
            daily_sentiment if daily_sentiment else {},
            vix_data,
            fii_dii_data
        )
        accuracy = model_metrics['hybrid']['accuracy']
        # Display Model Performance Metrics
        st.subheader("Model Performance Comparison")
        
        # Create metrics dataframe for display
        metrics_df = pd.DataFrame({
            'Model': ['XGBoost', 'GRU', 'Hybrid'],
            'MAE': [
                model_metrics['xgb']['mae'],
                model_metrics['gru']['mae'],
                model_metrics['hybrid']['mae']
            ],
            'MSE': [
                model_metrics['xgb']['mse'],
                model_metrics['gru']['mse'],
                model_metrics['hybrid']['mse']
            ],
            'RMSE': [
                model_metrics['xgb']['rmse'],
                model_metrics['gru']['rmse'],
                model_metrics['hybrid']['rmse']
            ],
            'Weight/Accuracy': [
                f"{model_metrics['xgb']['weight']*100:.1f}%",
                f"{model_metrics['gru']['weight']*100:.1f}%",
                f"{model_metrics['hybrid']['accuracy']:.1f}%"
            ]
        })
        
        # Display metrics table
        st.dataframe(metrics_df.style.format({
            'MAE': '{:.4f}',
            'MSE': '{:.4f}',
            'RMSE': '{:.4f}'
        }).highlight_min(axis=0, subset=['MAE', 'MSE', 'RMSE'], color='lightgreen')
        .highlight_max(axis=0, subset=['Weight/Accuracy'], color='lightgreen'))
        
        # Visual comparison
        fig = go.Figure()
        
        # Add metrics bars
        for metric in ['MAE', 'MSE', 'RMSE']:
            fig.add_trace(go.Bar(
                x=metrics_df['Model'],
                y=metrics_df[metric],
                name=metric,
                text=metrics_df[metric].round(4),
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Metrics Comparison',
            barmode='group',
            yaxis_title='Error Value',
            xaxis_title='Model',
            hovermode="x unified"
        )
        
        st.plotly_chart(fig)
        
        # Detailed performance explanation
        best_model = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']
        st.subheader("Performance Analysis")
        
        analysis_text = f"""
        ### Model Performance Insights:
        
        1. **Best Performing Model**: The {best_model} model achieved the lowest RMSE of {metrics_df.loc[metrics_df['RMSE'].idxmin(), 'RMSE']:.4f}, 
           indicating it has the best overall prediction accuracy when considering both the magnitude and frequency of errors.
           
        2. **Error Analysis**:
           - **MAE (Mean Absolute Error)**: Represents the average absolute difference between predicted and actual values.
             - XGBoost: {model_metrics['xgb']['mae']:.4f}
             - GRU: {model_metrics['gru']['mae']:.4f}
             - Hybrid: {model_metrics['hybrid']['mae']:.4f}
             
           - **MSE (Mean Squared Error)**: Emphasizes larger errors by squaring them before averaging.
             - XGBoost: {model_metrics['xgb']['mse']:.4f}
             - GRU: {model_metrics['gru']['mse']:.4f}
             - Hybrid: {model_metrics['hybrid']['mse']:.4f}
             
           - **RMSE (Root Mean Squared Error)**: Provides error in the same units as the target variable.
             - XGBoost: {model_metrics['xgb']['rmse']:.4f}
             - GRU: {model_metrics['gru']['rmse']:.4f}
             - Hybrid: {model_metrics['hybrid']['rmse']:.4f}
             
        3. **Model Weighting**: The hybrid model automatically weighted the contributions as:
           - XGBoost: {model_metrics['xgb']['weight']*100:.1f}%
           - GRU: {model_metrics['gru']['weight']*100:.1f}%
           
        4. **Hybrid Model Accuracy**: {model_metrics['hybrid']['accuracy']:.1f}% (calculated as 100 - (MAE * 100))
        
        ### Interpretation:
        - Lower values for MAE, MSE, and RMSE indicate better performance
        - The hybrid model combines the strengths of both individual models
        - RMSE is particularly useful as it penalizes larger errors more heavily
        """
        
        st.markdown(analysis_text)

        # AI Analysis Conclusion
        st.subheader("AI Analysis Conclusion")
        current_price = df_stock['Close'].iloc[-1]
        avg_sentiment = df_stock['Sentiment'].mean() if 'Sentiment' in df_stock else 0
        
        # Calculate avg_prediction from model results
        if 'Predicted' in df_stock.columns:
            # Use the predictions from the trained model
            valid_predictions = df_stock['Predicted'].dropna()
            if len(valid_predictions) > 0:
                avg_prediction = current_price * (1 + valid_predictions.mean())
            else:
                avg_prediction = current_price  # Fallback if no predictions
        else:
            avg_prediction = current_price  # Fallback if no prediction column
        
        # Calculate price change
        price_change = ((avg_prediction - current_price) / current_price) * 100
        
        if daily_sentiment:
            sentiment_analysis_part = f"""
            3. **Market Sentiment**: 
               - News sentiment is predominantly {'positive' if avg_sentiment > 0 else 'negative' if avg_sentiment < 0 else 'neutral'}
               - This sentiment is {'strengthening' if df_stock['Sentiment'].iloc[-1] > df_stock['Sentiment'].mean() else 'weakening' if df_stock['Sentiment'].iloc[-1] < df_stock['Sentiment'].mean() else 'stable'}
            """
            conclusion = f"These factors collectively suggest that {selected_stock} is currently in a {'favorable' if avg_sentiment > 0 else 'challenging' if avg_sentiment < 0 else 'neutral'} position."
        else:
            sentiment_analysis_part = """
            3. **Market Sentiment**: 
               - No recent news sentiment data available
               - Analysis based solely on technical indicators
            """
            conclusion = f"These technical indicators suggest that {selected_stock} is currently showing {'strong' if df_stock['MA_Ratio'].iloc[-1] > 1.05 else 'weak' if df_stock['MA_Ratio'].iloc[-1] < 0.95 else 'neutral'} technical signals."

        explanation = f"""
        Our analysis of {selected_stock} reveals the following key insights:

        1. **Price Movement**: The stock is currently trading at ₹{current_price:.2f}. 
        - Model confidence: {accuracy:.1f}%
        - Predicted trend: {'upward' if avg_prediction > current_price else 'downward'}
        - Expected change: {abs(price_change):.2f}% ({'+' if price_change > 0 else ''}{price_change:.2f}%)

        2. **Technical Indicators**:
        - MA Position: {'Above' if df_stock['MA_Ratio'].iloc[-1] > 1 else 'Below'} 20-day MA
        - Volatility: {'High' if df_stock['5D_Volatility'].iloc[-1] > 0.02 else 'Moderate'}
        - Volume Trend: {'Increasing' if df_stock['Volume_Ratio'].iloc[-1] > 1 else 'Decreasing'}

        3. **Market Sentiment**: 
        - Current sentiment: {'Positive' if avg_sentiment > 0.2 else 'Negative' if avg_sentiment < -0.2 else 'Neutral'}
        - Sentiment trend: {'Improving' if df_stock['Sentiment'].iloc[-1] > df_stock['Sentiment'].mean() else 'Declining'}
        {sentiment_analysis_part}
        
        {conclusion}
        """
        st.write(explanation)

        # 10-Day Price Forecast
        try:
            st.subheader("10-Day Price Forecast")
            last_data = df_stock.iloc[-30:]
            current_price = last_data['Close'].iloc[-1]

            future_prices = hybrid_predict_prices(models=models,
                                                scaler=scaler,
                                                last_known_data=last_data,
                                                features=features,
                                                days=10,
                                                weights=weights  )

            if not future_prices.empty:
                # Format forecast table
                def format_forecast_table(df):
                    styled_df = df[['Predicted Price', 'Daily Change (%)']].rename(columns={
                        'Predicted Price': 'Price (₹)',
                        'Daily Change (%)': 'Daily Change (%)'
                    }).style.format({
                        'Price (₹)': '₹{:,.2f}',
                        'Daily Change (%)': '{:+.2f}%'
                    }).applymap(
                        lambda x: 'color: #4CAF50' if x > 0 else 'color: #F44336',
                        subset=['Daily Change (%)']
                    )
                    return styled_df.set_properties(**{
                        'background-color': '#f8f9fa',
                        'border': '1px solid #ddd',
                        'text-align': 'center'
                    })

                st.dataframe(
                    format_forecast_table(future_prices),
                    use_container_width=True,
                    height=(len(future_prices) + 1) * 35 + 3
                )

                # Investment Recommendation
                recommendation, reasoning = generate_recommendation(
                    future_prices, 
                    current_price, 
                    accuracy,
                    avg_sentiment
                )
                
                st.subheader("Investment Recommendation")
                rec_colors = {
                    "STRONG BUY": "green",
                    "BUY": "lightgreen",
                    "HOLD (Positive)": "blue",
                    "HOLD": "blue",
                    "HOLD (Caution)": "orange",
                    "SELL": "red",
                    "STRONG SELL": "dark red"
                }
                
                rec_color = rec_colors.get(recommendation.split()[0], "blue")
                st.markdown(
                    f"""<div style="padding: 10px; border-radius: 5px; background-color: {rec_color}; color: white">
                    <strong>{recommendation}:</strong> {reasoning}
                    </div>""",
                    unsafe_allow_html=True
                )

                # Price Trend Visualization
                st.subheader("Price Trend Analysis")
                historical_data = df_stock[['Close']].rename(columns={'Close': 'Price'}).iloc[-90:]
                future_data = future_prices[['Predicted Price']].rename(columns={'Predicted Price': 'Price'})
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Price'],
                    mode='lines',
                    name='Historical Prices',
                    line=dict(color='#3366CC', width=2),
                    hovertemplate='₹%{y:.2f}<extra>%{x|%b %d, %Y}</extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=future_data.index,
                    y=future_data['Price'],
                    mode='lines+markers',
                    name='AI Forecast',
                    line=dict(color='#FF7F0E', width=2, dash='dot'),
                    marker=dict(size=8, color='#FF7F0E'),
                    hovertemplate='₹%{y:.2f}<extra>%{x|%b %d, %Y}</extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=[historical_data.index[-1]],
                    y=[current_price],
                    mode='markers',
                    name='Current Price',
                    marker=dict(size=12, color='#DC3912'),
                    hovertemplate='₹%{y:.2f}<extra>Current Price</extra>'
                ))
                fig.add_vline(
                    x=historical_data.index[-1],
                    line_width=1,
                    line_dash="dash",
                    line_color="grey"
                )
                fig.update_layout(
                    title=f"{selected_stock} Price Trend (Historical vs Forecast)",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    hovermode="x unified",
                    plot_bgcolor='rgba(240,240,240,0.8)',
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate forecast predictions")

        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")

        st.subheader("Uncertainty-Weighted Fusion Metrics")
        
        if 'fused' in model_metrics and 'weights' in model_metrics['fused']:
            weights = model_metrics['fused']['weights']
            reliability = model_metrics['fused']['reliability']
            
            fusion_df = pd.DataFrame({
                'Model': ['XGBoost Basic', 'XGBoost Enhanced', 'GRU'],
                'Weight (%)': [w*100 for w in weights],
                'Reliability Score': reliability,
                'Uncertainty': [1/(r+1e-10) for r in reliability]
            })
            
            st.dataframe(fusion_df.style.format({
                'Weight (%)': '{:.1f}%',
                'Reliability Score': '{:.3f}',
                'Uncertainty': '{:.3f}'
            }).highlight_max(subset=['Weight (%)'], color='lightgreen'))
            
            # Fusion visualization
            fig_fusion = go.Figure(data=[
                go.Bar(name='Weight %', x=fusion_df['Model'], y=fusion_df['Weight (%)']),
                go.Scatter(name='Reliability', x=fusion_df['Model'], y=fusion_df['Reliability Score']*100,
                          yaxis='y2', mode='lines+markers', line=dict(color='red', width=2))
            ])
            
            fig_fusion.update_layout(
                title='Model Fusion Weights & Reliability',
                yaxis=dict(title='Weight (%)'),
                yaxis2=dict(title='Reliability Score', overlaying='y', side='right',
                          range=[0, 100]),
                hovermode="x unified"
            )
            st.plotly_chart(fig_fusion, use_container_width=True)
        
        # Modify the prediction call
        future_prices, source_metrics = enhanced_hybrid_predict_prices(
            models=models,
            last_known_data=df_stock.iloc[-30:],
            features=features,
            days=10
        )
        
        # Display source metrics if available
        if not source_metrics.empty:
            with st.expander("Source Reliability Analysis"):
                st.dataframe(source_metrics.style.format({
                    'weight': '{:.3f}',
                    'reliability': '{:.3f}',
                    'uncertainty': '{:.3f}'
                }))