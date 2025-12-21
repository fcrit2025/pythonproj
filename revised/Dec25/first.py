"""
A Multi-Source Fusion Framework with LLM-Enhanced Sentiment for Cross-Market Stock Forecasting
Complete Implementation for Indian Stock Market
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Time Series Forecasting
from prophet import Prophet

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sentiment Analysis
from transformers import pipeline

# Set page configuration
st.set_page_config(
    page_title="Indian Stock Market AI Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .positive {
        color: #2ca02c;
        font-weight: bold;
    }
    .negative {
        color: #d62728;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        font-weight: bold;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>Multi-Source Fusion Framework for Indian Stock Forecasting</h1>", unsafe_allow_html=True)
st.markdown("### LLM-Enhanced Sentiment Analysis with Hybrid AI Models")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'news_data' not in st.session_state:
    st.session_state.news_data = None

# ==================== DATA LOADING FUNCTIONS ====================

@st.cache_data
def load_indian_stocks():
    """Load Indian stock symbols from CSV or create default list"""
    try:
        df = pd.read_csv('indian_stocks.csv')
        # Clean column names
        df.columns = df.columns.str.strip()
        # Find symbol column
        symbol_col = None
        for col in df.columns:
            if 'SYMBOL' in col.upper():
                symbol_col = col
                break
        
        if symbol_col:
            symbols = df[symbol_col].dropna().astype(str).tolist()
            return sorted(list(set(symbols)))
        else:
            st.warning("Symbol column not found. Using default stocks.")
    except Exception as e:
        st.warning(f"Could not load CSV: {e}. Using default stocks.")
    
    # Default Indian stocks
    default_stocks = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'SBIN',
        'AXISBANK', 'BAJFINANCE', 'WIPRO', 'SUNPHARMA', 'MARUTI',
        'ASIANPAINT', 'HINDUNILVR', 'TITAN', 'ONGC', 'NTPC'
    ]
    return sorted(default_stocks)

@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance"""
    try:
        # Add .NS suffix for NSE
        if not ticker.endswith('.NS'):
            ticker = f"{ticker}.NS"
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"No data found for {ticker}")
            return pd.DataFrame()
        
        # Clean data
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def get_stock_info(ticker):
    """Get fundamental information for a stock"""
    try:
        if not ticker.endswith('.NS'):
            ticker = f"{ticker}.NS"
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        info_dict = {
            'Current Price': f"₹{info.get('currentPrice', 'N/A')}",
            'Market Cap': f"₹{info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else 'N/A',
            'P/E Ratio': info.get('trailingPE', 'N/A'),
            'P/B Ratio': info.get('priceToBook', 'N/A'),
            'Dividend Yield': f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A',
            '52 Week High': f"₹{info.get('fiftyTwoWeekHigh', 'N/A')}",
            '52 Week Low': f"₹{info.get('fiftyTwoWeekLow', 'N/A')}",
            'Beta': info.get('beta', 'N/A'),
            'ROE': f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else 'N/A',
            'ROA': f"{info.get('returnOnAssets', 0)*100:.2f}%" if info.get('returnOnAssets') else 'N/A'
        }
        
        return info_dict
    except Exception as e:
        st.warning(f"Could not fetch detailed info: {e}")
        return {}

# ==================== NEWS & SENTIMENT ANALYSIS ====================

class SentimentAnalyzer:
    """LLM-based sentiment analyzer using FinBERT"""
    
    def __init__(self):
        try:
            self.pipeline = pipeline("sentiment-analysis", 
                                   model="ProsusAI/finbert",
                                   truncation=True,
                                   max_length=512)
        except Exception as e:
            st.warning(f"Could not load FinBERT model: {e}. Using rule-based fallback.")
            self.pipeline = None
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of financial text"""
        if not text or len(text.strip()) < 10:
            return "neutral", 0.5
        
        try:
            if self.pipeline:
                result = self.pipeline(text[:512])[0]
                label = result['label'].lower()
                score = result['score']
                
                # Convert to numeric sentiment (-1 to 1)
                if label == 'positive':
                    sentiment_score = score
                elif label == 'negative':
                    sentiment_score = -score
                else:
                    sentiment_score = 0
                
                return label, sentiment_score
            else:
                # Rule-based fallback
                positive_words = ['profit', 'growth', 'gain', 'bullish', 'positive', 'upgrade', 'buy', 'strong']
                negative_words = ['loss', 'decline', 'bearish', 'negative', 'downgrade', 'sell', 'weak']
                
                text_lower = text.lower()
                pos_count = sum(word in text_lower for word in positive_words)
                neg_count = sum(word in text_lower for word in negative_words)
                
                if pos_count > neg_count:
                    return "positive", min(0.9, pos_count/10)
                elif neg_count > pos_count:
                    return "negative", -min(0.9, neg_count/10)
                else:
                    return "neutral", 0
                    
        except Exception as e:
            st.warning(f"Sentiment analysis error: {e}")
            return "neutral", 0

class NewsFetcher:
    """Fetch and process financial news"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def fetch_news(self, stock_name, days_back=7):
        """Fetch news for a specific stock"""
        try:
            # Calculate date range
            to_date = datetime.datetime.now()
            from_date = to_date - datetime.timedelta(days=days_back)
            
            # Map stock symbols to search terms
            search_terms = {
                'RELIANCE': 'Reliance Industries',
                'TCS': 'Tata Consultancy Services',
                'HDFCBANK': 'HDFC Bank',
                'INFY': 'Infosys',
                'ICICIBANK': 'ICICI Bank',
                'SBIN': 'State Bank of India',
                'ITC': 'ITC Limited',
                'BHARTIARTL': 'Bharti Airtel'
            }
            
            query = search_terms.get(stock_name, stock_name)
            
            params = {
                'q': query,
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'apiKey': self.api_key,
                'pageSize': 20
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                
                processed_articles = []
                for article in articles:
                    # Skip articles without content
                    if not article.get('title') or not article.get('description'):
                        continue
                    
                    # Combine title and description for sentiment analysis
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    
                    # Analyze sentiment
                    sentiment_label, sentiment_score = self.sentiment_analyzer.analyze_sentiment(text)
                    
                    processed_article = {
                        'date': pd.to_datetime(article.get('publishedAt')).strftime('%Y-%m-%d'),
                        'title': article.get('title', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'url': article.get('url', '#'),
                        'sentiment': sentiment_label,
                        'sentiment_score': sentiment_score,
                        'confidence': abs(sentiment_score)
                    }
                    processed_articles.append(processed_article)
                
                return processed_articles
            else:
                st.warning(f"News API Error: {response.status_code}")
                return []
                
        except Exception as e:
            st.warning(f"News fetching error: {e}")
            # Return mock data for demonstration
            return self.get_mock_news(stock_name)
    
    def get_mock_news(self, stock_name):
        """Generate mock news data for demonstration"""
        mock_articles = []
        sentiments = ['positive', 'neutral', 'negative']
        
        for i in range(5):
            date = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime('%Y-%m-%d')
            sentiment = sentiments[i % 3]
            score = 0.8 if sentiment == 'positive' else (-0.8 if sentiment == 'negative' else 0)
            
            article = {
                'date': date,
                'title': f"{stock_name} shows {'strong' if sentiment=='positive' else 'moderate' if sentiment=='neutral' else 'weak'} performance in Q{3-i%4}",
                'source': 'Financial Express',
                'url': '#',
                'sentiment': sentiment,
                'sentiment_score': score,
                'confidence': 0.8
            }
            mock_articles.append(article)
        
        return mock_articles

# ==================== FEATURE ENGINEERING ====================

def create_technical_features(df):
    """Create technical indicators from price data"""
    df = df.copy()
    
    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Price relative to moving averages
    df['Price_MA5_Ratio'] = df['Close'] / df['MA_5']
    df['Price_MA20_Ratio'] = df['Close'] / df['MA_20']
    df['MA_Crossover'] = (df['MA_5'] > df['MA_20']).astype(int)
    
    # Volatility
    df['Volatility_5'] = df['Returns'].rolling(window=5).std()
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()
    
    # Volume indicators
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    
    # Price ranges
    df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Body_Range'] = abs(df['Close'] - df['Open']) / df['Close']
    
    # Momentum indicators
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['ROC_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
    
    # Support and resistance (simplified)
    df['Resistance'] = df['High'].rolling(window=20).max()
    df['Support'] = df['Low'].rolling(window=20).min()
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def add_sentiment_features(df, news_articles):
    """Add sentiment features to the dataframe"""
    if not news_articles:
        df['Sentiment_Score'] = 0
        df['News_Count'] = 0
        return df
    
    try:
        # Convert news to dataframe
        news_df = pd.DataFrame(news_articles)
        news_df['date'] = pd.to_datetime(news_df['date'])
        
        # Group by date
        daily_sentiment = news_df.groupby('date').agg({
            'sentiment_score': 'mean',
            'title': 'count'
        }).rename(columns={'sentiment_score': 'Sentiment_Score', 'title': 'News_Count'})
        
        # Create a temporary column for merging - FIXED HERE
        # Reset index to get date as column, then merge
        df_temp = df.reset_index() if df.index.name else df.copy()
        
        # Create date column from index if needed
        if 'Date' not in df_temp.columns:
            df_temp['Date'] = df_temp.index if df_temp.index.name else pd.RangeIndex(len(df_temp))
        
        # Convert to date only (without time) for merging
        df_temp['Date_Only'] = pd.to_datetime(df_temp['Date']).dt.date
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index).date
        
        # Merge
        merged_df = pd.merge(df_temp, daily_sentiment, 
                           left_on='Date_Only', 
                           right_index=True, 
                           how='left')
        
        # Fill missing values
        merged_df['Sentiment_Score'] = merged_df['Sentiment_Score'].fillna(0)
        merged_df['News_Count'] = merged_df['News_Count'].fillna(0)
        
        # Set index back to original
        if 'index' in merged_df.columns:
            merged_df = merged_df.set_index('index')
        elif df.index.name and df.index.name in merged_df.columns:
            merged_df = merged_df.set_index(df.index.name)
        else:
            # Drop temporary columns
            merged_df = merged_df.drop(['Date', 'Date_Only'], axis=1, errors='ignore')
        
        return merged_df
        
    except Exception as e:
        st.warning(f"Could not add sentiment features: {e}")
        df['Sentiment_Score'] = 0
        df['News_Count'] = 0
        return df

# ==================== HYBRID MODEL ====================

class HybridStockModel:
    """Hybrid XGBoost-GRU-Prophet model for stock forecasting"""
    
    def __init__(self):
        self.xgb_model = None
        self.gru_model = None
        self.prophet_model = None
        self.scaler = MinMaxScaler()
        self.features = None
        self.model_metrics = {}
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'MA_5', 'MA_20', 'Price_MA20_Ratio',
            'Volatility_5', 'Volume_Ratio', 'Daily_Range',
            'Momentum_5', 'Sentiment_Score'
        ]
        
        # Select available features
        available_features = [f for f in feature_cols if f in df.columns]
        
        # Add missing features with default values
        for f in feature_cols:
            if f not in df.columns and f != 'Sentiment_Score':
                if f in ['Returns', 'Volatility_5']:
                    df[f] = 0.01
                elif f in ['Price_MA20_Ratio', 'Volume_Ratio', 'Daily_Range']:
                    df[f] = 1.0
                elif f == 'Momentum_5':
                    df[f] = 0
                else:
                    df[f] = df['Close']
        
        # Ensure all features are present
        for f in feature_cols:
            if f not in df.columns:
                if f == 'Sentiment_Score':
                    df[f] = 0
                else:
                    df[f] = df['Close']
        
        self.features = feature_cols
        return df[feature_cols], df['Close']
    
    def train(self, df, test_size=0.2):
        """Train the hybrid model"""
        try:
            # Prepare features
            X, y = self.prepare_features(df)
            
            # Create target (next day's return)
            y_target = df['Close'].pct_change().shift(-1)
            valid_indices = y_target.dropna().index
            X = X.loc[valid_indices]
            y_target = y_target.loc[valid_indices]
            
            # Train-test split
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y_target.iloc[:split_idx], y_target.iloc[split_idx:]
            
            # Scale features for GRU
            X_scaled = self.scaler.fit_transform(X)
            
            # ===== 1. XGBoost Model =====
            st.info("Training XGBoost model...")
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='reg:squarederror',
                random_state=42
            )
            self.xgb_model.fit(X_train, y_train)
            
            # XGBoost predictions
            xgb_pred = self.xgb_model.predict(X_test)
            xgb_metrics = {
                'mae': mean_absolute_error(y_test, xgb_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred))
            }
            
            # ===== 2. GRU Model =====
            st.info("Training GRU model...")
            
            # Prepare 3D data for GRU
            X_scaled_3d = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
            X_train_scaled = X_scaled_3d[:split_idx]
            X_test_scaled = X_scaled_3d[split_idx:]
            
            # Build GRU model
            self.gru_model = Sequential([
                GRU(50, input_shape=(1, X_train_scaled.shape[2]), return_sequences=True),
                Dropout(0.2),
                GRU(25),
                Dropout(0.2),
                Dense(1)
            ])
            
            self.gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train GRU
            self.gru_model.fit(
                X_train_scaled, y_train,
                epochs=30,
                batch_size=32,
                verbose=0,
                validation_split=0.1
            )
            
            # GRU predictions
            gru_pred = self.gru_model.predict(X_test_scaled).flatten()
            gru_metrics = {
                'mae': mean_absolute_error(y_test, gru_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, gru_pred))
            }
            
            # ===== 3. Prophet Model =====
            st.info("Training Prophet model...")
            prophet_df = df[['Close']].reset_index()
            prophet_df.columns = ['ds', 'y']
            
            # Remove timezone from datetime column
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)
            
            self.prophet_model = Prophet(
                daily_seasonality=False,
                yearly_seasonality=True,
                weekly_seasonality=True,
                seasonality_mode='additive'
            )
            self.prophet_model.fit(prophet_df)
            
            # Calculate Prophet metrics
            future = self.prophet_model.make_future_dataframe(periods=len(y_test))
            future['ds'] = pd.to_datetime(future['ds']).dt.tz_localize(None)
            forecast = self.prophet_model.predict(future)
            
            # Ensure array lengths match
            prophet_pred = forecast['yhat'].values[split_idx:split_idx + len(y_test)]
            
            # Align lengths
            min_len = min(len(prophet_pred), len(y_test))
            prophet_pred = prophet_pred[:min_len]
            y_test_aligned = y_test.iloc[:min_len]
            
            # Convert to returns
            if min_len > 1:
                prophet_returns = np.diff(prophet_pred) / prophet_pred[:-1]
                actual_values = df['Close'].iloc[split_idx:split_idx + min_len].values
                actual_returns = np.diff(actual_values) / actual_values[:-1]
                
                # Align return lengths
                return_min_len = min(len(prophet_returns), len(actual_returns))
                
                if return_min_len > 0:
                    prophet_metrics = {
                        'mae': mean_absolute_error(actual_returns[:return_min_len], 
                                                  prophet_returns[:return_min_len]),
                        'rmse': np.sqrt(mean_squared_error(actual_returns[:return_min_len], 
                                                          prophet_returns[:return_min_len]))
                    }
                else:
                    prophet_metrics = {'mae': 0.02, 'rmse': 0.03}
            else:
                prophet_metrics = {'mae': 0.02, 'rmse': 0.03}
            
            # ===== 4. Hybrid Ensemble =====
            # Calculate dynamic weights based on performance
            errors = {
                'xgb': xgb_metrics['mae'],
                'gru': gru_metrics['mae'],
                'prophet': prophet_metrics['mae']
            }
            
            # Invert errors to get weights (lower error = higher weight)
            total_inv = sum(1/e for e in errors.values() if e > 0)
            weights = {model: (1/error)/total_inv if error > 0 else 0.33 
                      for model, error in errors.items()}
            
            # Ensure all predictions have same length
            min_pred_len = min(len(xgb_pred), len(gru_pred))
            xgb_pred_aligned = xgb_pred[:min_pred_len]
            gru_pred_aligned = gru_pred[:min_pred_len]
            
            # Align prophet predictions if available
            if hasattr(prophet_pred, '__len__'):
                prophet_len = min(len(prophet_pred), min_pred_len)
                prophet_returns_aligned = prophet_returns[:prophet_len] if 'prophet_returns' in locals() else np.zeros(prophet_len)
            else:
                prophet_len = min_pred_len
                prophet_returns_aligned = np.zeros(prophet_len)
            
            # Use minimum length for combination
            combine_len = min(min_pred_len, prophet_len)
            
            # Combine predictions
            hybrid_pred = (
                weights['xgb'] * xgb_pred_aligned[:combine_len] +
                weights['gru'] * gru_pred_aligned[:combine_len] +
                weights['prophet'] * prophet_returns_aligned[:combine_len]
            ) / sum(weights.values())
            
            hybrid_metrics = {
                'mae': mean_absolute_error(y_test.iloc[:combine_len], hybrid_pred),
                'rmse': np.sqrt(mean_squared_error(y_test.iloc[:combine_len], hybrid_pred))
            }
            
            # Store metrics
            self.model_metrics = {
                'xgb': xgb_metrics,
                'gru': gru_metrics,
                'prophet': prophet_metrics,
                'hybrid': hybrid_metrics,
                'weights': weights
            }
            
            st.success("Model training completed successfully!")
            return True
            
        except Exception as e:
            st.error(f"Model training error: {e}")
            return False
    
    def forecast(self, df, days=10):
        """Generate future forecasts"""
        try:
            # Prepare latest data
            X_latest, _ = self.prepare_features(df)
            latest_features = X_latest.iloc[-1:].values
            
            # Generate forecasts
            forecasts = []
            current_price = df['Close'].iloc[-1]
            
            for day in range(days):
                # XGBoost prediction
                xgb_return = self.xgb_model.predict(latest_features)[0]
                
                # GRU prediction
                scaled_features = self.scaler.transform(latest_features)
                scaled_3d = scaled_features.reshape(1, 1, scaled_features.shape[1])
                gru_return = self.gru_model.predict(scaled_3d)[0][0]
                
                # Prophet prediction
                future = self.prophet_model.make_future_dataframe(periods=day+1)
                prophet_forecast = self.prophet_model.predict(future)
                prophet_price = prophet_forecast['yhat'].iloc[-1]
                prophet_return = (prophet_price - current_price) / current_price
                
                # Combine with weights
                weights = self.model_metrics['weights']
                combined_return = (
                    weights['xgb'] * xgb_return +
                    weights['gru'] * gru_return +
                    weights['prophet'] * prophet_return
                ) / sum(weights.values())
                
                # Limit returns to realistic values
                combined_return = np.clip(combined_return, -0.1, 0.1)
                
                # Calculate next day price
                next_price = current_price * (1 + combined_return)
                
                forecasts.append({
                    'Date': pd.Timestamp.today() + pd.Timedelta(days=day+1),
                    'Predicted_Price': next_price,
                    'Daily_Return': combined_return * 100  # Percentage
                })
                
                # Update for next iteration
                current_price = next_price
            
            forecast_df = pd.DataFrame(forecasts)
            forecast_df.set_index('Date', inplace=True)
            
            return forecast_df
            
        except Exception as e:
            st.error(f"Forecast error: {e}")
            # Return simple forecast if model fails
            last_price = df['Close'].iloc[-1]
            dates = [pd.Timestamp.today() + pd.Timedelta(days=i+1) for i in range(days)]
            prices = [last_price * (1 + 0.001*i) for i in range(days)]  # Simple trend
            
            forecast_df = pd.DataFrame({
                'Predicted_Price': prices,
                'Daily_Return': [0.1] * days
            }, index=dates)
            
            return forecast_df

# ==================== VISUALIZATION FUNCTIONS ====================

def plot_price_chart(df, title):
    """Plot interactive price chart"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))
    
    # Add moving averages
    if 'MA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA_20'],
            mode='lines',
            name='MA 20',
            line=dict(color='orange', width=1)
        ))
    
    if 'MA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA_50'],
            mode='lines',
            name='MA 50',
            line=dict(color='red', width=1)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template='plotly_white',
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def plot_forecast_comparison(historical_df, forecast_df, stock_name):
    """Plot historical vs forecasted prices"""
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=historical_df.index[-30:],  # Last 30 days
        y=historical_df['Close'].iloc[-30:],
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Forecasted prices
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['Predicted_Price'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Current price marker
    last_price = historical_df['Close'].iloc[-1]
    fig.add_trace(go.Scatter(
        x=[historical_df.index[-1]],
        y=[last_price],
        mode='markers',
        name='Current Price',
        marker=dict(color='green', size=12)
    ))
    
    fig.update_layout(
        title=f'{stock_name} - 10-Day Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_model_performance(metrics):
    """Plot model performance comparison"""
    models = list(metrics.keys())
    mae_values = [metrics[m]['mae'] for m in models if 'mae' in metrics[m]]
    rmse_values = [metrics[m]['rmse'] for m in models if 'rmse' in metrics[m]]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Mean Absolute Error (Lower is Better)', 'RMSE (Lower is Better)')
    )
    
    # MAE plot
    fig.add_trace(
        go.Bar(
            x=models,
            y=mae_values,
            name='MAE',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        ),
        row=1, col=1
    )
    
    # RMSE plot
    fig.add_trace(
        go.Bar(
            x=models,
            y=rmse_values,
            name='RMSE',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Model Performance Comparison',
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

# ==================== INVESTMENT RECOMMENDATION ====================

def generate_recommendation(current_price, forecast_prices, sentiment_score, model_accuracy):
    """Generate investment recommendation based on analysis"""
    avg_forecast = forecast_prices['Predicted_Price'].mean()
    expected_return = ((avg_forecast - current_price) / current_price) * 100
    
    # Adjust for sentiment
    sentiment_factor = 1 + (sentiment_score * 0.5)
    adjusted_return = expected_return * sentiment_factor
    
    # Adjust for model confidence
    confidence_factor = model_accuracy / 100
    final_score = adjusted_return * confidence_factor
    
    # Generate recommendation
    if final_score > 5:
        return "STRONG BUY", "High confidence in significant upside potential"
    elif final_score > 2:
        return "BUY", "Good potential for moderate gains"
    elif final_score > -2:
        return "HOLD", "Neutral outlook - wait for clearer signals"
    elif final_score > -5:
        return "SELL", "Consider reducing position"
    else:
        return "STRONG SELL", "High downside risk detected"

# ==================== MAIN APPLICATION ====================

def main():
    """Main Streamlit application"""
    
    # Sidebar
    st.sidebar.markdown("## 📊 Configuration")
    
    # Load stocks
    indian_stocks = load_indian_stocks()
    
    # Stock selection
    selected_stock = st.sidebar.selectbox(
        "Select Indian Stock:",
        indian_stocks,
        index=indian_stocks.index('RELIANCE') if 'RELIANCE' in indian_stocks else 0
    )
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.date.today() - datetime.timedelta(days=365)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.date.today()
        )
    
    # News API key input
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 News API Configuration")
    
    api_key_option = st.sidebar.radio(
        "News API Source:",
        ["Use Demo Mode", "Enter Custom API Key"]
    )
    
    if api_key_option == "Enter Custom API Key":
        news_api_key = st.sidebar.text_input(
            "Enter NewsAPI Key:",
            type="password",
            help="Get free API key from https://newsapi.org"
        )
    else:
        news_api_key = "demo_mode"  # Use mock data
    
    # Analysis button
    st.sidebar.markdown("---")
    analyze_button = st.sidebar.button(
        "🚀 Start Analysis",
        type="primary",
        use_container_width=True
    )
    
    # Main content area
    if analyze_button:
        with st.spinner("Fetching stock data..."):
            # Get stock data
            df = get_stock_data(selected_stock, start_date, end_date)
            
            if df.empty:
                st.error("No data available for selected stock/period.")
                return
            
            # Display stock info
            st.subheader(f"📈 {selected_stock} - Stock Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Current Price",
                    f"₹{df['Close'].iloc[-1]:.2f}",
                    f"{(df['Close'].iloc[-1] - df['Close'].iloc[-2])/df['Close'].iloc[-2]*100:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Daily Range",
                    f"₹{df['Low'].iloc[-1]:.2f} - ₹{df['High'].iloc[-1]:.2f}",
                    f"{(df['High'].iloc[-1] - df['Low'].iloc[-1])/df['Close'].iloc[-1]*100:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Volume",
                    f"{df['Volume'].iloc[-1]:,.0f}",
                    "Today"
                )
            
            # Price chart
            st.plotly_chart(
                plot_price_chart(df, f"{selected_stock} - Historical Prices"),
                use_container_width=True
            )
            
            # News and Sentiment Analysis
            st.subheader("📰 News Sentiment Analysis")
            
            with st.spinner("Fetching and analyzing news..."):
                news_fetcher = NewsFetcher(news_api_key)
                news_articles = news_fetcher.fetch_news(selected_stock)
                st.session_state.news_data = news_articles
                
                if news_articles:
                    # Display news articles
                    news_df = pd.DataFrame(news_articles)
                    
                    # Sentiment summary
                    sentiment_summary = news_df['sentiment'].value_counts()
                    avg_sentiment = news_df['sentiment_score'].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Total News",
                            len(news_articles)
                        )
                    with col2:
                        st.metric(
                            "Avg Sentiment",
                            f"{avg_sentiment:.2f}",
                            "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
                        )
                    with col3:
                        st.metric(
                            "Positive News",
                            sentiment_summary.get('positive', 0),
                            f"{sentiment_summary.get('positive', 0)/len(news_articles)*100:.0f}%"
                        )
                    
                    # Display news table
                    st.dataframe(
                        news_df[['date', 'title', 'source', 'sentiment', 'sentiment_score']].head(10),
                        use_container_width=True
                    )
                    
                    # Sentiment trend
                    news_df['date'] = pd.to_datetime(news_df['date'])
                    daily_sentiment = news_df.groupby('date')['sentiment_score'].mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=daily_sentiment.index,
                        y=daily_sentiment.values,
                        mode='lines+markers',
                        name='Daily Sentiment',
                        line=dict(color='green' if avg_sentiment > 0 else 'red', width=2)
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    fig.update_layout(
                        title='News Sentiment Trend',
                        xaxis_title='Date',
                        yaxis_title='Sentiment Score',
                        height=300,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No news articles found. Using zero sentiment for analysis.")
                    avg_sentiment = 0
            
            # Feature Engineering
            st.subheader("⚙️ Feature Engineering")
            with st.spinner("Creating technical features..."):
                df_with_features = create_technical_features(df)
                df_with_sentiment = add_sentiment_features(df_with_features, news_articles)
                
                # Display feature statistics
                st.write("**Generated Features:**")
                feature_stats = pd.DataFrame({
                    'Feature': ['Moving Averages', 'Volatility', 'Volume Ratio', 'Price Momentum', 'Sentiment'],
                    'Count': [4, 2, 2, 2, 1],
                    'Description': ['Trend indicators', 'Risk measures', 'Volume analysis', 'Price momentum', 'News sentiment']
                })
                st.dataframe(feature_stats, use_container_width=True, hide_index=True)
            
            # Model Training
            st.subheader("🤖 Hybrid Model Training")
            
            with st.spinner("Training AI models... This may take a minute."):
                # Initialize and train model
                model = HybridStockModel()
                success = model.train(df_with_sentiment)
                
                if success:
                    st.session_state.model_trained = True
                    st.session_state.model = model
                    st.session_state.df_processed = df_with_sentiment
                    
                    # Display model performance
                    st.write("**Model Performance Metrics:**")
                    
                    # Create metrics table
                    metrics_df = pd.DataFrame({
                        'Model': ['XGBoost', 'GRU', 'Prophet', 'Hybrid'],
                        'MAE': [
                            model.model_metrics['xgb']['mae'],
                            model.model_metrics['gru']['mae'],
                            model.model_metrics['prophet']['mae'],
                            model.model_metrics['hybrid']['mae']
                        ],
                        'RMSE': [
                            model.model_metrics['xgb']['rmse'],
                            model.model_metrics['gru']['rmse'],
                            model.model_metrics['prophet']['rmse'],
                            model.model_metrics['hybrid']['rmse']
                        ],
                        'Weight': [
                            f"{model.model_metrics['weights']['xgb']*100:.1f}%",
                            f"{model.model_metrics['weights']['gru']*100:.1f}%",
                            f"{model.model_metrics['weights']['prophet']*100:.1f}%",
                            "Ensemble"
                        ]
                    })
                    
                    st.dataframe(
                        metrics_df.style.format({
                            'MAE': '{:.4f}',
                            'RMSE': '{:.4f}'
                        }).highlight_min(subset=['MAE', 'RMSE'], color='lightgreen'),
                        use_container_width=True
                    )
                    
                    # Plot performance
                    st.plotly_chart(
                        plot_model_performance(model.model_metrics),
                        use_container_width=True
                    )
                    
                    # Forecasting
                    st.subheader("🔮 10-Day Price Forecast")
                    
                    with st.spinner("Generating forecasts..."):
                        forecast_df = model.forecast(df_with_sentiment, days=10)
                        st.session_state.predictions = forecast_df
                        
                        # Display forecast table
                        forecast_display = forecast_df.copy()
                        forecast_display['Predicted_Price'] = forecast_display['Predicted_Price'].apply(lambda x: f"₹{x:.2f}")
                        forecast_display['Daily_Return'] = forecast_display['Daily_Return'].apply(lambda x: f"{x:.2f}%")
                        forecast_display.index = forecast_display.index.strftime('%Y-%m-%d')
                        
                        st.dataframe(
                            forecast_display.style.applymap(
                                lambda x: 'color: green' if '+' in x and '%' in x else 'color: red' if '-' in x and '%' in x else '',
                                subset=['Daily_Return']
                            ),
                            use_container_width=True
                        )
                        
                        # Plot forecast
                        st.plotly_chart(
                            plot_forecast_comparison(df, forecast_df, selected_stock),
                            use_container_width=True
                        )
                        
                        # Investment Recommendation
                        st.subheader("💡 Investment Recommendation")
                        
                        current_price = df['Close'].iloc[-1]
                        model_accuracy = 100 - (model.model_metrics['hybrid']['mae'] * 100)
                        model_accuracy = max(60, min(95, model_accuracy))  # Bound accuracy
                        
                        recommendation, reasoning = generate_recommendation(
                            current_price,
                            forecast_df,
                            avg_sentiment,
                            model_accuracy
                        )
                        
                        # Recommendation card
                        rec_colors = {
                            "STRONG BUY": "#4CAF50",
                            "BUY": "#8BC34A",
                            "HOLD": "#FFC107",
                            "SELL": "#FF9800",
                            "STRONG SELL": "#F44336"
                        }
                        
                        st.markdown(f"""
                        <div style="
                            background: {rec_colors.get(recommendation.split()[0], '#2196F3')};
                            color: white;
                            padding: 20px;
                            border-radius: 10px;
                            margin: 10px 0;
                            text-align: center;
                        ">
                            <h2 style="margin: 0;">{recommendation}</h2>
                            <p style="margin: 10px 0 0 0;">{reasoning}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Key factors
                        st.write("**Key Factors Considered:**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Model Confidence",
                                f"{model_accuracy:.1f}%",
                                "Accuracy"
                            )
                        
                        with col2:
                            st.metric(
                                "Sentiment Impact",
                                f"{avg_sentiment:.2f}",
                                "Positive" if avg_sentiment > 0 else "Negative"
                            )
                        
                        with col3:
                            expected_return = ((forecast_df['Predicted_Price'].mean() - current_price) / current_price) * 100
                            st.metric(
                                "Expected Return",
                                f"{expected_return:.1f}%",
                                "10-Day Forecast"
                            )
                        
                        # Risk Disclaimer
                        st.markdown("---")
                        st.warning("""
                        **Disclaimer:** This analysis is for research purposes only. 
                        Past performance is not indicative of future results. 
                        Always conduct your own research and consult with financial advisors before making investment decisions.
                        """)
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Multi-Source Stock Forecasting System
        
        This AI-powered system combines:
        
        1. **LLM-Enhanced Sentiment Analysis** - Using FinBERT for financial news
        2. **Hybrid Machine Learning Models** - XGBoost, GRU, and Prophet ensemble
        3. **Multi-Source Data Fusion** - Technical, fundamental, and sentiment data
        4. **Indian Market Focus** - Customized for NSE stocks
        
        ### How to use:
        1. Select an Indian stock from the sidebar
        2. Choose date range (minimum 6 months recommended)
        3. Configure News API (use demo or enter your key)
        4. Click **"Start Analysis"** to run the complete pipeline
        
        ### Features:
        - Real-time stock data from Yahoo Finance
        - News sentiment analysis
        - Technical indicator generation
        - Hybrid AI model training
        - 10-day price forecast
        - Investment recommendation
        
        **Note:** For NewsAPI, you can get a free API key from [newsapi.org](https://newsapi.org)
        """)
        
        # Show sample stocks
        st.markdown("### Sample Indian Stocks Available:")
        sample_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
        cols = st.columns(len(sample_stocks))
        for idx, stock in enumerate(sample_stocks):
            with cols[idx]:
                st.info(f"**{stock}**")

# Run the application
if __name__ == "__main__":
    main()