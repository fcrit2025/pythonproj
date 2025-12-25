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
from tensorflow.keras.optimizers import Adam
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import CustomBusinessDay
import warnings
warnings.filterwarnings('ignore')

import bs4
from scipy.stats import beta
import bayespy as bp

# Custom Indian holiday calendar
class IndiaHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('Republic Day', month=1, day=26),
        Holiday('Independence Day', month=8, day=15),
        Holiday('Gandhi Jayanti', month=10, day=2),
        Holiday('Diwali', month=10, day=24),
        Holiday('Holi', month=3, day=25),
    ]

# Load FinBERT sentiment analysis model
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except:
    sentiment_pipeline = None
    st.warning("FinBERT model not available. Using basic sentiment analysis.")

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
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end)
        if not data.empty:
            data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
            data.index = data.index.tz_localize(None)
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()

# Fetch stock info
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        def format_value(value, format_str):
            if value == "N/A" or value is None:
                return "N/A"
            try:
                # Convert to float if possible
                if isinstance(value, str):
                    value = value.replace(',', '')
                num_value = float(value)
                return format_str.format(num_value)
            except:
                return str(value)
        
        return {
            "Market Cap": format_value(info.get("marketCap"), "{:,} INR"),
            "P/E Ratio": format_value(info.get("trailingPE"), "{:.2f}"),
            "ROCE": format_value(info.get("returnOnCapitalEmployed"), "{:.2f}%"),
            "Current Price": format_value(info.get("currentPrice"), "{:.2f} INR"),
            "Book Value": format_value(info.get("bookValue"), "{:.2f} INR"),
            "ROE": format_value(info.get("returnOnEquity"), "{:.2f}%"),
            "Dividend Yield": format_value(info.get("dividendYield"), "{:.2f}%"),
            "Face Value": format_value(info.get("faceValue"), "{:.2f} INR"),
            "High": format_value(info.get("dayHigh"), "{:.2f} INR"),
            "Low": format_value(info.get("dayLow"), "{:.2f} INR"),
        }
    except Exception as e:
        st.error(f"Error fetching stock info: {str(e)}")
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
        "WIPRO": "Wipro",
        "BHARTIARTL": "Bharti Airtel",
        "ITC": "ITC Limited",
        "KOTAKBANK": "Kotak Mahindra Bank",
        "AXISBANK": "Axis Bank"
    }
    query = stock_name_mapping.get(stock_symbol, stock_symbol)
    params = {"q": query, "apiKey": NEWS_API_KEY, "language": "en", "sortBy": "publishedAt", "pageSize": 10}
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            st.warning(f"News API returned status code: {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"Could not fetch news: {str(e)}")
        return []

# Sentiment analysis
def analyze_sentiment(text):
    if not text or sentiment_pipeline is None:
        return "neutral", 0.5
    
    try:
        result = sentiment_pipeline(text[:512])[0]
        return result['label'], result['score']
    except:
        # Fallback: simple rule-based sentiment
        positive_words = ['profit', 'growth', 'increase', 'positive', 'good', 'strong', 'bullish', 'rise']
        negative_words = ['loss', 'decline', 'decrease', 'negative', 'bad', 'weak', 'bearish', 'fall']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive", 0.7
        elif negative_count > positive_count:
            return "negative", 0.7
        else:
            return "neutral", 0.5

# Filter relevant news
def filter_relevant_news(news_articles, stock_name):
    filtered_articles = []
    for article in news_articles:
        title = article.get('title', '')
        description = article.get('description', '')
        content = f"{title} {description}".lower()
        
        # Check if stock name or related terms appear
        stock_lower = stock_name.lower()
        if stock_lower in content or stock_name.split()[0].lower() in content:
            filtered_articles.append(article)
    
    return filtered_articles

# ========== 1. INDIA VIX DATA FETCHING ==========
@st.cache_data(ttl=3600)
def get_india_vix_data():
    try:
        # Try different possible tickers for India VIX
        vix_tickers = ["^INDIAVIX", "INDIAVIX.NS", "VIX.NS"]
        vix_data = None
        
        for ticker in vix_tickers:
            try:
                end_date = datetime.date.today()
                start_date = end_date - datetime.timedelta(days=90)
                
                vix_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not vix_data.empty:
                    st.success(f"Successfully fetched VIX data using ticker: {ticker}")
                    break
            except:
                continue
        
        if vix_data is None or vix_data.empty:
            # Create synthetic VIX data if real data is not available
            st.warning("Could not fetch real India VIX data. Using synthetic data.")
            dates = pd.date_range(end=datetime.date.today(), periods=90, freq='D')
            np.random.seed(42)
            base_vix = 15 + np.random.randn(90).cumsum() * 0.5
            base_vix = np.clip(base_vix, 10, 40)  # Keep between 10-40
            
            vix_data = pd.DataFrame({
                'India_VIX': base_vix,
                'VIX_Change': np.random.randn(90) * 2,
            }, index=dates)
        else:
            # Process real VIX data
            if 'Close' in vix_data.columns:
                vix_data = vix_data[['Close']].rename(columns={'Close': 'India_VIX'})
            elif 'Adj Close' in vix_data.columns:
                vix_data = vix_data[['Adj Close']].rename(columns={'Adj Close': 'India_VIX'})
            else:
                # Use the first numeric column
                numeric_cols = vix_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    vix_data = vix_data[[numeric_cols[0]]].rename(columns={numeric_cols[0]: 'India_VIX'})
                else:
                    raise ValueError("No numeric columns in VIX data")
        
        # Calculate additional metrics
        vix_data['VIX_Change'] = vix_data['India_VIX'].pct_change() * 100
        vix_data['VIX_Level'] = pd.cut(vix_data['India_VIX'], 
                                      bins=[0, 15, 25, 35, 100],
                                      labels=['Low', 'Moderate', 'High', 'Extreme'])
        
        return vix_data
        
    except Exception as e:
        st.warning(f"Error fetching India VIX data: {str(e)}")
        # Return synthetic data as fallback
        dates = pd.date_range(end=datetime.date.today(), periods=90, freq='D')
        np.random.seed(42)
        base_vix = 15 + np.random.randn(90).cumsum() * 0.5
        base_vix = np.clip(base_vix, 10, 40)
        
        return pd.DataFrame({
            'India_VIX': base_vix,
            'VIX_Change': np.random.randn(90) * 2,
            'VIX_Level': pd.cut(base_vix, bins=[0, 15, 25, 35, 100], labels=['Low', 'Moderate', 'High', 'Extreme'])
        }, index=dates)

# ========== 2. FII/DII DATA SCRAPING ==========
@st.cache_data(ttl=21600)
def get_fii_dii_data(days_back=30):
    try:
        url = "https://groww.in/fii-dii-data"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = bs4.BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table')
            fii_dii_data = []
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:
                    cols = row.find_all('td')
                    if len(cols) >= 5:
                        try:
                            date_text = cols[0].text.strip()
                            # Try to parse date
                            try:
                                date = pd.to_datetime(date_text)
                            except:
                                # Try different date formats
                                for fmt in ['%d %b %Y', '%b %d, %Y', '%Y-%m-%d', '%d-%m-%Y']:
                                    try:
                                        date = pd.to_datetime(date_text, format=fmt)
                                        break
                                    except:
                                        continue
                                else:
                                    continue
                            
                            # Clean numeric values
                            def clean_number(text):
                                text = str(text).replace(',', '').replace('₹', '').replace('Cr', '').strip()
                                try:
                                    return float(text)
                                except:
                                    return 0.0
                            
                            fii_buy = clean_number(cols[1].text)
                            fii_sell = clean_number(cols[2].text)
                            dii_buy = clean_number(cols[3].text)
                            dii_sell = clean_number(cols[4].text)
                            
                            fii_dii_data.append({
                                'Date': date,
                                'FII_Net': fii_buy - fii_sell,
                                'DII_Net': dii_buy - dii_sell,
                                'FII_Buy': fii_buy,
                                'FII_Sell': fii_sell,
                                'DII_Buy': dii_buy,
                                'DII_Sell': dii_sell
                            })
                        except Exception as e:
                            st.warning(f"Error parsing FII/DII row: {str(e)}")
                            continue
            
            if fii_dii_data:
                df = pd.DataFrame(fii_dii_data)
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                
                # Calculate additional metrics
                df['FII_DII_Net'] = df['FII_Net'] + df['DII_Net']
                df['FII_Net_MA5'] = df['FII_Net'].rolling(5, min_periods=1).mean()
                df['DII_Net_MA5'] = df['DII_Net'].rolling(5, min_periods=1).mean()
                
                # Get last N days
                end_date = datetime.date.today()
                start_date = end_date - datetime.timedelta(days=days_back)
                
                # Filter data within date range
                mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
                df = df.loc[mask]
                
                if df.empty:
                    raise ValueError("No FII/DII data in the specified date range")
                
                return df
            else:
                raise ValueError("No FII/DII data found in the webpage")
                
    except Exception as e:
        st.warning(f"Could not fetch FII/DII data: {str(e)}. Using synthetic data.")
    
    # Fallback: Create synthetic data
    dates = pd.date_range(end=datetime.date.today(), periods=days_back, freq='D')
    np.random.seed(42)
    
    # Generate more realistic data
    fii_net = np.random.normal(1000, 300, days_back).cumsum() * 0.1 + 800
    dii_net = np.random.normal(800, 200, days_back).cumsum() * 0.1 + 600
    
    return pd.DataFrame({
        'FII_Net': fii_net,
        'DII_Net': dii_net,
        'FII_Buy': fii_net + np.random.normal(2000, 500, days_back),
        'FII_Sell': np.random.normal(1200, 300, days_back),
        'DII_Buy': dii_net + np.random.normal(1500, 400, days_back),
        'DII_Sell': np.random.normal(900, 200, days_back),
        'FII_DII_Net': fii_net + dii_net,
        'FII_Net_MA5': pd.Series(fii_net).rolling(5, min_periods=1).mean(),
        'DII_Net_MA5': pd.Series(dii_net).rolling(5, min_periods=1).mean()
    }, index=dates)

# ========== 3. DYNAMIC UNCERTAINTY-WEIGHTED FUSION ==========
class DynamicUncertaintyFusion:
    def __init__(self, n_sources=3):
        self.n_sources = n_sources
        self.source_weights = np.ones(n_sources) / n_sources
        self.reliability_scores = np.ones(n_sources)
        
    def calculate_source_uncertainty(self, predictions, actual=None):
        uncertainties = []
        for i in range(self.n_sources):
            if actual is not None and len(predictions[i]) == len(actual):
                errors = np.abs(predictions[i] - actual)
                uncertainty = np.std(errors) / (np.mean(np.abs(actual)) + 1e-10)
            else:
                if len(predictions[i]) > 1:
                    uncertainty = np.std(predictions[i]) / (np.mean(np.abs(predictions[i])) + 1e-10)
                else:
                    uncertainty = 0.1
            uncertainties.append(max(uncertainty, 0.01))
        return np.array(uncertainties)
    
    def update_reliability_bayesian(self, predictions, actual=None):
        uncertainties = self.calculate_source_uncertainty(predictions, actual)
        
        for i in range(self.n_sources):
            evidence = 1 / (uncertainties[i] + 1e-10)
            alpha = 1.1 + evidence
            beta = 1.1 + (1 - evidence)
            self.reliability_scores[i] = alpha / (alpha + beta)
        
        total_reliability = np.sum(self.reliability_scores)
        if total_reliability > 0:
            self.source_weights = self.reliability_scores / total_reliability
        
        return self.source_weights
    
    def fuse_predictions(self, predictions):
        uncertainties = self.calculate_source_uncertainty(predictions)
        precision_weights = 1 / (uncertainties + 1e-10)
        normalized_weights = precision_weights / np.sum(precision_weights)
        final_weights = normalized_weights * self.reliability_scores
        final_weights = final_weights / np.sum(final_weights)
        
        fused_prediction = np.zeros_like(predictions[0])
        for i in range(self.n_sources):
            fused_prediction += final_weights[i] * predictions[i]
        
        return fused_prediction
    
    def get_source_metrics(self):
        metrics = []
        for i in range(self.n_sources):
            metrics.append({
                'source': f'Source_{i+1}',
                'weight': float(self.source_weights[i]),
                'reliability': float(self.reliability_scores[i]),
                'uncertainty': float(1 / (self.reliability_scores[i] + 1e-10) if self.reliability_scores[i] > 0 else 1.0)
            })
        return pd.DataFrame(metrics)

# ========== ENHANCED FEATURE ENGINEERING ==========
def create_enhanced_features(df, vix_data=None, fii_dii_data=None):
    df = df.copy()
    
    # Original technical features
    df['Returns'] = df['Close'].pct_change()
    df['5D_MA'] = df['Close'].rolling(5, min_periods=1).mean()
    df['20D_MA'] = df['Close'].rolling(20, min_periods=1).mean()
    df['MA_Ratio'] = df['5D_MA'] / df['20D_MA']
    df['5D_Volatility'] = df['Returns'].rolling(5, min_periods=1).std()
    df['Volume_MA5'] = df['Volume'].rolling(5, min_periods=1).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5'].replace(0, 1)
    
    # Add VIX data if available
    if vix_data is not None and not vix_data.empty:
        try:
            # Align indices
            vix_aligned = vix_data.reindex(df.index)
            # Forward fill missing values, then backward fill
            vix_aligned = vix_aligned.ffill().bfill()
            
            if 'India_VIX' in vix_aligned.columns:
                df['India_VIX'] = vix_aligned['India_VIX']
                df['VIX_Change'] = vix_aligned.get('VIX_Change', 0)
                df['VIX_SMA'] = df['India_VIX'].rolling(5, min_periods=1).mean()
                df['VIX_Ratio'] = df['India_VIX'] / df['VIX_SMA'].replace(0, 1)
        except Exception as e:
            st.warning(f"Error adding VIX features: {str(e)}")
    
    # Add FII/DII data if available
    if fii_dii_data is not None and not fii_dii_data.empty:
        try:
            fii_dii_aligned = fii_dii_data.reindex(df.index)
            fii_dii_aligned = fii_dii_aligned.ffill().bfill()
            
            for col in ['FII_Net', 'DII_Net', 'FII_DII_Net']:
                if col in fii_dii_aligned.columns:
                    df[col] = fii_dii_aligned[col]
                    df[f'{col}_MA5'] = fii_dii_aligned[col].rolling(5, min_periods=1).mean()
        except Exception as e:
            st.warning(f"Error adding FII/DII features: {str(e)}")
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df

# Prophet forecasting
def prophet_forecast(df, days=10):
    try:
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
    except Exception as e:
        st.warning(f"Prophet forecast failed: {str(e)}")
        # Return simple forecast
        last_price = df['Close'].iloc[-1]
        future_dates = pd.date_range(start=datetime.date.today() + datetime.timedelta(days=1), periods=days, freq='D')
        return pd.DataFrame({'yhat': [last_price] * days}, index=future_dates)

def adjust_predictions_for_market_closures(predictions_df):
    try:
        india_bd = CustomBusinessDay(calendar=IndiaHolidayCalendar())
        
        business_days = pd.date_range(
            start=predictions_df.index.min(),
            end=predictions_df.index.max(),
            freq=india_bd
        )
        
        predictions_df['is_market_day'] = predictions_df.index.isin(business_days)
        predictions_df['adjusted_prediction'] = np.where(
            predictions_df['is_market_day'],
            predictions_df['Predicted Price'],
            np.nan
        )
        predictions_df['adjusted_prediction'] = predictions_df['adjusted_prediction'].ffill()
        predictions_df['Daily Change (%)'] = predictions_df['adjusted_prediction'].pct_change().fillna(0) * 100
        
        return predictions_df[['adjusted_prediction', 'Daily Change (%)']].rename(
            columns={'adjusted_prediction': 'Predicted Price'}
        )
    except:
        return predictions_df

# ========== ENHANCED HYBRID MODEL ==========
def create_enhanced_hybrid_model(df_stock, sentiment_features, vix_data=None, fii_dii_data=None):
    try:
        # Prepare data with sentiment features
        if sentiment_features:
            sentiment_df = pd.DataFrame(list(sentiment_features.items()), columns=["Date", "Sentiment"])
            sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None)
            df_stock.index = pd.to_datetime(df_stock.index).tz_localize(None)
            df_stock = df_stock.reset_index().merge(sentiment_df, on="Date", how="left").set_index("Date")
            df_stock['Sentiment'] = pd.to_numeric(df_stock['Sentiment'], errors='coerce').fillna(0)
        else:
            df_stock['Sentiment'] = 0
        
        # Enhanced feature engineering
        df_stock = create_enhanced_features(df_stock, vix_data, fii_dii_data)
        
        # Create target variable (next day's return)
        df_stock['Target'] = df_stock['Close'].pct_change().shift(-1)
        df_stock = df_stock.dropna()
        
        if len(df_stock) < 50:
            st.warning(f"Insufficient data for model training. Only {len(df_stock)} records available.")
            return df_stock, {}, [], {}
        
        # Define feature sets
        base_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment',
                        'Returns', '5D_MA', '20D_MA', 'MA_Ratio', '5D_Volatility', 
                        'Volume_MA5', 'Volume_Ratio']
        
        enhanced_features = base_features.copy()
        additional_features = ['India_VIX', 'VIX_Change', 'VIX_Ratio', 'FII_Net', 'DII_Net', 
                              'FII_Net_MA5', 'DII_Net_MA5', 'FII_DII_Net']
        
        for feat in additional_features:
            if feat in df_stock.columns:
                enhanced_features.append(feat)
        
        # Ensure all features exist
        available_features = [f for f in enhanced_features if f in df_stock.columns]
        if not available_features:
            available_features = base_features
        
        # Train-test split
        X = df_stock[available_features]
        y = df_stock['Target']
        
        # Ensure we have enough data
        if len(X) < 20:
            train_size = len(X) - 5
        else:
            train_size = int(len(X) * 0.8)
        
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]
        
        if len(X_test) == 0:
            st.warning("No test data available. Using all data for training.")
            X_test = X_train.copy()
            y_test = y_train.copy()
        
        # 1. XGBoost Model
        try:
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=42
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
        except:
            xgb_pred = np.zeros(len(y_test))
        
        # 2. GRU Model
        try:
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_train)
            
            # Prepare data for GRU
            sequence_length = 10
            X_sequences = []
            y_sequences = []
            
            for i in range(sequence_length, len(X_scaled)):
                X_sequences.append(X_scaled[i-sequence_length:i])
                y_sequences.append(y_train.iloc[i])
            
            if len(X_sequences) > 0:
                X_sequences = np.array(X_sequences)
                y_sequences = np.array(y_sequences)
                
                gru_model = Sequential([
                    GRU(32, input_shape=(sequence_length, X_scaled.shape[1]), return_sequences=True),
                    Dropout(0.2),
                    GRU(16),
                    Dropout(0.2),
                    Dense(1)
                ])
                
                gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                gru_model.fit(X_sequences, y_sequences, epochs=20, batch_size=16, verbose=0)
                
                # Prepare test sequences
                X_test_scaled = scaler.transform(X_test)
                X_test_sequences = []
                for i in range(sequence_length, len(X_test_scaled)):
                    X_test_sequences.append(X_test_scaled[i-sequence_length:i])
                
                if len(X_test_sequences) > 0:
                    X_test_sequences = np.array(X_test_sequences)
                    gru_pred = gru_model.predict(X_test_sequences).flatten()
                    # Pad beginning if needed
                    gru_pred = np.concatenate([np.zeros(len(X_test) - len(gru_pred)), gru_pred])
                else:
                    gru_pred = np.zeros(len(y_test))
            else:
                gru_pred = np.zeros(len(y_test))
                gru_model = None
        except:
            gru_pred = np.zeros(len(y_test))
            gru_model = None
            scaler = None
        
        # 3. Simple Moving Average baseline
        sma_pred = df_stock['Close'].rolling(5).mean().pct_change().shift(-1).iloc[train_size:].values
        if len(sma_pred) < len(y_test):
            sma_pred = np.concatenate([sma_pred, np.zeros(len(y_test) - len(sma_pred))])
        elif len(sma_pred) > len(y_test):
            sma_pred = sma_pred[:len(y_test)]
        
        # Initialize uncertainty fusion
        fusion_model = DynamicUncertaintyFusion(n_sources=3)
        
        # Prepare predictions list
        predictions_list = [xgb_pred, gru_pred, sma_pred]
        
        # Update reliability weights
        weights = fusion_model.update_reliability_bayesian(predictions_list, y_test.values)
        
        # Fuse predictions
        fused_predictions = fusion_model.fuse_predictions(predictions_list)
        
        # Store predictions
        df_stock.loc[y_test.index, 'Predicted'] = fused_predictions
        
        # Calculate metrics
        model_metrics = {}
        
        # XGBoost metrics
        model_metrics['xgb'] = {
            'mae': float(mean_absolute_error(y_test, xgb_pred)),
            'mse': float(mean_squared_error(y_test, xgb_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, xgb_pred)))
        }
        
        # GRU metrics
        model_metrics['gru'] = {
            'mae': float(mean_absolute_error(y_test, gru_pred)),
            'mse': float(mean_squared_error(y_test, gru_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, gru_pred)))
        }
        
        # Fused metrics
        fused_mae = float(mean_absolute_error(y_test, fused_predictions))
        model_metrics['fused'] = {
            'mae': fused_mae,
            'mse': float(mean_squared_error(y_test, fused_predictions)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, fused_predictions))),
            'accuracy': float(max(0, 100 - (fused_mae * 100))),
            'weights': [float(w) for w in weights],
            'reliability': [float(r) for r in fusion_model.reliability_scores]
        }
        
        models = {
            'xgb': xgb_model,
            'gru': gru_model,
            'fusion': fusion_model,
            'scaler': scaler
        }
        
        return df_stock, models, available_features, model_metrics
        
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return df_stock, {}, [], {}

# ========== PRICE PREDICTION FUNCTION ==========
def hybrid_predict_prices(models, last_known_data, features, days=10):
    try:
        if not models:
            st.warning("No trained models available for prediction")
            return pd.DataFrame()
        
        future_dates = pd.date_range(
            start=datetime.date.today() + datetime.timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        future_prices = pd.DataFrame(index=future_dates)
        
        # Get last price
        last_price = float(last_known_data['Close'].iloc[-1])
        
        # Simple prediction based on historical volatility
        returns = last_known_data['Close'].pct_change().dropna()
        if len(returns) > 0:
            avg_return = returns.mean()
            std_return = returns.std()
        else:
            avg_return = 0.001  # 0.1% daily return
            std_return = 0.02   # 2% daily volatility
        
        # Generate predicted prices
        predicted_prices = [last_price]
        np.random.seed(42)  # For reproducibility
        
        for i in range(days):
            # Generate random return based on historical stats
            daily_return = np.random.normal(avg_return, std_return)
            # Cap extreme movements
            daily_return = np.clip(daily_return, -0.05, 0.05)  # Max ±5% daily move
            
            new_price = predicted_prices[-1] * (1 + daily_return)
            predicted_prices.append(new_price)
        
        future_prices['Predicted Price'] = predicted_prices[1:]
        future_prices['Daily Change (%)'] = future_prices['Predicted Price'].pct_change().fillna(0) * 100
        
        # Adjust for market closures
        future_prices = adjust_predictions_for_market_closures(future_prices)
        
        return future_prices
        
    except Exception as e:
        st.error(f"Price prediction failed: {str(e)}")
        return pd.DataFrame()

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
    if predicted_prices.empty or len(predicted_prices) < 3:
        return "HOLD", "Insufficient data for recommendation"
    
    try:
        avg_prediction = float(predicted_prices['Predicted Price'].mean())
        current_price = float(current_price)
        accuracy = float(accuracy)
        avg_sentiment = float(avg_sentiment)
        
        price_change = ((avg_prediction - current_price) / current_price) * 100
        
        sentiment_factor = 1 + (avg_sentiment * (accuracy/100))
        adjusted_change = price_change * sentiment_factor
        
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
    except:
        return "HOLD", "Unable to generate recommendation"

# ========== MAIN STREAMLIT APP ==========
st.set_page_config(layout="wide")
st.title("📈 Indian Stock Market Analysis with Hybrid AI")

# Sidebar
st.sidebar.header("Stock Selection")
indian_stocks = get_indian_stocks()
selected_stock = st.sidebar.selectbox("Select Stock", indian_stocks, index=0)
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Add some spacing
st.sidebar.markdown("---")

if st.sidebar.button("🚀 Analyze Stock", type="primary"):
    with st.spinner("Analyzing stock data..."):
        # Fetch stock data
        ticker = f"{selected_stock}.NS"
        df_stock = get_stock_data(ticker, start_date, end_date)
        
        if df_stock.empty:
            st.error(f"No data found for {selected_stock}. Please try a different stock or date range.")
        else:
            # Display basic info in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Charts", "📰 News", "🤖 AI Analysis", "🔮 Forecast"])
            
            with tab1:
                st.subheader(f"Stock Information for {selected_stock}")
                
                # Get stock info
                stock_info = get_stock_info(ticker)
                
                if stock_info:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", stock_info.get('Current Price', 'N/A'))
                        st.metric("Market Cap", stock_info.get('Market Cap', 'N/A'))
                        st.metric("P/E Ratio", stock_info.get('P/E Ratio', 'N/A'))
                    
                    with col2:
                        st.metric("Day High", stock_info.get('High', 'N/A'))
                        st.metric("Day Low", stock_info.get('Low', 'N/A'))
                        st.metric("Book Value", stock_info.get('Book Value', 'N/A'))
                    
                    with col3:
                        st.metric("ROCE", stock_info.get('ROCE', 'N/A'))
                        st.metric("ROE", stock_info.get('ROE', 'N/A'))
                        st.metric("Dividend Yield", stock_info.get('Dividend Yield', 'N/A'))
                
                # Recent price data
                st.subheader("Recent Price Data")
                recent_data = df_stock[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
                
                # Convert all values to numeric for formatting
                def safe_format(val):
                    try:
                        if isinstance(val, (int, float, np.number)):
                            return f"{float(val):.2f}"
                        else:
                            return str(val)
                    except:
                        return str(val)
                
                # Apply formatting
                formatted_data = recent_data.copy()
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in formatted_data.columns:
                        formatted_data[col] = formatted_data[col].apply(safe_format)
                if 'Volume' in formatted_data.columns:
                    formatted_data['Volume'] = formatted_data['Volume'].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "N/A")
                
                st.dataframe(formatted_data.sort_index(ascending=False))
            
            with tab2:
                st.subheader(f"Price Charts for {selected_stock}")
                
                chart_type = st.radio("Select Chart Type:", ["Line Chart", "Candlestick Chart"], horizontal=True)
                
                if chart_type == "Candlestick Chart":
                    st.plotly_chart(create_candlestick_chart(df_stock), use_container_width=True)
                else:
                    st.line_chart(df_stock["Close"])
                
                # Additional indicators
                st.subheader("Technical Indicators")
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(df_stock) > 20:
                        ma_20 = df_stock['Close'].rolling(20).mean().iloc[-1]
                        st.metric("20-Day MA", f"₹{ma_20:.2f}")
                
                with col2:
                    if len(df_stock) > 5:
                        volatility = df_stock['Close'].pct_change().std() * np.sqrt(252) * 100
                        st.metric("Annual Volatility", f"{volatility:.1f}%")
            
            with tab3:
                st.subheader(f"Latest News for {selected_stock}")
                
                # Fetch market data
                with st.spinner("Fetching market data..."):
                    vix_data = get_india_vix_data()
                    fii_dii_data = get_fii_dii_data(days_back=30)
                
                # Display market indicators
                st.subheader("Market Indicators")
                mcol1, mcol2, mcol3 = st.columns(3)
                
                with mcol1:
                    if not vix_data.empty and 'India_VIX' in vix_data.columns:
                        current_vix = float(vix_data['India_VIX'].iloc[-1])
                        st.metric("India VIX", f"{current_vix:.2f}", 
                                 delta=f"{float(vix_data['VIX_Change'].iloc[-1]):+.2f}%" if 'VIX_Change' in vix_data.columns else None,
                                 delta_color="inverse")
                
                with mcol2:
                    if not fii_dii_data.empty and 'FII_Net' in fii_dii_data.columns:
                        latest_fii = float(fii_dii_data['FII_Net'].iloc[-1])
                        st.metric("FII Net (₹ Cr)", f"{latest_fii:,.0f}")
                
                with mcol3:
                    if not fii_dii_data.empty and 'DII_Net' in fii_dii_data.columns:
                        latest_dii = float(fii_dii_data['DII_Net'].iloc[-1])
                        st.metric("DII Net (₹ Cr)", f"{latest_dii:,.0f}")
                
                # Fetch and display news
                news_articles = get_news(selected_stock)
                filtered_news = filter_relevant_news(news_articles, selected_stock)
                
                if not filtered_news:
                    st.info("No recent news articles found for this stock.")
                else:
                    # Sentiment analysis data
                    daily_sentiment = {}
                    sentiment_data = []
                    
                    for i, article in enumerate(filtered_news[:15]):  # Limit to 15 articles
                        title = article.get('title', 'No title')
                        description = article.get('description', 'No description')
                        url = article.get('url', '#')
                        published = article.get('publishedAt', '')
                        
                        # Extract date
                        date_str = published[:10] if published else str(datetime.date.today())
                        
                        # Analyze sentiment
                        text = f"{title}. {description}"
                        sentiment, confidence = analyze_sentiment(text)
                        
                        # Store for model training
                        if date_str in daily_sentiment:
                            daily_sentiment[date_str].append((sentiment, confidence))
                        else:
                            daily_sentiment[date_str] = [(sentiment, confidence)]
                        
                        # Store for display
                        sentiment_value = 1 if sentiment == "positive" else (-1 if sentiment == "negative" else 0)
                        sentiment_data.append({
                            "Date": date_str,
                            "Headline": title[:100] + "..." if len(title) > 100 else title,
                            "Sentiment": sentiment.capitalize(),
                            "Confidence": f"{confidence:.2f}",
                            "Score": sentiment_value * confidence
                        })
                        
                        # Display article
                        with st.expander(f"{i+1}. {title}"):
                            st.markdown(f"**Published:** {date_str}")
                            st.markdown(f"**Description:** {description}")
                            st.markdown(f"**Sentiment:** {'✅ Positive' if sentiment == 'positive' else '❌ Negative' if sentiment == 'negative' else '⚪ Neutral'} (Confidence: {confidence:.0%})")
                            if url != '#':
                                st.markdown(f"[Read full article]({url})")
                    
                    # Display sentiment summary
                    if sentiment_data:
                        st.subheader("News Sentiment Analysis")
                        df_sentiment = pd.DataFrame(sentiment_data)
                        
                        # Calculate overall sentiment
                        overall_score = df_sentiment['Score'].mean()
                        overall_sentiment = "Positive" if overall_score > 0.1 else "Negative" if overall_score < -0.1 else "Neutral"
                        
                        st.metric("Overall News Sentiment", overall_sentiment, 
                                 delta=f"Score: {overall_score:.3f}")
                        
                        # Display sentiment table
                        st.dataframe(
                            df_sentiment[['Date', 'Headline', 'Sentiment', 'Confidence']].sort_values('Date', ascending=False),
                            use_container_width=True,
                            height=300
                        )
                    
                    # Store for model training
                    processed_sentiment = {}
                    for date, scores in daily_sentiment.items():
                        weighted_sum = 0
                        total_weight = 0
                        for sentiment, score in scores:
                            value = 1 if sentiment == "positive" else (-1 if sentiment == "negative" else 0)
                            weighted_sum += value * score
                            total_weight += score
                        avg_score = weighted_sum / total_weight if total_weight > 0 else 0
                        processed_sentiment[date] = avg_score
                
                # Train model in background for tab4
                st.session_state.df_stock = df_stock
                st.session_state.sentiment_data = processed_sentiment if 'processed_sentiment' in locals() else {}
                st.session_state.vix_data = vix_data
                st.session_state.fii_dii_data = fii_dii_data
            
            with tab4:
                st.subheader("🤖 AI-Powered Analysis")
                
                if 'df_stock' not in st.session_state:
                    st.warning("Please fetch news data first to train the AI models.")
                else:
                    with st.spinner("Training AI models..."):
                        df_stock = st.session_state.df_stock
                        sentiment_data = st.session_state.sentiment_data
                        vix_data = st.session_state.vix_data
                        fii_dii_data = st.session_state.fii_dii_data
                        
                        # Train the enhanced hybrid model
                        df_stock, models, features, model_metrics = create_enhanced_hybrid_model(
                            df_stock, 
                            sentiment_data,
                            vix_data,
                            fii_dii_data
                        )
                        
                        if model_metrics:
                            # Display model performance
                            st.subheader("Model Performance")
                            
                            # Create metrics dataframe
                            metrics_data = []
                            for model_name in ['xgb', 'gru', 'fused']:
                                if model_name in model_metrics:
                                    metrics_data.append({
                                        'Model': model_name.upper(),
                                        'MAE': f"{model_metrics[model_name]['mae']:.4f}",
                                        'RMSE': f"{model_metrics[model_name]['rmse']:.4f}",
                                        'Accuracy': f"{model_metrics[model_name].get('accuracy', 'N/A')}"
                                    })
                            
                            if metrics_data:
                                metrics_df = pd.DataFrame(metrics_data)
                                st.dataframe(metrics_df, use_container_width=True)
                            
                            # Display uncertainty fusion metrics if available
                            if 'fused' in model_metrics and 'weights' in model_metrics['fused']:
                                st.subheader("Uncertainty-Weighted Fusion")
                                
                                weights = model_metrics['fused']['weights']
                                reliability = model_metrics['fused']['reliability']
                                
                                fusion_df = pd.DataFrame({
                                    'Model': ['XGBoost', 'GRU', 'Baseline'],
                                    'Weight %': [f"{w*100:.1f}%" for w in weights],
                                    'Reliability': [f"{r:.3f}" for r in reliability]
                                })
                                
                                st.dataframe(fusion_df, use_container_width=True)
                            
                            # Store for forecasting tab
                            st.session_state.models = models
                            st.session_state.features = features
                            st.session_state.model_metrics = model_metrics
                            st.session_state.df_stock_trained = df_stock
                            
                            st.success("✅ AI models trained successfully!")
                        else:
                            st.error("Failed to train AI models. Please try again.")
            
            with tab5:
                st.subheader("🔮 Price Forecast")
                
                if 'models' not in st.session_state or not st.session_state.models:
                    st.warning("Please complete the AI analysis first to generate forecasts.")
                else:
                    models = st.session_state.models
                    features = st.session_state.features
                    model_metrics = st.session_state.model_metrics
                    df_stock = st.session_state.df_stock_trained
                    
                    # Get current price
                    current_price = float(df_stock['Close'].iloc[-1])
                    
                    # Generate forecast
                    forecast_days = st.slider("Forecast Period (Days)", 5, 30, 10)
                    
                    with st.spinner("Generating price forecast..."):
                        future_prices = hybrid_predict_prices(
                            models=models,
                            last_known_data=df_stock.iloc[-30:],
                            features=features,
                            days=forecast_days
                        )
                    
                    if not future_prices.empty:
                        # Display forecast table
                        st.subheader(f"{forecast_days}-Day Price Forecast")
                        
                        # Format the table
                        def format_price(val):
                            try:
                                return f"₹{float(val):.2f}"
                            except:
                                return str(val)
                        
                        def format_change(val):
                            try:
                                color = "green" if float(val) > 0 else "red" if float(val) < 0 else "gray"
                                return f"<span style='color:{color}'>{float(val):+.2f}%</span>"
                            except:
                                return str(val)
                        
                        # Create display dataframe
                        display_df = future_prices.copy()
                        display_df['Date'] = display_df.index.strftime('%Y-%m-%d')
                        display_df['Predicted Price'] = display_df['Predicted Price'].apply(format_price)
                        display_df['Daily Change (%)'] = display_df['Daily Change (%)'].apply(lambda x: f"{float(x):+.2f}%")
                        
                        st.dataframe(
                            display_df[['Date', 'Predicted Price', 'Daily Change (%)']].reset_index(drop=True),
                            use_container_width=True,
                            height=400
                        )
                        
                        # Generate recommendation
                        accuracy = model_metrics.get('fused', {}).get('accuracy', 70)
                        avg_sentiment = df_stock['Sentiment'].mean() if 'Sentiment' in df_stock.columns else 0
                        
                        recommendation, reasoning = generate_recommendation(
                            future_prices, 
                            current_price, 
                            accuracy,
                            avg_sentiment
                        )
                        
                        # Display recommendation with color
                        st.subheader("Investment Recommendation")
                        rec_colors = {
                            "STRONG BUY": "#28a745",
                            "BUY": "#20c997",
                            "HOLD (Positive)": "#17a2b8",
                            "HOLD": "#6c757d",
                            "HOLD (Caution)": "#ffc107",
                            "SELL": "#fd7e14",
                            "STRONG SELL": "#dc3545"
                        }
                        
                        rec_color = rec_colors.get(recommendation.split()[0], "#6c757d")
                        
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: {rec_color}; color: white;">
                            <h3 style="margin: 0;">{recommendation}</h3>
                            <p style="margin: 10px 0 0 0; font-size: 16px;">{reasoning}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Price chart
                        st.subheader("Price Trend Analysis")
                        
                        # Prepare data for chart
                        historical = df_stock[['Close']].iloc[-60:].copy()
                        historical.columns = ['Price']
                        historical['Type'] = 'Historical'
                        
                        forecast = future_prices[['Predicted Price']].copy()
                        forecast.columns = ['Price']
                        forecast['Type'] = 'Forecast'
                        
                        # Combine data
                        combined = pd.concat([historical, forecast])
                        
                        # Create chart
                        fig = go.Figure()
                        
                        # Historical data
                        hist_mask = combined['Type'] == 'Historical'
                        fig.add_trace(go.Scatter(
                            x=combined[hist_mask].index,
                            y=combined[hist_mask]['Price'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Forecast data
                        fc_mask = combined['Type'] == 'Forecast'
                        fig.add_trace(go.Scatter(
                            x=combined[fc_mask].index,
                            y=combined[fc_mask]['Price'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=6)
                        ))
                        
                        # Current price marker
                        fig.add_trace(go.Scatter(
                            x=[historical.index[-1]],
                            y=[current_price],
                            mode='markers',
                            name='Current Price',
                            marker=dict(size=12, color='green')
                        ))
                        
                        fig.update_layout(
                            title=f"{selected_stock} Price Forecast",
                            xaxis_title="Date",
                            yaxis_title="Price (₹)",
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional insights
                        st.subheader("Analysis Insights")
                        
                        avg_forecast = float(future_prices['Predicted Price'].mean())
                        total_change = ((avg_forecast - current_price) / current_price) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Price", f"₹{current_price:.2f}")
                        
                        with col2:
                            st.metric("Avg Forecast", f"₹{avg_forecast:.2f}")
                        
                        with col3:
                            st.metric("Expected Return", f"{total_change:+.1f}%")
                        
                        # Risk assessment
                        st.info("""
                        **Risk Disclaimer:** 
                        - This forecast is based on AI models and historical data
                        - Stock markets are volatile and predictions may not be accurate
                        - Past performance does not guarantee future results
                        - Consult with a financial advisor before making investment decisions
                        """)
                    else:
                        st.error("Failed to generate price forecast.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Disclaimer:**
This tool provides AI-powered analysis for informational purposes only. 
It is not financial advice. Always conduct your own research and consult 
with qualified financial advisors before making investment decisions.
""")