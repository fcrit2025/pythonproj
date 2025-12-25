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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, LSTM, Concatenate, BatchNormalization
# For TensorFlow 2.x (modern versions)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import CustomBusinessDay
import warnings
warnings.filterwarnings('ignore')

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

# =============================================
# NEW: Fetch India VIX Data
# =============================================
def get_india_vix_data(start_date, end_date):
    """
    Fetch India VIX data from Yahoo Finance
    India VIX ticker: ^INDIAVIX
    """
    try:
        vix_ticker = "^INDIAVIX"
        vix_data = yf.download(vix_ticker, start=start_date, end=end_date, progress=False)
        
        if vix_data.empty:
            # Fallback: Create synthetic VIX data based on NIFTY volatility
            nifty_data = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
            if not nifty_data.empty:
                # Calculate rolling volatility as proxy for VIX
                returns = nifty_data['Close'].pct_change()
                vix_data = pd.DataFrame({
                    'Open': returns.rolling(20).std() * 100 * 16,  # Annualized volatility
                    'High': returns.rolling(20).std() * 100 * 16 * 1.1,
                    'Low': returns.rolling(20).std() * 100 * 16 * 0.9,
                    'Close': returns.rolling(20).std() * 100 * 16,
                    'Volume': nifty_data['Volume']
                })
                vix_data = vix_data.dropna()
        
        return vix_data
    except Exception as e:
        st.warning(f"Could not fetch India VIX data: {str(e)}")
        return pd.DataFrame()

# =============================================
# NEW: Technical Indicators Generator
# =============================================
def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    df = df.copy()
    
    # Basic indicators
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(5, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['MA200'] = df['Close'].rolling(200, min_periods=1).mean()
    
    # Moving Average Ratios
    df['MA_Ratio_5_20'] = df['MA5'] / df['MA20']
    df['MA_Ratio_5_50'] = df['MA5'] / df['MA50']
    df['MA_Ratio_20_200'] = df['MA20'] / df['MA200']
    
    # Volatility Measures
    df['Volatility_5D'] = df['Returns'].rolling(5, min_periods=1).std()
    df['Volatility_20D'] = df['Returns'].rolling(20, min_periods=1).std()
    df['ATR'] = calculate_atr(df)  # Average True Range
    
    # Volume Indicators
    df['Volume_MA5'] = df['Volume'].rolling(5, min_periods=1).mean()
    df['Volume_MA20'] = df['Volume'].rolling(20, min_periods=1).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    df['OBV'] = calculate_obv(df)  # On-Balance Volume
    
    # Momentum Indicators
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Support/Resistance Levels
    df['Pivot_Point'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = 2 * df['Pivot_Point'] - df['Low']
    df['S1'] = 2 * df['Pivot_Point'] - df['High']
    
    # Price Position
    df['Price_vs_MA20'] = df['Close'] / df['MA20'] - 1
    df['Price_vs_MA50'] = df['Close'] / df['MA50'] - 1
    
    # Gap Analysis
    df['Gap'] = df['Open'] / df['Close'].shift(1) - 1
    
    return df.dropna()

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period, min_periods=1).mean()

def calculate_obv(df):
    """Calculate On-Balance Volume"""
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# =============================================
# NEW: Dynamic Fusion Framework Models
# =============================================
class TechnicalExpertModel:
    """GRU-based model for technical data"""
    def __init__(self, lookback=30, n_features=25):
        self.lookback = lookback
        self.n_features = n_features
        self.model = self.build_model()
        self.scaler = MinMaxScaler()
        self.recent_errors = []
        self.max_error_window = 10
        
    def build_model(self):
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=(self.lookback, self.n_features)),
            BatchNormalization(),
            Dropout(0.3),
            GRU(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            GRU(32),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def prepare_data(self, X, y, lookback=30):
        """Prepare sequential data for GRU"""
        X_scaled = self.scaler.fit_transform(X)
        X_3d = []
        y_3d = []
        
        for i in range(lookback, len(X_scaled)):
            X_3d.append(X_scaled[i-lookback:i])
            y_3d.append(y[i])
        
        return np.array(X_3d), np.array(y_3d)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train the model"""
        X_seq, y_seq = self.prepare_data(X_train, y_train, self.lookback)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_data(X_val, y_val, self.lookback)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=0
        )
        return history
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        X_seq = []
        
        if len(X_scaled) >= self.lookback:
            X_seq = X_scaled[-self.lookback:].reshape(1, self.lookback, -1)
            prediction = self.model.predict(X_seq, verbose=0)[0][0]
        else:
            # Pad with zeros if not enough data
            padding = np.zeros((self.lookback - len(X_scaled), X_scaled.shape[1]))
            X_padded = np.vstack([padding, X_scaled])
            X_seq = X_padded.reshape(1, self.lookback, -1)
            prediction = self.model.predict(X_seq, verbose=0)[0][0]
        
        return prediction
    
    def update_errors(self, true_value, predicted_value):
        """Update error tracking for uncertainty calculation"""
        error = (true_value - predicted_value) ** 2
        self.recent_errors.append(error)
        
        # Keep only last N errors
        if len(self.recent_errors) > self.max_error_window:
            self.recent_errors.pop(0)
    
    def get_uncertainty(self):
        """Calculate uncertainty (variance of recent errors)"""
        if len(self.recent_errors) == 0:
            return 1.0  # Maximum uncertainty if no data
        
        return np.mean(self.recent_errors)

class SentimentExpertModel:
    """Transformer-based model for sentiment data"""
    def __init__(self):
        self.model = self.build_model()
        self.scaler = MinMaxScaler()
        self.recent_errors = []
        self.max_error_window = 10
        
    def build_model(self):
        # Simplified transformer-like architecture using dense layers
        input_layer = Input(shape=(5,))  # Sentiment features
        x = Dense(64, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        output_layer = Dense(1)(x)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def extract_sentiment_features(self, sentiment_data):
        """Extract features from sentiment data"""
        features = []
        
        if len(sentiment_data) > 0:
            # Calculate various sentiment metrics
            sentiments = [s[0] for s in sentiment_data]  # Sentiment labels
            confidences = [s[1] for s in sentiment_data]  # Confidence scores
            
            # Convert sentiments to numerical values
            sentiment_values = []
            for sentiment, confidence in zip(sentiments, confidences):
                if sentiment == 'positive':
                    sentiment_values.append(confidence)
                elif sentiment == 'negative':
                    sentiment_values.append(-confidence)
                else:
                    sentiment_values.append(0)
            
            # Calculate features
            if sentiment_values:
                features = [
                    np.mean(sentiment_values),  # Average sentiment
                    np.std(sentiment_values),   # Sentiment volatility
                    len([v for v in sentiment_values if v > 0]) / len(sentiment_values),  # Positive ratio
                    len([v for v in sentiment_values if v < 0]) / len(sentiment_values),  # Negative ratio
                    np.max(sentiment_values)    # Maximum sentiment intensity
                ]
            else:
                features = [0, 0, 0.5, 0.5, 0]
        else:
            features = [0, 0, 0.5, 0.5, 0]  # Neutral if no sentiment data
        
        return np.array(features).reshape(1, -1)
    
    def train(self, X_train, y_train, epochs=30, batch_size=16):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        return history
    
    def predict(self, sentiment_data):
        """Make prediction based on sentiment"""
        features = self.extract_sentiment_features(sentiment_data)
        prediction = self.model.predict(features, verbose=0)[0][0]
        return prediction
    
    def update_errors(self, true_value, predicted_value):
        """Update error tracking"""
        error = (true_value - predicted_value) ** 2
        self.recent_errors.append(error)
        
        if len(self.recent_errors) > self.max_error_window:
            self.recent_errors.pop(0)
    
    def get_uncertainty(self):
        """Calculate uncertainty"""
        if len(self.recent_errors) == 0:
            return 1.0
        
        return np.mean(self.recent_errors)

class VolatilityExpertModel:
    """MLP model for volatility (VIX) data"""
    def __init__(self):
        self.model = self.build_model()
        self.scaler = MinMaxScaler()
        self.recent_errors = []
        self.max_error_window = 10
        
    def build_model(self):
        model = Sequential([
            Dense(32, activation='relu', input_shape=(3,)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            Dense(8, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def extract_volatility_features(self, vix_data, stock_data):
        """Extract volatility features from VIX and stock data"""
        if vix_data is None or stock_data is None or vix_data.empty or stock_data.empty:
            return np.array([[0.0, 0.0, 0.0]])  # Default features

        # Ensure we operate on the 'Close' series for VIX and the volatility column for stock
        if 'Close' in vix_data.columns:
            vix_close = vix_data['Close']
        else:
            vix_close = pd.Series(dtype=float)

        # Latest VIX close (scalar) - handle scalar or array-like safely
        latest_vix_close = 0.0
        if len(vix_close) > 0:
            last = vix_close.iloc[-1]
            try:
                if isinstance(last, (pd.Series, np.ndarray)):
                    arr = np.asarray(last).ravel()
                    if arr.size > 0 and not np.isnan(arr[-1]):
                        latest_vix_close = float(arr[-1])
                else:
                    if pd.notna(last):
                        latest_vix_close = float(last)
            except Exception:
                latest_vix_close = 0.0

        # VIX vs MA20 (safe computation)
        vix_vs_ma = 1.0
        if len(vix_close) >= 20:
            try:
                vix_ma20_raw = vix_close.rolling(20).mean().iloc[-1]
                # handle scalar or array-like
                if isinstance(vix_ma20_raw, (pd.Series, np.ndarray)):
                    vix_ma20_arr = np.asarray(vix_ma20_raw).ravel()
                    vix_ma20 = float(vix_ma20_arr[-1]) if vix_ma20_arr.size > 0 and not np.isnan(vix_ma20_arr[-1]) else None
                else:
                    vix_ma20 = float(vix_ma20_raw) if pd.notna(vix_ma20_raw) else None

                if vix_ma20 and vix_ma20 != 0:
                    vix_vs_ma = latest_vix_close / vix_ma20
                else:
                    vix_vs_ma = 1.0
            except Exception:
                vix_vs_ma = 1.0

        # Latest stock volatility (scalar)
        if 'Volatility_20D' in stock_data.columns and len(stock_data) > 0:
            latest_stock_vol = stock_data['Volatility_20D'].iloc[-1]
            latest_stock_vol = float(latest_stock_vol) if not pd.isna(latest_stock_vol) else 0.0
        else:
            latest_stock_vol = 0.0

        features = [latest_vix_close, vix_vs_ma, latest_stock_vol]
        return np.array(features, dtype=float).reshape(1, -1)
    
    def train(self, X_train, y_train, epochs=30, batch_size=16):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        return history
    
    def predict(self, vix_data, stock_data):
        """Make prediction based on volatility"""
        features = self.extract_volatility_features(vix_data, stock_data)
        prediction = self.model.predict(features, verbose=0)[0][0]
        return prediction
    
    def update_errors(self, true_value, predicted_value):
        """Update error tracking"""
        error = (true_value - predicted_value) ** 2
        self.recent_errors.append(error)
        
        if len(self.recent_errors) > self.max_error_window:
            self.recent_errors.pop(0)
    
    def get_uncertainty(self):
        """Calculate uncertainty"""
        if len(self.recent_errors) == 0:
            return 1.0
        
        return np.mean(self.recent_errors)

# =============================================
# NEW: Dynamic Fusion Framework
# =============================================
class DynamicFusionFramework:
    """Dynamic Fusion Framework with uncertainty-based weighting"""
    
    def __init__(self):
        self.technical_model = TechnicalExpertModel()
        self.sentiment_model = SentimentExpertModel()
        self.volatility_model = VolatilityExpertModel()
        
        # Track model performances
        self.model_predictions = {
            'technical': [],
            'sentiment': [],
            'volatility': []
        }
        self.true_values = []
        
    def calculate_dynamic_weights(self):
        """Calculate dynamic weights based on model uncertainties"""
        uncertainties = {
            'technical': self.technical_model.get_uncertainty(),
            'sentiment': self.sentiment_model.get_uncertainty(),
            'volatility': self.volatility_model.get_uncertainty()
        }
        
        # Apply Bayesian weighting formula: w_i = exp(-σ_i²) / Σ exp(-σ_j²)
        weights = {}
        total_weight = 0
        
        for model_name, uncertainty in uncertainties.items():
            # Avoid extreme values
            uncertainty = max(uncertainty, 1e-6)  # Prevent division by zero
            weight = np.exp(-uncertainty)
            weights[model_name] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for model_name in weights:
                weights[model_name] /= total_weight
        else:
            # Equal weights if all uncertainties are too high
            for model_name in weights:
                weights[model_name] = 1/3
        
        return weights, uncertainties
    
    def train_models(self, stock_data, sentiment_data, vix_data):
        """Train all three expert models"""
        
        # Prepare data
        stock_data_with_indicators = calculate_technical_indicators(stock_data)
        
        # Technical model features
        tech_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA20', 'MA50', 'MA_Ratio_5_20',
            'Volatility_5D', 'Volatility_20D', 'ATR',
            'Volume_Ratio', 'RSI', 'MACD', 'MACD_Histogram',
            'Price_vs_MA20', 'Price_vs_MA50', 'Gap'
        ]
        
        # Ensure all features exist
        available_features = [f for f in tech_features if f in stock_data_with_indicators.columns]
        X_tech = stock_data_with_indicators[available_features]
        y_tech = stock_data_with_indicators['Returns'].shift(-1).dropna()
        X_tech = X_tech.iloc[:-1]  # Align with y
        
        # Train-test split
        split_idx = int(len(X_tech) * 0.8)
        X_tech_train, X_tech_test = X_tech.iloc[:split_idx], X_tech.iloc[split_idx:]
        y_tech_train, y_tech_test = y_tech.iloc[:split_idx], y_tech.iloc[split_idx:]
        
        # Train technical model
        # Ensure the technical model expects the correct number of features
        n_features_actual = X_tech.shape[1]
        if getattr(self.technical_model, 'n_features', None) != n_features_actual:
            self.technical_model.n_features = n_features_actual
            # Rebuild model to match the actual input feature count
            self.technical_model.model = self.technical_model.build_model()

        self.technical_model.train(X_tech_train, y_tech_train, X_tech_test, y_tech_test)
        
        # Prepare sentiment data
        sentiment_features = []
        sentiment_targets = []
        
        for date in stock_data_with_indicators.index:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in sentiment_data:
                daily_sentiments = sentiment_data[date_str]
                features = self.sentiment_model.extract_sentiment_features(daily_sentiments)
                sentiment_features.append(features[0])
                
                # Use next day's return as target
                if date in stock_data_with_indicators.index:
                    idx = stock_data_with_indicators.index.get_loc(date)
                    if idx + 1 < len(stock_data_with_indicators):
                        target = stock_data_with_indicators['Returns'].iloc[idx + 1]
                        sentiment_targets.append(target)
        
        if len(sentiment_features) > 10:
            X_sent = np.array(sentiment_features)[:len(sentiment_targets)]
            y_sent = np.array(sentiment_targets)[:len(sentiment_features)]
            self.sentiment_model.train(X_sent, y_sent)
        
        # Prepare volatility data
        volatility_features = []
        volatility_targets = []
        
        for i in range(len(stock_data_with_indicators)):
            if i >= 20:  # Need enough data for volatility calculation
                vix_slice = vix_data.iloc[:i+1] if len(vix_data) > i else vix_data
                stock_slice = stock_data_with_indicators.iloc[:i+1]
                
                features = self.volatility_model.extract_volatility_features(vix_slice, stock_slice)
                volatility_features.append(features[0])
                
                # Use next day's return as target
                if i + 1 < len(stock_data_with_indicators):
                    target = stock_data_with_indicators['Returns'].iloc[i + 1]
                    volatility_targets.append(target)
        
        if len(volatility_features) > 10:
            X_vol = np.array(volatility_features)[:len(volatility_targets)]
            y_vol = np.array(volatility_targets)[:len(volatility_features)]
            self.volatility_model.train(X_vol, y_vol)
    
    def predict(self, stock_data, sentiment_data, vix_data):
        """Make combined prediction using dynamic fusion"""
        
        # Get individual predictions
        stock_features = calculate_technical_indicators(stock_data)
        
        # Get technical features - CORRECTED VERSION
        if hasattr(self.technical_model.scaler, 'feature_names_in_'):
            tech_features = [f for f in self.technical_model.scaler.feature_names_in_ 
                            if f in stock_features.columns]
        else:
            # Use default features if feature_names_in_ not available
            tech_features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA5', 'MA20', 'MA50', 'MA_Ratio_5_20',
                'Volatility_5D', 'Volatility_20D', 'ATR',
                'Volume_Ratio', 'RSI', 'MACD', 'MACD_Histogram',
                'Price_vs_MA20', 'Price_vs_MA50', 'Gap'
            ]
            # Filter to only include features that exist in the DataFrame
            tech_features = [f for f in tech_features if f in stock_features.columns]
        
        # Ensure we have enough features
        if len(tech_features) < 5:  # Minimum number of features needed
            # Use all available columns except the target
            tech_features = [col for col in stock_features.columns 
                            if col not in ['Returns', 'Target', 'Predicted']]
        
        # Limit to the model's expected number of features
        tech_features = tech_features[:self.technical_model.n_features]
        
        X_tech = stock_features[tech_features].iloc[-self.technical_model.lookback:]
        
        # If we have fewer than the required lookback days, warn but allow prediction by padding inside the model
        if len(X_tech) < self.technical_model.lookback:
            st.warning(f"Insufficient data for technical model. Need {self.technical_model.lookback} days, have {len(X_tech)} — padding will be applied to predict.")

        tech_pred = self.technical_model.predict(X_tech)
        
        # Get sentiment for latest date
        latest_date = stock_data.index[-1].strftime('%Y-%m-%d')
        latest_sentiment = sentiment_data.get(latest_date, [])
        sentiment_pred = self.sentiment_model.predict(latest_sentiment)
        
        # Volatility prediction
        volatility_pred = self.volatility_model.predict(vix_data, stock_features)
        
        # Calculate dynamic weights
        weights, uncertainties = self.calculate_dynamic_weights()
        
        # Store predictions for tracking
        self.model_predictions['technical'].append(tech_pred)
        self.model_predictions['sentiment'].append(sentiment_pred)
        self.model_predictions['volatility'].append(volatility_pred)
        
        # Combine predictions with dynamic weights
        combined_pred = (
            weights['technical'] * tech_pred +
            weights['sentiment'] * sentiment_pred +
            weights['volatility'] * volatility_pred
        )
        
        return {
            'combined_prediction': combined_pred,
            'individual_predictions': {
                'technical': tech_pred,
                'sentiment': sentiment_pred,
                'volatility': volatility_pred
            },
            'weights': weights,
            'uncertainties': uncertainties
        }
    
    def update_model_performance(self, true_return):
        """Update models with true value for error calculation"""
        self.true_values.append(true_return)
        
        if len(self.true_values) > 1 and len(self.model_predictions['technical']) > 0:
            last_true = self.true_values[-2]  # Previous true value
            
            # Update each model's error tracking
            for model_name in ['technical', 'sentiment', 'volatility']:
                if len(self.model_predictions[model_name]) > 0:
                    last_pred = self.model_predictions[model_name][-1]
                    
                    if model_name == 'technical':
                        self.technical_model.update_errors(last_true, last_pred)
                    elif model_name == 'sentiment':
                        self.sentiment_model.update_errors(last_true, last_pred)
                    elif model_name == 'volatility':
                        self.volatility_model.update_errors(last_true, last_pred)

# =============================================
# NEW: Enhanced Visualization Functions
# =============================================
def create_dynamic_weights_visualization(weights_history):
    """Create visualization for dynamic weights over time"""
    fig = go.Figure()
    
    dates = list(weights_history.keys())
    tech_weights = [w['technical'] for w in weights_history.values()]
    sent_weights = [w['sentiment'] for w in weights_history.values()]
    vol_weights = [w['volatility'] for w in weights_history.values()]
    
    fig.add_trace(go.Scatter(
        x=dates, y=tech_weights,
        mode='lines+markers',
        name='Technical Model Weight',
        line=dict(color='blue', width=2),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=sent_weights,
        mode='lines+markers',
        name='Sentiment Model Weight',
        line=dict(color='green', width=2),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=vol_weights,
        mode='lines+markers',
        name='Volatility Model Weight',
        line=dict(color='red', width=2),
        stackgroup='one'
    ))
    
    fig.update_layout(
        title='Dynamic Model Weights Over Time',
        xaxis_title='Date',
        yaxis_title='Weight',
        hovermode='x unified',
        yaxis=dict(tickformat='.0%'),
        showlegend=True
    )
    
    return fig

def create_uncertainty_visualization(uncertainties_history):
    """Create visualization for model uncertainties"""
    fig = go.Figure()
    
    dates = list(uncertainties_history.keys())
    tech_unc = [u['technical'] for u in uncertainties_history.values()]
    sent_unc = [u['sentiment'] for u in uncertainties_history.values()]
    vol_unc = [u['volatility'] for u in uncertainties_history.values()]
    
    fig.add_trace(go.Scatter(
        x=dates, y=tech_unc,
        mode='lines',
        name='Technical Model Uncertainty',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=sent_unc,
        mode='lines',
        name='Sentiment Model Uncertainty',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=vol_unc,
        mode='lines',
        name='Volatility Model Uncertainty',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Model Uncertainties Over Time',
        xaxis_title='Date',
        yaxis_title='Uncertainty (σ²)',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_model_performance_radar(weights, uncertainties):
    """Create radar chart for model performance comparison"""
    fig = go.Figure()
    
    models = ['Technical', 'Sentiment', 'Volatility']
    
    # Inverse of uncertainty = confidence
    confidences = [1/(u+1e-6) for u in uncertainties.values()]

    weight_vals = list(weights.values())

    fig.add_trace(go.Scatterpolar(
        r=weight_vals,
        theta=models,
        fill='toself',
        name='Model Weights',
        line_color='blue'
    ))

    fig.add_trace(go.Scatterpolar(
        r=confidences,
        theta=models,
        fill='toself',
        name='Model Confidence',
        line_color='red'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(weight_vals), max(confidences))]
            )),
        showlegend=True,
        title='Model Performance Radar Chart'
    )
    
    return fig

# =============================================
# MODIFIED: Enhanced Streamlit UI Integration
# =============================================

# Fetch stock data (existing function)
def get_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    # Ensure start/end are passed as ISO date strings to yfinance
    try:
        start_str = pd.to_datetime(start).strftime('%Y-%m-%d')
    except Exception:
        start_str = start
    try:
        end_str = pd.to_datetime(end).strftime('%Y-%m-%d')
    except Exception:
        end_str = end

    data = stock.history(start=start_str, end=end_str)
    if not data.empty:
        data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        data.index = data.index.tz_localize(None)
        data = data.sort_index()
    return data

# Fetch stock info (existing function)
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

# News API (existing function)
NEWS_API_KEY = "563215a35c1a47968f46271e04083ea3"
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

# Sentiment analysis (existing function)
def analyze_sentiment(text):
    if not text:
        return "neutral", 0.0
    result = sentiment_pipeline(text[:512])[0]
    return result['label'], result['score']

# Filter relevant news (existing function)
def filter_relevant_news(news_articles, stock_name):
    filtered_articles = []
    for article in news_articles:
        title = article.get('title', '')
        if title and re.search(stock_name, title, re.IGNORECASE):  
            filtered_articles.append(article)
    return filtered_articles

# Feature engineering (existing function)
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

# Prophet forecasting (existing function)
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

# Hybrid XGBoost-GRU-Prophet Model (existing function)
def create_hybrid_model(df_stock, sentiment_features):
    # Prepare data with sentiment
    sentiment_df = pd.DataFrame(sentiment_features.items(), columns=["Date", "Sentiment"])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None)
    df_stock.index = pd.to_datetime(df_stock.index).tz_localize(None)
    df_stock = df_stock.reset_index().merge(sentiment_df, on="Date", how="left").set_index("Date")
    df_stock['Sentiment'] = pd.to_numeric(df_stock['Sentiment'], errors='coerce').fillna(0)
    
    # Feature engineering
    df_stock = create_advanced_features(df_stock)
    df_stock['Target'] = df_stock['Close'].pct_change().shift(-1)
    df_stock.dropna(inplace=True)
    
    # Features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment',
               '5D_MA', 'MA_Ratio', '5D_Volatility', 'Volume_Ratio']
    
    # Train-test split
    X = df_stock[features]
    y = df_stock['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 1. XGBoost Model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        early_stopping_rounds=30,
        random_state=42
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # XGBoost predictions and metrics
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(xgb_mse)
    
    # 2. GRU Model
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_3d = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    
    gru_model = Sequential([
        GRU(64, input_shape=(1, X_3d.shape[2]), return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])
    
    gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    gru_model.fit(X_3d[:len(X_train)], y_train[:len(X_train)], 
                epochs=50, batch_size=32, verbose=0)
    
    # GRU predictions and metrics
    gru_pred = gru_model.predict(X_3d[len(X_train):]).flatten()
    gru_mae = mean_absolute_error(y_test, gru_pred)
    gru_mse = mean_squared_error(y_test, gru_pred)
    gru_rmse = np.sqrt(gru_mse)
    
    # Calculate model weights based on test performance
    total = (1/xgb_mae + 1/gru_mae)
    xgb_weight = (1/xgb_mae)/total
    gru_weight = (1/gru_mae)/total
    
    # Final predictions with dynamic weighting
    final_pred = (xgb_weight * xgb_pred) + (gru_weight * gru_pred)
    df_stock.loc[y_test.index, 'Predicted'] = final_pred
    
    # Hybrid model metrics
    hybrid_mae = mean_absolute_error(y_test, final_pred)
    hybrid_mse = mean_squared_error(y_test, final_pred)
    hybrid_rmse = np.sqrt(hybrid_mse)
    accuracy = max(0, 100 - (hybrid_mae * 100))
    
    # Create metrics dictionary
    model_metrics = {
        'xgb': {
            'mae': xgb_mae,
            'mse': xgb_mse,
            'rmse': xgb_rmse,
            'weight': xgb_weight
        },
        'gru': {
            'mae': gru_mae,
            'mse': gru_mse,
            'rmse': gru_rmse,
            'weight': gru_weight
        },
        'hybrid': {
            'mae': hybrid_mae,
            'mse': hybrid_mse,
            'rmse': hybrid_rmse,
            'accuracy': accuracy
        }
    }
    
    return df_stock, {'xgb': xgb_model, 'gru': gru_model}, scaler, features, model_metrics

# Hybrid predict prices (existing function)
def hybrid_predict_prices(models, scaler, last_known_data, features, days=10, weights=None):
    """Generate predictions with realistic price bounds and adjust for market closures"""
    try:
        # Default weights if not provided
        if weights is None:
            weights = {'xgb_weight': 0.6, 'gru_weight': 0.4}
            
        # Get Prophet forecast for trend baseline
        prophet_forecast_df = prophet_forecast(last_known_data, days=days)
        future_dates = prophet_forecast_df.index
        
        # Initialize results DataFrame
        future_prices = pd.DataFrame(index=future_dates, 
                                   columns=['Predicted Price', 'Daily Change (%)'])
        
        current_data = last_known_data.copy()
        last_close = current_data['Close'].iloc[-1]
        recent_volatility = current_data['5D_Volatility'].iloc[-1]
        
        for i, date in enumerate(future_dates):
            # 1. Get individual model predictions
            try:
                # XGBoost prediction
                xgb_input = current_data[features].iloc[-1:].copy()
                xgb_pred = models['xgb'].predict(xgb_input)[0]
                
                # GRU prediction
                input_scaled = scaler.transform(xgb_input)
                input_3d = input_scaled.reshape(1, 1, input_scaled.shape[1])
                gru_pred = models['gru'].predict(input_3d)[0][0]
                
                # Prophet prediction (convert to % change)
                prophet_pred = (prophet_forecast_df['yhat'].iloc[i] - last_close) / last_close
                
            except Exception as e:
                st.error(f"Model prediction failed: {str(e)}")
                return pd.DataFrame()
            
            # 2. Create weighted ensemble prediction
            combined_pred = (
                weights['xgb_weight'] * xgb_pred + 
                weights['gru_weight'] * gru_pred
            )
            
            # Blend with Prophet (20% weight)
            final_pred = 0.8 * combined_pred + 0.2 * prophet_pred
            
            # 3. Apply realistic noise (scaled to recent volatility)
            max_daily_change = 0.05  # 5% max daily change
            noise = np.random.normal(0, min(recent_volatility, 0.03))  # Cap volatility impact
            adj_pred = np.clip(final_pred + noise, -max_daily_change, max_daily_change)
            
            # 4. Calculate new price with bounds
            new_close = last_close * (1 + adj_pred)
            
            # Ensure price stays within reasonable bounds (10% of current price)
            price_bound_low = last_close * 0.90
            price_bound_high = last_close * 1.10
            new_close = np.clip(new_close, price_bound_low, price_bound_high)
            
            future_prices.loc[date, 'Predicted Price'] = new_close
            last_close = new_close
            
            # 5. Update simulated data for next prediction
            new_row = {
                'Open': new_close * 0.998,  # Typical open slightly below previous close
                'High': new_close * 1.005,  # Assume 0.5% higher than close
                'Low': new_close * 0.995,   # Assume 0.5% lower than close
                'Close': new_close,
                'Volume': current_data['Volume'].iloc[-1],  # Keep same volume initially
                'Sentiment': current_data['Sentiment'].iloc[-1],  # Carry forward sentiment
                '5D_MA': (current_data['5D_MA'].iloc[-1] * 4 + new_close) / 5,  # Update MA
                '20D_MA': (current_data['20D_MA'].iloc[-1] * 19 + new_close) / 20,
                'MA_Ratio': ((current_data['5D_MA'].iloc[-1] * 4 + new_close) / 5) / 
                ((current_data['20D_MA'].iloc[-1] * 19 + new_close) / 20),
                '5D_Volatility': np.sqrt(((current_data['Returns'].iloc[-4:]**2).sum() + 
                             ((new_close - current_data['Close'].iloc[-1])/current_data['Close'].iloc[-1])**2)/5),
                'Volume_MA5': (current_data['Volume_MA5'].iloc[-1] * 4 + current_data['Volume'].iloc[-1]) / 5,
                'Volume_Ratio': current_data['Volume'].iloc[-1] / 
                   ((current_data['Volume_MA5'].iloc[-1] * 4 + current_data['Volume'].iloc[-1]) / 5)
            }
            current_data = pd.concat([current_data.iloc[1:], pd.DataFrame(new_row, index=[date])])
        
        # Calculate initial daily changes
        future_prices['Daily Change (%)'] = future_prices['Predicted Price'].pct_change().fillna(0) * 100
        
        # Adjust predictions for market closures
        future_prices = adjust_predictions_for_market_closures(future_prices)
        
        return future_prices
    
    except Exception as e:
        st.error(f"Forecast generation failed: {str(e)}")
        return pd.DataFrame()

# Candlestick chart (existing function)
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

# Generate investment recommendation (existing function)
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

# =============================================
# MAIN STREAMLIT APP WITH DYNAMIC FUSION
# =============================================

# Streamlit UI
st.title("Indian Stock Market Analysis with Dynamic Fusion AI Framework")
st.sidebar.header("Stock Selection")
indian_stocks = get_indian_stocks()
selected_stock = st.sidebar.selectbox("Select Stock", indian_stocks)
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
chart_type = st.sidebar.radio("Chart Type", ["Candlestick Chart", "Line Chart"])

# Add Dynamic Fusion toggle
enable_dynamic_fusion = st.sidebar.checkbox("Enable Dynamic Fusion Framework", value=True)
forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 10)

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
        
        # =============================================
        # NEW: Dynamic Fusion Framework Integration
        # =============================================
        if enable_dynamic_fusion:
            st.header("🔬 Dynamic Fusion Framework Analysis")
            
            with st.spinner("Training Dynamic Fusion Models..."):
                # Fetch India VIX data
                st.subheader("India VIX Data")
                vix_data = get_india_vix_data(start_date, end_date)
                
                if not vix_data.empty:
                    vix_close = vix_data['Close']
                    # get last non-NA close value to avoid ambiguous truth-value of Series
                    vix_close_non_na = vix_close.dropna()
                    if not vix_close_non_na.empty:
                        vix_val = vix_close_non_na.iloc[-1]
                        st.write(f"Latest India VIX: {float(vix_val):.2f}")
                    else:
                        st.write("Latest India VIX: N/A")
                else:
                    st.warning("Could not fetch India VIX data. Using volatility proxy.")
                
                # Initialize Dynamic Fusion Framework
                fusion_framework = DynamicFusionFramework()
                
                # Prepare sentiment data in required format
                formatted_sentiment = {}
                for date, scores in daily_sentiment.items():
                    formatted_scores = []
                    for sentiment, confidence in scores:
                        formatted_scores.append((sentiment, confidence))
                    formatted_sentiment[date] = formatted_scores
                
                # Train the framework
                fusion_framework.train_models(df_stock, formatted_sentiment, vix_data)
                # Debug: show fetched stock data range to ensure dates are up-to-date
                if not df_stock.empty:
                    st.write(f"Fetched stock data: {df_stock.index.min().date()} -> {df_stock.index.max().date()} ({len(df_stock)} rows)")
                
                # Generate predictions using dynamic fusion
                st.subheader("Dynamic Fusion Predictions")
                
                # Create a DataFrame to store dynamic predictions
                # Use the actual last dates present in `df_stock` (up to 30 most recent rows)
                recent_index = df_stock.index[-min(30, len(df_stock)):]
                dynamic_predictions = pd.DataFrame(index=recent_index, 
                                                  columns=['Actual Return', 'Fusion Prediction', 
                                                          'Technical Weight', 'Sentiment Weight', 'Volatility Weight'])
                
                weights_history = {}
                uncertainties_history = {}
                
                # Generate predictions for recent data
                for i in range(20, min(30, len(df_stock))):
                    date = df_stock.index[i]
                    date_str = date.strftime('%Y-%m-%d')
                    
                    # Use data up to current point
                    stock_slice = df_stock.iloc[:i]
                    
                    # Get prediction
                    result = fusion_framework.predict(stock_slice, formatted_sentiment, vix_data)
                    
                    # Store results
                    if i < len(df_stock) - 1:
                        actual_return = df_stock['Close'].iloc[i+1] / df_stock['Close'].iloc[i] - 1
                    else:
                        actual_return = np.nan
                    
                    dynamic_predictions.loc[date, 'Actual Return'] = actual_return
                    dynamic_predictions.loc[date, 'Fusion Prediction'] = result['combined_prediction']
                    dynamic_predictions.loc[date, 'Technical Weight'] = result['weights']['technical']
                    dynamic_predictions.loc[date, 'Sentiment Weight'] = result['weights']['sentiment']
                    dynamic_predictions.loc[date, 'Volatility Weight'] = result['weights']['volatility']
                    
                    # Store for visualization
                    weights_history[date_str] = result['weights']
                    uncertainties_history[date_str] = result['uncertainties']
                    
                    # Update model performance
                    if not np.isnan(actual_return):
                        fusion_framework.update_model_performance(actual_return)
                
                # Display dynamic weights visualization
                st.subheader("Dynamic Model Weights Over Time")
                weights_fig = create_dynamic_weights_visualization(weights_history)
                st.plotly_chart(weights_fig, use_container_width=True)
                
                # Display uncertainty visualization
                st.subheader("Model Uncertainties Over Time")
                uncertainty_fig = create_uncertainty_visualization(uncertainties_history)
                st.plotly_chart(uncertainty_fig, use_container_width=True)
                
                # Display model performance radar
                st.subheader("Model Performance Comparison")
                latest_weights = list(weights_history.values())[-1] if weights_history else {'technical': 0.33, 'sentiment': 0.33, 'volatility': 0.33}
                latest_uncertainties = list(uncertainties_history.values())[-1] if uncertainties_history else {'technical': 0.5, 'sentiment': 0.5, 'volatility': 0.5}
                
                radar_fig = create_model_performance_radar(latest_weights, latest_uncertainties)
                st.plotly_chart(radar_fig, use_container_width=True)
                
                # Display dynamic predictions table
                st.subheader("Dynamic Fusion Predictions Table")
                display_df = dynamic_predictions.dropna().copy()
                display_df['Prediction Error'] = display_df['Actual Return'] - display_df['Fusion Prediction']
                
                # Format for display
                styled_df = display_df.style.format({
                    'Actual Return': '{:.4f}',
                    'Fusion Prediction': '{:.4f}',
                    'Prediction Error': '{:.4f}',
                    'Technical Weight': '{:.1%}',
                    'Sentiment Weight': '{:.1%}',
                    'Volatility Weight': '{:.1%}'
                }).applymap(
                    lambda x: 'color: green' if x > 0 else 'color: red',
                    subset=['Actual Return', 'Fusion Prediction', 'Prediction Error']
                )
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Calculate and display metrics
                if len(display_df) > 5:
                    mae = np.mean(np.abs(display_df['Prediction Error']))
                    rmse = np.sqrt(np.mean(display_df['Prediction Error']**2))
                    accuracy = max(0, 100 - (mae * 100))
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Dynamic Fusion MAE", f"{mae:.4f}")
                    with col2:
                        st.metric("Dynamic Fusion RMSE", f"{rmse:.4f}")
                    with col3:
                        st.metric("Dynamic Fusion Accuracy", f"{accuracy:.1f}%")
                    
                    # Show latest weights
                    st.subheader("Current Model Weights (Latest)")
                    latest_date = list(weights_history.keys())[-1]
                    latest_weights = weights_history[latest_date]
                    
                    weight_col1, weight_col2, weight_col3 = st.columns(3)
                    with weight_col1:
                        st.metric("Technical Model", f"{latest_weights['technical']:.1%}")
                    with weight_col2:
                        st.metric("Sentiment Model", f"{latest_weights['sentiment']:.1%}")
                    with weight_col3:
                        st.metric("Volatility Model", f"{latest_weights['volatility']:.1%}")
        
        # Continue with existing hybrid model analysis
        st.header("📊 Hybrid AI Model Analysis")
        
        # Train hybrid model - NOW WITH COMPREHENSIVE METRICS
        df_stock, models, scaler, features, model_metrics = create_hybrid_model(df_stock, daily_sentiment if daily_sentiment else {})
        weights = {
            'xgb_weight': model_metrics['xgb']['weight'],
            'gru_weight': model_metrics['gru']['weight']
        }
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
            st.subheader(f"{forecast_days}-Day Price Forecast")
            last_data = df_stock.iloc[-30:]
            current_price = last_data['Close'].iloc[-1]

            future_prices = hybrid_predict_prices(models=models,
                                                scaler=scaler,
                                                last_known_data=last_data,
                                                features=features,
                                                days=forecast_days,
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