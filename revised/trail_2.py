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
from prophet import Prophet
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
    ]

# Load FinBERT sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except:
        return None

sentiment_pipeline = load_sentiment_model()

# Enhanced Indian stock symbols
@st.cache_data
def get_indian_stocks():
    return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
            "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "SBIN.NS"]

# Fetch India VIX data with proper timezone handling
@st.cache_data
def get_india_vix_data(start_date, end_date):
    """Fetch India VIX data from Yahoo Finance"""
    try:
        vix = yf.Ticker("^INDIAVIX")
        vix_data = vix.history(start=start_date, end=end_date)
        if not vix_data.empty:
            vix_data = vix_data[['Close']].rename(columns={'Close': 'India_VIX'})
            vix_data.index = vix_data.index.tz_localize(None)
            return vix_data
    except Exception as e:
        st.warning(f"Could not fetch India VIX data: {e}")
    return pd.DataFrame()

# Fetch FII/DII data (mock implementation)
@st.cache_data
def get_fii_dii_data(start_date, end_date):
    """Generate mock FII/DII data for demonstration"""
    try:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        fii_data = pd.DataFrame({
            'Date': dates,
            'FII_Net_Investment': np.random.normal(1000, 500, len(dates)),
            'DII_Net_Investment': np.random.normal(800, 400, len(dates))
        })
        fii_data.set_index('Date', inplace=True)
        return fii_data
    except Exception as e:
        st.warning(f"Could not generate FII/DII data: {e}")
    return pd.DataFrame()

# Enhanced stock data fetching with proper timezone handling
def get_stock_data(ticker, start, end):
    """Fetch stock data with additional market indicators"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end)
        if data.empty:
            st.error(f"No data found for {ticker} in the selected date range.")
            return pd.DataFrame()
        
        # Clean data and remove timezone
        data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        data.index = data.index.tz_localize(None)
        
        # Add India VIX data with proper date alignment
        vix_data = get_india_vix_data(start, end)
        if not vix_data.empty:
            data = data.merge(vix_data, left_index=True, right_index=True, how='left')
            data['India_VIX'] = data['India_VIX'].ffill()
        
        # Add FII/DII data with proper date alignment
        fii_dii_data = get_fii_dii_data(start, end)
        if not fii_dii_data.empty:
            data = data.merge(fii_dii_data, left_index=True, right_index=True, how='left')
            data[['FII_Net_Investment', 'DII_Net_Investment']] = data[['FII_Net_Investment', 'DII_Net_Investment']].fillna(0)
        
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# Enhanced feature engineering with market indicators
def create_advanced_features(df):
    """Create comprehensive technical and market features"""
    df = df.copy()
    
    # Basic price features
    df['Returns'] = df['Close'].pct_change()
    
    # Moving averages
    df['5D_MA'] = df['Close'].rolling(5).mean()
    df['20D_MA'] = df['Close'].rolling(20).mean()
    df['50D_MA'] = df['Close'].rolling(50).mean()
    
    # Technical indicators
    df['MA_Ratio'] = df['5D_MA'] / df['20D_MA']
    df['5D_Volatility'] = df['Returns'].rolling(5).std()
    
    # Volume features
    df['Volume_MA5'] = df['Volume'].rolling(5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Market sentiment features
    if 'India_VIX' in df.columns:
        df['VIX_Change'] = df['India_VIX'].pct_change()
    
    # FII/DII features
    if 'FII_Net_Investment' in df.columns:
        df['FII_Flow_5D'] = df['FII_Net_Investment'].rolling(5).sum()
        df['DII_Flow_5D'] = df['DII_Net_Investment'].rolling(5).sum()
        df['Net_FII_DII'] = df['FII_Flow_5D'] - df['DII_Flow_5D']
    
    return df

# Create features for prediction (without target variable)
def create_prediction_features(df):
    """Create features for prediction without target variable"""
    df = df.copy()
    
    # Basic price features
    df['Returns'] = df['Close'].pct_change()
    
    # Moving averages
    df['5D_MA'] = df['Close'].rolling(5).mean()
    df['20D_MA'] = df['Close'].rolling(20).mean()
    
    # Technical indicators
    df['MA_Ratio'] = df['5D_MA'] / df['20D_MA']
    df['5D_Volatility'] = df['Returns'].rolling(5).std()
    
    # Volume features
    df['Volume_MA5'] = df['Volume'].rolling(5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Market sentiment features
    if 'India_VIX' in df.columns:
        df['VIX_Change'] = df['India_VIX'].pct_change()
    
    # FII/DII features
    if 'FII_Net_Investment' in df.columns:
        df['FII_Flow_5D'] = df['FII_Net_Investment'].rolling(5).sum()
        df['DII_Flow_5D'] = df['DII_Net_Investment'].rolling(5).sum()
        df['Net_FII_DII'] = df['FII_Flow_5D'] - df['DII_Flow_5D']
    
    return df

# Enhanced hybrid model with multiple architectures
def create_enhanced_hybrid_model(df_stock, sentiment_features):
    """Create ensemble model with XGBoost, GRU, LSTM, BiLSTM"""
    
    # Prepare data with sentiment
    if sentiment_features:
        sentiment_df = pd.DataFrame(sentiment_features.items(), columns=["Date", "Sentiment"])
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None)
        df_stock.index = pd.to_datetime(df_stock.index)
        df_stock = df_stock.reset_index().merge(sentiment_df, on="Date", how="left").set_index("Date")
        df_stock['Sentiment'] = pd.to_numeric(df_stock['Sentiment'], errors='coerce').fillna(0)
    
    # Feature engineering
    df_stock = create_advanced_features(df_stock)
    if df_stock.empty:
        st.error("No features generated after preprocessing. Check data quality.")
        return None, None, None, None, None
    
    df_stock['Target'] = df_stock['Close'].pct_change().shift(-1)
    df_stock.dropna(inplace=True)
    
    if len(df_stock) < 50:
        st.error("Insufficient data for model training. Need at least 50 data points.")
        return None, None, None, None, None
    
    # Select features - only use basic features that will be available during prediction
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                    '5D_MA', '20D_MA', 'MA_Ratio', '5D_Volatility', 'Volume_Ratio', 'RSI']
    
    market_features = []
    if 'India_VIX' in df_stock.columns:
        market_features.extend(['India_VIX', 'VIX_Change'])
    if 'FII_Net_Investment' in df_stock.columns:
        market_features.extend(['FII_Flow_5D', 'DII_Flow_5D', 'Net_FII_DII'])
    if 'Sentiment' in df_stock.columns:
        market_features.append('Sentiment')
    
    features = base_features + market_features
    
    # Ensure all features exist
    available_features = [f for f in features if f in df_stock.columns]
    if len(available_features) < 5:
        st.error("Insufficient features for model training.")
        return None, None, None, None, None
    
    features = available_features
    
    # Train-test split
    X = df_stock[features]
    y = df_stock['Target']
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    if len(X_train) == 0 or len(X_test) == 0:
        st.error("Insufficient data for train-test split.")
        return None, None, None, None, None
    
    # Model training
    models = {}
    predictions = {}
    metrics = {}
    
    try:
        # 1. XGBoost Model
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        models['xgb'] = xgb_model
        predictions['xgb'] = xgb_pred
        
        # Prepare data for neural networks
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_3d = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_3d = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
        X_test_3d = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # 2. GRU Model
        gru_model = Sequential([
            GRU(32, input_shape=(1, X_train_3d.shape[2]), return_sequences=True),
            Dropout(0.2),
            GRU(16, return_sequences=False),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1)
        ])
        gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        gru_model.fit(X_train_3d, y_train, epochs=50, batch_size=16, 
                     validation_data=(X_test_3d, y_test), callbacks=callbacks, verbose=0)
        gru_pred = gru_model.predict(X_test_3d, verbose=0).flatten()
        models['gru'] = gru_model
        predictions['gru'] = gru_pred
        
        # 3. LSTM Model
        lstm_model = Sequential([
            LSTM(32, input_shape=(1, X_train_3d.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(16, return_sequences=False),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1)
        ])
        lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        lstm_model.fit(X_train_3d, y_train, epochs=50, batch_size=16, 
                      validation_data=(X_test_3d, y_test), callbacks=callbacks, verbose=0)
        lstm_pred = lstm_model.predict(X_test_3d, verbose=0).flatten()
        models['lstm'] = lstm_model
        predictions['lstm'] = lstm_pred
        
        # 4. BiLSTM Model
        bilstm_model = Sequential([
            Bidirectional(LSTM(16, return_sequences=True), input_shape=(1, X_train_3d.shape[2])),
            Dropout(0.2),
            Bidirectional(LSTM(8, return_sequences=False)),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1)
        ])
        bilstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        bilstm_model.fit(X_train_3d, y_train, epochs=50, batch_size=16, 
                        validation_data=(X_test_3d, y_test), callbacks=callbacks, verbose=0)
        bilstm_pred = bilstm_model.predict(X_test_3d, verbose=0).flatten()
        models['bilstm'] = bilstm_model
        predictions['bilstm'] = bilstm_pred
        
    except Exception as e:
        st.error(f"Error training models: {e}")
        return None, None, None, None, None
    
    # Calculate metrics for all models
    for model_name, pred in predictions.items():
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)
        
        metrics[model_name] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'accuracy': max(0, 100 - (mae * 100))
        }
    
    # Dynamic ensemble weighting based on performance
    model_weights = {}
    total_inverse_mae = sum(1/metrics[model]['mae'] for model in predictions.keys())
    for model_name in predictions.keys():
        model_weights[model_name] = (1/metrics[model_name]['mae']) / total_inverse_mae
    
    # Create ensemble prediction
    ensemble_pred = np.zeros_like(y_test)
    for model_name, weight in model_weights.items():
        ensemble_pred += weight * predictions[model_name]
    
    # Ensemble metrics
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_accuracy = max(0, 100 - (ensemble_mae * 100))
    
    metrics['ensemble'] = {
        'mae': ensemble_mae,
        'accuracy': ensemble_accuracy,
        'weights': model_weights
    }
    
    # Store predictions in dataframe
    df_stock.loc[y_test.index, 'Ensemble_Predicted'] = ensemble_pred
    for model_name, pred in predictions.items():
        df_stock.loc[y_test.index, f'{model_name}_Predicted'] = pred
    
    return df_stock, models, scaler, features, metrics

# FIXED: Enhanced forecasting with proper feature handling
def enhanced_hybrid_predict_prices(models, scaler, last_known_data, features, days=10):
    """Generate enhanced predictions with proper feature handling"""
    try:
        # Create features for the last known data
        last_data_with_features = create_prediction_features(last_known_data)
        last_data_with_features = last_data_with_features.dropna()
        
        if last_data_with_features.empty:
            st.error("No features could be created for prediction.")
            return pd.DataFrame()
        
        # Ensure all required features are present
        missing_features = [f for f in features if f not in last_data_with_features.columns]
        if missing_features:
            st.warning(f"Missing features for prediction: {missing_features}. Using available features.")
            available_features = [f for f in features if f in last_data_with_features.columns]
        else:
            available_features = features
        
        if len(available_features) < 5:
            st.error("Insufficient features for prediction.")
            return pd.DataFrame()
        
        # Get Prophet forecast for trend baseline
        prophet_forecast_df = prophet_forecast(last_known_data, days=days)
        future_dates = prophet_forecast_df.index
        
        # Initialize results
        future_prices = pd.DataFrame(index=future_dates, 
                                   columns=['Predicted Price', 'Daily Change (%)', 'Confidence'])
        
        current_data = last_data_with_features.copy()
        last_close = current_data['Close'].iloc[-1]
        
        for i, date in enumerate(future_dates):
            try:
                # Prepare input data with available features only
                current_features = current_data[available_features].iloc[-1:].copy()
                
                # Get predictions from all models
                model_predictions = {}
                
                # XGBoost prediction
                xgb_pred = models['xgb'].predict(current_features)[0]
                model_predictions['xgb'] = xgb_pred
                
                # Neural network predictions
                input_scaled = scaler.transform(current_features)
                input_3d = input_scaled.reshape(1, 1, input_scaled.shape[1])
                
                for nn_model in ['gru', 'lstm', 'bilstm']:
                    if nn_model in models:
                        pred = models[nn_model].predict(input_3d, verbose=0)[0][0]
                        model_predictions[nn_model] = pred
                
                # Prophet prediction
                prophet_pred = (prophet_forecast_df['yhat'].iloc[i] - last_close) / last_close
                model_predictions['prophet'] = prophet_pred
                
                # Calculate weighted ensemble
                weights = {'xgb': 0.3, 'gru': 0.2, 'lstm': 0.2, 'bilstm': 0.2, 'prophet': 0.1}
                final_pred = 0
                total_weight = 0
                
                for model_name, pred in model_predictions.items():
                    if model_name in weights:
                        final_pred += weights[model_name] * pred
                        total_weight += weights[model_name]
                
                if total_weight > 0:
                    final_pred /= total_weight
                
                # Apply realistic bounds
                max_daily_change = 0.08
                final_pred = np.clip(final_pred, -max_daily_change, max_daily_change)
                
                # Calculate new price
                new_close = last_close * (1 + final_pred)
                
                # Store results
                future_prices.loc[date, 'Predicted Price'] = new_close
                future_prices.loc[date, 'Confidence'] = 75
                
                # Update last_close for next iteration
                last_close = new_close
                
                # Create a simple new row for the next prediction (without complex feature updates)
                new_row = {
                    'Open': new_close * 0.998,
                    'High': new_close * 1.005,
                    'Low': new_close * 0.995,
                    'Close': new_close,
                    'Volume': current_data['Volume'].iloc[-1],  # Keep volume constant for simplicity
                }
                
                # Add market data if available
                if 'India_VIX' in current_data.columns:
                    new_row['India_VIX'] = current_data['India_VIX'].iloc[-1]
                if 'FII_Net_Investment' in current_data.columns:
                    new_row['FII_Net_Investment'] = current_data['FII_Net_Investment'].iloc[-1]
                    new_row['DII_Net_Investment'] = current_data['DII_Net_Investment'].iloc[-1]
                
                # Create new DataFrame row and update current_data
                new_row_df = pd.DataFrame([new_row], index=[date])
                current_data = pd.concat([current_data, new_row_df])
                
                # Recalculate features for the new data point
                current_data = create_prediction_features(current_data)
                current_data = current_data.dropna()
                
            except Exception as e:
                st.error(f"Error in prediction step {i}: {e}")
                continue
        
        # Calculate daily changes
        future_prices['Daily Change (%)'] = future_prices['Predicted Price'].pct_change().fillna(0) * 100
        
        return future_prices
    
    except Exception as e:
        st.error(f"Enhanced forecast generation failed: {str(e)}")
        return pd.DataFrame()

# Prophet forecasting function
def prophet_forecast(df, days=10):
    """Prophet-based forecasting"""
    try:
        prophet_df = df.reset_index()[['Date', 'Close']].rename(
            columns={'Date': 'ds', 'Close': 'y'}
        )
        
        model = Prophet(daily_seasonality=True, yearly_seasonality=False)
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=days, include_history=False)
        forecast = model.predict(future)
        
        forecast.set_index('ds', inplace=True)
        return forecast[['yhat']]
    except Exception as e:
        st.warning(f"Prophet forecast failed: {e}")
        # Return simple forecast as fallback
        last_price = df['Close'].iloc[-1]
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days, freq='D')
        return pd.DataFrame({'yhat': [last_price] * days}, index=future_dates)

# Display functions
def display_stock_info(ticker, df_stock):
    """Display stock fundamental information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"₹{df_stock['Close'].iloc[-1]:.2f}")
            st.metric("Market Cap", f"₹{info.get('marketCap', 0)/1e7:.0f} Cr" if info.get('marketCap') else "N/A")
        
        with col2:
            st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            st.metric("ROCE", f"{info.get('returnOnCapitalEmployed', 0)*100:.1f}%" if info.get('returnOnCapitalEmployed') else "N/A")
        
        with col3:
            st.metric("Book Value", f"₹{info.get('bookValue', 'N/A')}")
            st.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%" if info.get('returnOnEquity') else "N/A")
        
        with col4:
            st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")
            st.metric("Volume", f"{df_stock['Volume'].iloc[-1]:,}")
    
    except Exception as e:
        st.warning(f"Could not fetch complete fundamental data: {e}")

def display_price_charts(df_stock, selected_stock):
    """Display interactive price charts"""
    
    tab1, tab2, tab3 = st.tabs(["Candlestick", "Trend Analysis", "Technical Indicators"])
    
    with tab1:
        fig = go.Figure(data=[go.Candlestick(
            x=df_stock.index,
            open=df_stock['Open'],
            high=df_stock['High'],
            low=df_stock['Low'],
            close=df_stock['Close'],
            name='Price'
        )])
        fig.update_layout(title=f"{selected_stock} Candlestick Chart", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['Close'], name='Close Price'))
        if '5D_MA' in df_stock.columns:
            fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['5D_MA'], name='5D MA'))
        if '20D_MA' in df_stock.columns:
            fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['20D_MA'], name='20D MA'))
        fig.update_layout(title="Price Trend with Moving Averages", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("RSI")
            if 'RSI' in df_stock.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['RSI'], name='RSI'))
                fig.add_hline(y=70, line_dash="dash", line_color="red")
                fig.add_hline(y=30, line_dash="dash", line_color="green")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Volatility")
            if '5D_Volatility' in df_stock.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['5D_Volatility'], name='5D Volatility'))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

def display_market_indicators(df_stock):
    """Display market environment indicators"""
    
    if 'India_VIX' in df_stock.columns:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_vix = df_stock['India_VIX'].iloc[-1]
            vix_status = "High" if current_vix > 25 else "Medium" if current_vix > 15 else "Low"
            st.metric("India VIX", f"{current_vix:.2f}", vix_status)
        
        with col2:
            if 'FII_Net_Investment' in df_stock.columns:
                recent_fii = df_stock['FII_Net_Investment'].tail(5).mean()
                st.metric("FII Flow (5D Avg)", f"₹{recent_fii:.0f} Cr")
        
        with col3:
            if 'DII_Net_Investment' in df_stock.columns:
                recent_dii = df_stock['DII_Net_Investment'].tail(5).mean()
                st.metric("DII Flow (5D Avg)", f"₹{recent_dii:.0f} Cr")
        
        # VIX chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['India_VIX'], name='India VIX'))
        fig.update_layout(title="India VIX Trend", height=300)
        st.plotly_chart(fig, use_container_width=True)

def display_model_comparison(metrics):
    """Display comprehensive model comparison"""
    
    st.subheader("📊 Model Performance Comparison")
    
    # Create metrics table
    models_list = ['xgb', 'gru', 'lstm', 'bilstm', 'ensemble']
    model_names = ['XGBoost', 'GRU', 'LSTM', 'BiLSTM', 'Ensemble']
    
    metrics_data = []
    for model, name in zip(models_list, model_names):
        if model in metrics:
            model_metrics = metrics[model]
            metrics_data.append({
                'Model': name,
                'MAE': model_metrics.get('mae', 0),
                'RMSE': model_metrics.get('rmse', 0),
                'R²': model_metrics.get('r2', 0),
                'Accuracy': model_metrics.get('accuracy', 0)
            })
    
    if not metrics_data:
        st.warning("No model metrics available.")
        return
        
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display metrics table
    st.dataframe(
        metrics_df.style.format({
            'MAE': '{:.4f}',
            'RMSE': '{:.4f}',
            'R²': '{:.3f}',
            'Accuracy': '{:.1f}%'
        }).highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
        .highlight_max(subset=['R²', 'Accuracy'], color='lightgreen'),
        use_container_width=True
    )
    
    # Visual comparison
    fig = go.Figure()
    
    for metric in ['MAE', 'RMSE']:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Model'],
            y=metrics_df[metric],
            text=metrics_df[metric].round(4),
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Model Error Metrics Comparison',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def display_forecast_results(future_prices, df_stock, selected_stock, metrics):
    """Display forecast results and visualization"""
    
    # Forecast table
    st.subheader("📅 10-Day Price Forecast")
    
    forecast_display = future_prices[['Predicted Price', 'Daily Change (%)']].copy()
    forecast_display['Predicted Price'] = forecast_display['Predicted Price'].apply(lambda x: f'₹{x:,.2f}')
    forecast_display['Daily Change (%)'] = forecast_display['Daily Change (%)'].apply(lambda x: f'{x:+.2f}%')
    
    st.dataframe(forecast_display)
    
    # Price trend visualization
    st.subheader("📈 Historical vs Forecast Trend")
    
    historical = df_stock[['Close']].rename(columns={'Close': 'Price'}).iloc[-60:]
    forecast = future_prices[['Predicted Price']].rename(columns={'Predicted Price': 'Price'})
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical['Price'],
        mode='lines',
        name='Historical',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Forecast data
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast['Price'],
        mode='lines+markers',
        name='AI Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title=f"{selected_stock} Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode="x unified",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_investment_recommendation(future_prices, df_stock, metrics):
    """Generate and display investment recommendation"""
    
    if future_prices.empty:
        return
    
    current_price = df_stock['Close'].iloc[-1]
    avg_predicted = future_prices['Predicted Price'].mean()
    price_change_pct = ((avg_predicted - current_price) / current_price) * 100
    
    ensemble_accuracy = metrics.get('ensemble', {}).get('accuracy', 70)
    
    # Recommendation logic
    if price_change_pct > 10 and ensemble_accuracy > 75:
        recommendation = "STRONG BUY"
        reasoning = "High confidence in significant upward movement"
        color = "green"
    elif price_change_pct > 5 and ensemble_accuracy > 70:
        recommendation = "BUY"
        reasoning = "Good potential for moderate gains"
        color = "lightgreen"
    elif price_change_pct > 2 and ensemble_accuracy > 65:
        recommendation = "HOLD (Positive Bias)"
        reasoning = "Potential for slight appreciation"
        color = "blue"
    elif price_change_pct < -10 and ensemble_accuracy > 75:
        recommendation = "STRONG SELL"
        reasoning = "High confidence in significant decline"
        color = "darkred"
    elif price_change_pct < -5 and ensemble_accuracy > 70:
        recommendation = "SELL"
        reasoning = "Significant downside risk"
        color = "red"
    elif price_change_pct < -2 and ensemble_accuracy > 65:
        recommendation = "HOLD (Caution)"
        reasoning = "Potential for slight decline"
        color = "orange"
    else:
        recommendation = "HOLD"
        reasoning = "Neutral outlook with balanced risk-reward"
        color = "gray"
    
    st.header("💡 Investment Recommendation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Expected Return", f"{price_change_pct:+.1f}%")
    
    with col2:
        st.metric("Model Confidence", f"{ensemble_accuracy:.1f}%")
    
    with col3:
        st.metric("Recommendation", recommendation)
    
    # Recommendation box
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {color}; color: white; margin: 10px 0;">
        <h3 style="margin: 0;">{recommendation}</h3>
        <p style="margin: 10px 0 0 0;">{reasoning}</p>
    </div>
    """, unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Advanced Indian Stock Analysis", layout="wide")
    
    st.title("🧠 Advanced Indian Stock Market Analysis with Multi-Model AI")
    st.markdown("""
    **Academic Research-Grade Analysis** featuring:
    - **5 AI Models**: XGBoost, GRU, LSTM, BiLSTM, Prophet Ensemble
    - **Market Indicators**: India VIX, FII/DII Data
    - **Comprehensive Metrics**: MAE, MSE, RMSE, R², Accuracy Scores
    """)
    
    # Sidebar
    st.sidebar.header("🔧 Analysis Configuration")
    indian_stocks = get_indian_stocks()
    selected_stock = st.sidebar.selectbox("Select Stock", indian_stocks, index=0)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.date(2023, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.date.today())
    
    if st.sidebar.button("🚀 Run Advanced Analysis", type="primary"):
        with st.spinner("Running comprehensive AI analysis..."):
            
            # Get stock data
            df_stock = get_stock_data(selected_stock, start_date, end_date)
            
            if df_stock.empty:
                st.error("No data available for the selected stock and date range.")
                return
            
            # Display basic info
            st.header(f"📊 Fundamental Analysis: {selected_stock}")
            display_stock_info(selected_stock, df_stock)
            
            # Display charts
            st.header("📈 Price Analysis")
            display_price_charts(df_stock, selected_stock.replace('.NS', ''))
            
            # Market data analysis
            st.header("🌐 Market Environment Analysis")
            display_market_indicators(df_stock)
            
            # AI Model Analysis
            st.header("🤖 Multi-Model AI Analysis")
            with st.spinner("Training 5 AI models... This may take a few minutes."):
                result = create_enhanced_hybrid_model(df_stock, {})
                
                if result[0] is None:
                    st.error("Model training failed. Please try with a different date range or stock.")
                    return
                    
                df_enhanced, models, scaler, features, metrics = result
            
            # Display model comparison
            display_model_comparison(metrics)
            
            # Generate forecasts
            st.header("🔮 10-Day Price Forecast")
            last_data = df_stock.iloc[-30:]
            future_prices = enhanced_hybrid_predict_prices(models, scaler, last_data, features, days=10)
            
            if not future_prices.empty:
                display_forecast_results(future_prices, df_stock, selected_stock.replace('.NS', ''), metrics)
                display_investment_recommendation(future_prices, df_stock, metrics)

if __name__ == "__main__":
    main()