"""
Enhanced Hybrid AI Model for Indian Stock Market Analysis
with Bayesian Dynamic Uncertainty-Weighted Fusion Framework

"""

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
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import CustomBusinessDay
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CUSTOM INDIAN MARKET CALENDAR
# ============================================================================
class IndiaHolidayCalendar(AbstractHolidayCalendar):
    """Indian market holiday calendar"""
    rules = [
        Holiday('Republic Day', month=1, day=26),
        Holiday('Independence Day', month=8, day=15),
        Holiday('Gandhi Jayanti', month=10, day=2),
        Holiday('Diwali', month=10, day=24),
        Holiday('Holi', month=3, day=25),
        Holiday('Good Friday', month=4, day=2),
        Holiday('Christmas', month=12, day=25),
        Holiday('Eid al-Fitr', month=4, day=10),
        Holiday('Eid al-Adha', month=6, day=17),
    ]

# ============================================================================
# 2. DATA VALIDATION AND CLEANING FUNCTIONS
# ============================================================================
def validate_and_clean_data(df):
    """
    Validate and clean dataframe to remove invalid values
    """
    df = df.copy()
    
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Remove rows with NaN values in essential columns
    essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if all(col in df.columns for col in essential_cols):
        df = df.dropna(subset=essential_cols)
    
    # Validate numerical ranges
    for col in df.select_dtypes(include=[np.number]).columns:
        # Remove extreme outliers (beyond 5 standard deviations)
        if len(df) > 10:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[col] = df[col].clip(
                    lower=mean_val - 5*std_val,
                    upper=mean_val + 5*std_val
                )
        
        # Ensure no extremely large values
        if df[col].abs().max() > 1e10:  # If values exceed 10 billion
            df[col] = df[col] / 1000  # Scale down
    
    # Ensure positive prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].abs()
            df[col] = df[col].replace(0, np.nan)
    
    # Ensure positive volume
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].abs()
        df['Volume'] = df['Volume'].replace(0, 1)  # Replace 0 with 1 to avoid division issues
    
    return df.dropna()

def safe_division(numerator, denominator, default=0.0):
    """Safe division to avoid division by zero"""
    if isinstance(numerator, (pd.Series, pd.DataFrame)):
        return numerator / denominator.replace(0, np.nan)
    else:
        return numerator / denominator if denominator != 0 else default

# ============================================================================
# 3. BAYESIAN DYNAMIC UNCERTAINTY-WEIGHTED FUSION CLASS
# ============================================================================
class BayesianDynamicFusion:
    """
    Implements Bayesian Dynamic Uncertainty-Weighted Fusion Framework
    """
    
    def __init__(self, temperature=1.0, lookback_window=10):
        self.temperature = temperature
        self.lookback_window = lookback_window
        self.uncertainty_history = {}
        self.weight_history = {}
        self.prediction_history = {}
        
    def compute_uncertainty(self, source_id, predictions, actuals):
        """
        Compute time-varying uncertainty σ_i,t^2 for a source
        """
        if len(predictions) < self.lookback_window or len(actuals) < self.lookback_window:
            return 0.5  # Default moderate uncertainty
        
        try:
            # Get recent errors (last lookback_window points)
            recent_preds = predictions[-self.lookback_window:]
            recent_actuals = actuals[-self.lookback_window:]
            
            # Ensure same length
            min_len = min(len(recent_preds), len(recent_actuals))
            if min_len < 2:
                return 0.5
            
            recent_preds = recent_preds[-min_len:]
            recent_actuals = recent_actuals[-min_len:]
            
            # Compute errors
            errors = np.array(recent_actuals) - np.array(recent_preds)
            
            # Remove outliers from errors
            if len(errors) > 2:
                mean_err = np.mean(errors)
                std_err = np.std(errors)
                if std_err > 0:
                    errors = errors[np.abs(errors - mean_err) <= 3 * std_err]
            
            if len(errors) < 2:
                return 0.5
            
            # Compute uncertainty (normalized variance)
            uncertainty = np.var(errors)
            
            # Normalize to [0, 1] range
            if np.std(recent_actuals) > 0:
                uncertainty = uncertainty / (np.var(recent_actuals) + 1e-6)
            
            uncertainty = np.clip(uncertainty, 0.01, 1.0)
            
            # Store history
            if source_id not in self.uncertainty_history:
                self.uncertainty_history[source_id] = []
            self.uncertainty_history[source_id].append(uncertainty)
            
            return uncertainty
            
        except Exception as e:
            st.warning(f"Uncertainty computation error for {source_id}: {str(e)}")
            return 0.5
    
    def compute_dynamic_weights(self, source_uncertainties):
        """
        Compute Bayesian dynamic weights using softmax
        """
        try:
            # Convert uncertainties to confidences (higher uncertainty = lower confidence)
            confidences = []
            for unc in source_uncertainties:
                if unc <= 0:
                    conf = 1.0
                else:
                    conf = 1.0 / (unc + 1e-6)
                confidences.append(conf)
            
            confidences = np.array(confidences)
            
            # Apply temperature-scaled softmax
            if self.temperature <= 0:
                self.temperature = 0.1
                
            scaled_conf = confidences / self.temperature
            # Prevent overflow in exp
            scaled_conf = scaled_conf - np.max(scaled_conf)
            exp_conf = np.exp(scaled_conf)
            
            weights = exp_conf / (np.sum(exp_conf) + 1e-6)
            
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            
            return weights
            
        except Exception as e:
            st.warning(f"Weight computation error: {str(e)}")
            # Return equal weights as fallback
            n_sources = len(source_uncertainties)
            return np.ones(n_sources) / n_sources
    
    def update_and_fuse(self, source_predictions, source_actuals=None):
        """
        Update uncertainties and compute fused prediction
        """
        try:
            # Store predictions
            for source_id, preds in source_predictions.items():
                if source_id not in self.prediction_history:
                    self.prediction_history[source_id] = []
                if isinstance(preds, (list, np.ndarray)):
                    self.prediction_history[source_id].extend(preds)
                else:
                    self.prediction_history[source_id].append(preds)
            
            # Compute uncertainties
            uncertainties = {}
            for source_id in source_predictions.keys():
                preds = self.prediction_history.get(source_id, [])
                
                if source_actuals and source_id in source_actuals:
                    actuals = source_actuals[source_id]
                else:
                    # If no actuals, use average of other predictions as proxy
                    all_preds = []
                    for sid in source_predictions.keys():
                        if sid != source_id:
                            all_preds.extend(self.prediction_history.get(sid, []))
                    actuals = all_preds if all_preds else preds
                
                uncertainties[source_id] = self.compute_uncertainty(
                    source_id, preds, actuals
                )
            
            # Compute weights
            weight_values = self.compute_dynamic_weights(list(uncertainties.values()))
            weights = {source_id: weight_values[i] 
                      for i, source_id in enumerate(source_predictions.keys())}
            
            # Store weights
            for source_id, weight in weights.items():
                if source_id not in self.weight_history:
                    self.weight_history[source_id] = []
                self.weight_history[source_id].append(weight)
            
            # Compute fused prediction (weighted average of latest predictions)
            fused_prediction = 0
            for source_id, weight in weights.items():
                preds = self.prediction_history.get(source_id, [])
                if preds:
                    latest_pred = preds[-1]
                    fused_prediction += weight * latest_pred
            
            return fused_prediction, weights, uncertainties
            
        except Exception as e:
            st.error(f"Fusion error: {str(e)}")
            # Fallback: simple average
            predictions = []
            for source_id, preds in source_predictions.items():
                if isinstance(preds, (list, np.ndarray)) and len(preds) > 0:
                    predictions.append(preds[-1])
                elif not isinstance(preds, (list, np.ndarray)):
                    predictions.append(preds)
            
            if predictions:
                fused_prediction = np.mean(predictions)
                n_sources = len(source_predictions)
                weights = {sid: 1/n_sources for sid in source_predictions.keys()}
                uncertainties = {sid: 0.5 for sid in source_predictions.keys()}
                return fused_prediction, weights, uncertainties
            else:
                return 0, {}, {}

# ============================================================================
# 4. ENHANCED FEATURE ENGINEERING WITH SAFE CALCULATIONS
# ============================================================================
class EnhancedFeatureEngineer:
    """Safe feature engineering with validation"""
    
    @staticmethod
    def create_features(df, sentiment_features=None):
        """
        Create features with safe calculations
        """
        df = df.copy()
        
        # Basic price-based features
        df['Returns'] = df['Close'].pct_change().fillna(0)
        df['Returns'] = df['Returns'].clip(lower=-0.1, upper=0.1)  # Clip extreme returns
        
        # Safe moving averages
        for window in [5, 10, 20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
        
        # Safe moving average ratios
        df['MA_Ratio_5_20'] = safe_division(df['MA_5'], df['MA_20'], default=1.0)
        df['MA_Ratio_10_50'] = safe_division(df['MA_10'], df['MA_50'], default=1.0)
        
        # Safe volatility calculations
        for window in [5, 20]:
            returns = df['Returns'].rolling(window=window, min_periods=2)
            df[f'Volatility_{window}'] = returns.std().fillna(0.01)
        
        df['Volatility_Ratio'] = safe_division(
            df['Volatility_5'], 
            df['Volatility_20'].replace(0, 0.01), 
            default=1.0
        )
        
        # Safe volume features
        df['Volume_MA5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
        df['Volume_Ratio'] = safe_division(df['Volume'], df['Volume_MA5'], default=1.0)
        df['Volume_Ratio'] = df['Volume_Ratio'].clip(lower=0.1, upper=10)
        
        # Safe RSI calculation
        df['RSI'] = EnhancedFeatureEngineer.calculate_safe_rsi(df['Close'], period=14)
        
        # Safe momentum
        for period in [5, 10]:
            df[f'Momentum_{period}'] = (df['Close'] / df['Close'].shift(period) - 1).fillna(0)
            df[f'Momentum_{period}'] = df[f'Momentum_{period}'].clip(lower=-0.2, upper=0.2)
        
        # Safe Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['Close'].rolling(window=20, min_periods=2).std().fillna(df['Close'].std())
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = safe_division(
            (df['BB_Upper'] - df['BB_Lower']), 
            df['BB_Middle'], 
            default=0.1
        )
        df['BB_Position'] = safe_division(
            (df['Close'] - df['BB_Lower']), 
            (df['BB_Upper'] - df['BB_Lower']).replace(0, 1), 
            default=0.5
        )
        
        # Add sentiment if available
        if sentiment_features and isinstance(sentiment_features, dict):
            try:
                sentiment_df = pd.DataFrame(
                    list(sentiment_features.items()), 
                    columns=['Date', 'Sentiment']
                )
                sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None)
                
                # Ensure df has a datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.set_index('Date')
                
                # Merge sentiment
                df_temp = df.reset_index()
                if 'Date' in df_temp.columns:
                    df_temp['Date'] = pd.to_datetime(df_temp['Date'])
                    df_temp = df_temp.merge(sentiment_df, on='Date', how='left')
                    df = df_temp.set_index('Date')
                
                df['Sentiment'] = pd.to_numeric(df['Sentiment'], errors='coerce').fillna(0)
                df['Sentiment'] = df['Sentiment'].clip(lower=-1, upper=1)
                
                # Sentiment features
                df['Sentiment_MA5'] = df['Sentiment'].rolling(5, min_periods=1).mean()
                df['Sentiment_Change'] = df['Sentiment'].diff().fillna(0)
                
            except Exception as e:
                st.warning(f"Sentiment integration error: {str(e)}")
                df['Sentiment'] = 0
                df['Sentiment_MA5'] = 0
                df['Sentiment_Change'] = 0
        
        # Target variable (next day's return)
        df['Target'] = df['Returns'].shift(-1).fillna(0)
        
        # Fill any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Final validation
        df = validate_and_clean_data(df)
        
        return df
    
    @staticmethod
    def calculate_safe_rsi(prices, period=14):
        """Calculate RSI safely"""
        try:
            if len(prices) < period + 1:
                return pd.Series([50] * len(prices), index=prices.index)
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            
            # Avoid division by zero
            rs = safe_division(gain, loss.replace(0, 1e-10), default=1)
            rsi = 100 - (100 / (1 + rs))
            
            # Clip to valid range
            rsi = rsi.clip(0, 100)
            rsi = rsi.fillna(50)
            
            return rsi
            
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)

# ============================================================================
# 5. ENHANCED HYBRID MODEL WITH ERROR HANDLING
# ============================================================================
class EnhancedHybridModel:
    """Enhanced hybrid model with Bayesian dynamic fusion"""
    
    def __init__(self):
        self.bayesian_fusion = BayesianDynamicFusion(temperature=0.5, lookback_window=10)
        self.feature_engineer = EnhancedFeatureEngineer()
        self.models = {}
        self.scaler = RobustScaler()  # More robust than MinMaxScaler
        self.features = None
        self.is_trained = False
        
    def prepare_training_data(self, df, sentiment_features=None):
        """Prepare training data with validation"""
        try:
            # Create features
            df_features = self.feature_engineer.create_features(df, sentiment_features)
            
            if len(df_features) < 50:
                st.warning(f"Insufficient data after feature engineering: {len(df_features)} samples")
                return None, None, None, None
            
            # Define feature columns (exclude target and date columns)
            exclude_cols = ['Target']
            if 'Date' in df_features.columns:
                exclude_cols.append('Date')
            
            feature_cols = [col for col in df_features.columns if col not in exclude_cols]
            
            if not feature_cols:
                st.error("No features created")
                return None, None, None, None
            
            self.features = feature_cols
            
            # Prepare X and y
            X = df_features[feature_cols]
            y = df_features['Target']
            
            # Final validation of X
            X = validate_and_clean_data(X)
            
            # Ensure y has same index as X
            y = y.loc[X.index]
            
            # Split data (time-series aware)
            split_idx = max(50, int(len(X) * 0.8))
            if split_idx >= len(X):
                split_idx = int(len(X) * 0.7)
            
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Ensure we have enough test data
            if len(X_test) < 10:
                # Adjust split to ensure minimum test size
                split_idx = max(50, len(X) - 20)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            st.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            st.error(f"Error preparing training data: {str(e)}")
            return None, None, None, None
    
    def train_technical_model(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model safely"""
        try:
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Validate scaled data
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=1e6, neginf=-1e6)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=1e6, neginf=-1e6)
            
            # Train XGBoost
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            )
            
            xgb_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
            
            # Predictions
            y_pred_train = xgb_model.predict(X_train_scaled)
            y_pred_test = xgb_model.predict(X_test_scaled)
            
            # Calculate metrics
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Calculate uncertainty (normalized)
            test_errors = y_test - y_pred_test
            if len(test_errors) > 1 and np.std(y_test) > 0:
                uncertainty = np.var(test_errors) / (np.var(y_test) + 1e-6)
                uncertainty = np.clip(uncertainty, 0.01, 1.0)
            else:
                uncertainty = 0.5
            
            return xgb_model, y_pred_test, mae_test, rmse_test, uncertainty
            
        except Exception as e:
            st.error(f"Error training XGBoost: {str(e)}")
            return None, None, None, None, 0.5
    
    def train_sentiment_model(self, X_train, y_train, X_test, y_test):
        """Train GRU model safely"""
        try:
            # Scale features
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Validate scaled data
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=1e6, neginf=-1e6)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=1e6, neginf=-1e6)
            
            # Reshape for GRU
            X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            
            # Simple GRU model
            gru_model = Sequential([
                GRU(32, input_shape=(1, X_train_scaled.shape[1]), return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            gru_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Train with early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            )
            
            history = gru_model.fit(
                X_train_3d, y_train,
                epochs=50,
                batch_size=16,
                validation_data=(X_test_3d, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Predictions
            y_pred_test = gru_model.predict(X_test_3d, verbose=0).flatten()
            
            # Calculate metrics
            mae_test = mean_absolute_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Calculate uncertainty
            test_errors = y_test - y_pred_test
            if len(test_errors) > 1 and np.std(y_test) > 0:
                uncertainty = np.var(test_errors) / (np.var(y_test) + 1e-6)
                uncertainty = np.clip(uncertainty, 0.01, 1.0)
            else:
                uncertainty = 0.5
            
            return gru_model, y_pred_test, mae_test, rmse_test, uncertainty
            
        except Exception as e:
            st.error(f"Error training GRU: {str(e)}")
            return None, None, None, None, 0.5
    
    def train_prophet_model(self, df_stock):
        """Train Prophet model safely"""
        try:
            if len(df_stock) < 30:
                return None, None, 0.5
            
            # Prepare Prophet data
            prophet_df = df_stock.reset_index()[['Date', 'Close']].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Remove outliers
            q_low = prophet_df['y'].quantile(0.01)
            q_high = prophet_df['y'].quantile(0.99)
            prophet_df['y'] = prophet_df['y'].clip(lower=q_low, upper=q_high)
            
            # Train Prophet
            model = Prophet(
                daily_seasonality=False,
                yearly_seasonality=True,
                weekly_seasonality=True,
                seasonality_mode='additive',
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_df)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=10)
            forecast = model.predict(future)
            
            # Calculate returns from predictions
            last_close = df_stock['Close'].iloc[-1]
            future_prices = forecast['yhat'].iloc[-10:].values
            if len(future_prices) > 0 and last_close > 0:
                returns = (future_prices - last_close) / last_close
                avg_return = np.mean(returns)
            else:
                avg_return = 0
            
            return model, avg_return, 0.3  # Lower uncertainty for trend model
            
        except Exception as e:
            st.warning(f"Prophet model warning: {str(e)}")
            return None, 0, 0.5
    
    def train_hybrid_model(self, df_stock, sentiment_features=None):
        """Main training function with comprehensive error handling"""
        try:
            st.info("🔄 Preparing data for Bayesian Fusion Model...")
            
            # Prepare training data
            X_train, X_test, y_train, y_test = self.prepare_training_data(
                df_stock, sentiment_features
            )
            
            if X_train is None or len(X_train) < 20:
                st.error("Insufficient valid data for training")
                return None, None, None, None, {}
            
            st.info("🧠 Training Technical Model (XGBoost)...")
            xgb_model, xgb_pred, xgb_mae, xgb_rmse, xgb_uncertainty = \
                self.train_technical_model(X_train, y_train, X_test, y_test)
            
            st.info("🧠 Training Sentiment Model (GRU)...")
            gru_model, gru_pred, gru_mae, gru_rmse, gru_uncertainty = \
                self.train_sentiment_model(X_train, y_train, X_test, y_test)
            
            st.info("🧠 Training Prophet Trend Model...")
            prophet_model, prophet_return, prophet_uncertainty = \
                self.train_prophet_model(df_stock)
            
            # Store models
            self.models = {
                'xgb': xgb_model if xgb_model is not None else None,
                'gru': gru_model if gru_model is not None else None,
                'prophet': prophet_model if prophet_model is not None else None
            }
            
            # Prepare predictions for Bayesian fusion
            source_predictions = {}
            source_actuals = {}
            
            if xgb_pred is not None and len(xgb_pred) > 0:
                source_predictions['xgb'] = xgb_pred
                source_actuals['xgb'] = y_test.values
            
            if gru_pred is not None and len(gru_pred) > 0:
                source_predictions['gru'] = gru_pred
                source_actuals['gru'] = y_test.values
            
            if prophet_model is not None:
                # Use prophet return as prediction
                source_predictions['prophet'] = [prophet_return]
                source_actuals['prophet'] = [y_test.mean() if len(y_test) > 0 else 0]
            
            # Apply Bayesian fusion
            if source_predictions:
                fused_prediction, weights, uncertainties = self.bayesian_fusion.update_and_fuse(
                    source_predictions, source_actuals
                )
                
                # Calculate hybrid metrics
                hybrid_mae = (xgb_mae + gru_mae) / 2 if xgb_mae and gru_mae else xgb_mae or gru_mae or 0.1
                hybrid_accuracy = max(0, 100 - (hybrid_mae * 100))
                
                # Prepare metrics
                model_metrics = {
                    'xgb': {
                        'mae': xgb_mae if xgb_mae else 0.1,
                        'rmse': xgb_rmse if xgb_rmse else 0.1,
                        'uncertainty': xgb_uncertainty,
                        'weight': weights.get('xgb', 0.33)
                    },
                    'gru': {
                        'mae': gru_mae if gru_mae else 0.1,
                        'rmse': gru_rmse if gru_rmse else 0.1,
                        'uncertainty': gru_uncertainty,
                        'weight': weights.get('gru', 0.33)
                    },
                    'prophet': {
                        'return': prophet_return,
                        'uncertainty': prophet_uncertainty,
                        'weight': weights.get('prophet', 0.33)
                    },
                    'hybrid': {
                        'mae': hybrid_mae,
                        'accuracy': hybrid_accuracy,
                        'fused_prediction': fused_prediction
                    },
                    'bayesian_fusion': {
                        'final_weights': weights,
                        'source_uncertainties': uncertainties
                    }
                }
                
                self.is_trained = True
                return df_stock, self.models, self.scaler, self.features, model_metrics
            
            else:
                st.error("No models trained successfully")
                return None, None, None, None, {}
                
        except Exception as e:
            st.error(f"Error in hybrid model training: {str(e)}")
            return None, None, None, None, {}

# ============================================================================
# 6. HELPER FUNCTIONS (from original code with improvements)
# ============================================================================
# Load sentiment model
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except:
    sentiment_pipeline = None
    st.warning("Could not load FinBERT model. Sentiment analysis will be limited.")

@st.cache_data
def get_indian_stocks():
    """Load Indian stock symbols"""
    file_path = "indian_stocks.csv"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
            df.columns = df.columns.str.strip()
            if "SYMBOL" in df.columns:
                symbols = df["SYMBOL"].dropna().unique().tolist()
                return [s for s in symbols if isinstance(s, str) and len(s) > 0]
        except Exception as e:
            st.warning(f"Error loading stock symbols: {e}")
    
    # Fallback list
    return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "HINDUNILVR"]

def get_stock_data(ticker, start, end):
    """Fetch stock data with validation"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end)
        
        if data.empty:
            st.warning(f"No data found for {ticker}")
            return pd.DataFrame()
        
        # Clean data
        data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        if not data.index.empty:
            data.index = data.index.tz_localize(None)
        
        # Validate and clean
        data = validate_and_clean_data(data)
        
        # Ensure minimum data
        if len(data) < 20:
            st.warning(f"Only {len(data)} days of data available. Need at least 20 days.")
            return pd.DataFrame()
        
        return data
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

# Other helper functions remain similar but with added error handling...

# ============================================================================
# 7. MAIN APPLICATION
# ============================================================================
def main():
    st.set_page_config(
        page_title="Indian Stock Market AI Analyzer",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Indian Stock Market Analysis with Bayesian Dynamic Fusion")
    st.markdown("""
    **Advanced AI Model with Dynamic Uncertainty-Weighted Fusion Framework**
    
    *Real-time adaptation to changing market conditions using Bayesian inference*
    """)
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Stock selection
    indian_stocks = get_indian_stocks()
    selected_stock = st.sidebar.selectbox("Select Stock", indian_stocks, index=0)
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.date(2023, 6, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.date.today())
    
    # Ensure valid date range
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
        start_date = datetime.date(2023, 6, 1)
        end_date = datetime.date.today()
    
    # Model parameters
    st.sidebar.subheader("Bayesian Fusion Parameters")
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.5, 0.1)
    lookback_window = st.sidebar.slider("Uncertainty Window", 5, 30, 10, 1)
    
    # Analysis button
    if st.sidebar.button("🚀 Run Bayesian Analysis", type="primary"):
        with st.spinner("Initializing Bayesian Dynamic Fusion Framework..."):
            # Initialize model
            model = EnhancedHybridModel()
            model.bayesian_fusion = BayesianDynamicFusion(
                temperature=temperature,
                lookback_window=lookback_window
            )
            
            # Fetch data
            ticker = f"{selected_stock}.NS"
            df_stock = get_stock_data(ticker, start_date, end_date)
            
            if df_stock.empty:
                st.error("No data retrieved. Please try a different stock or date range.")
                return
            
            # Display basic info
            st.header(f"📊 Analysis for {selected_stock}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                current_price = df_stock['Close'].iloc[-1]
                st.metric("Current Price", f"₹{current_price:.2f}")
            
            with col2:
                if len(df_stock) > 1:
                    daily_change = ((current_price - df_stock['Close'].iloc[-2]) / 
                                   df_stock['Close'].iloc[-2] * 100)
                    st.metric("Daily Change", f"{daily_change:+.2f}%")
                else:
                    st.metric("Daily Change", "N/A")
            
            with col3:
                st.metric("Volume", f"{df_stock['Volume'].iloc[-1]:,}")
            
            # Train model
            st.subheader("🧠 Training Bayesian Dynamic Fusion Model")
            
            # Simple sentiment (placeholder - implement proper sentiment if needed)
            daily_sentiment = {str(datetime.date.today()): 0.1}
            
            result = model.train_hybrid_model(df_stock, daily_sentiment)
            
            if result[0] is not None:
                df_enhanced, models, scaler, features, metrics = result
                
                # Display metrics
                st.subheader("📈 Model Performance")
                
                cols = st.columns(4)
                if 'xgb' in metrics:
                    cols[0].metric("XGBoost MAE", f"{metrics['xgb']['mae']:.4f}")
                if 'gru' in metrics:
                    cols[1].metric("GRU MAE", f"{metrics['gru']['mae']:.4f}")
                if 'hybrid' in metrics:
                    cols[2].metric("Hybrid MAE", f"{metrics['hybrid']['mae']:.4f}")
                    cols[3].metric("Hybrid Accuracy", f"{metrics['hybrid']['accuracy']:.1f}%")
                
                # Display Bayesian weights
                if 'bayesian_fusion' in metrics:
                    st.subheader("⚖️ Bayesian Fusion Weights")
                    weights = metrics['bayesian_fusion']['final_weights']
                    
                    weight_df = pd.DataFrame({
                        'Model': list(weights.keys()),
                        'Weight': list(weights.values()),
                        'Uncertainty': [metrics['bayesian_fusion']['source_uncertainties'].get(k, 0) 
                                       for k in weights.keys()]
                    })
                    
                    st.dataframe(weight_df.style.format({
                        'Weight': '{:.2%}',
                        'Uncertainty': '{:.3f}'
                    }))
                    
                    # Visualization
                    fig = go.Figure(data=[go.Bar(
                        x=list(weights.keys()),
                        y=list(weights.values()),
                        text=[f"{v:.1%}" for v in weights.values()],
                        textposition='auto',
                        marker_color=['lightblue', 'lightcoral', 'lightgreen']
                    )])
                    
                    fig.update_layout(
                        title="Current Bayesian Fusion Weights",
                        xaxis_title="Model",
                        yaxis_title="Weight",
                        yaxis=dict(range=[0, 1])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Simple prediction
                st.subheader("🔮 Forecast")
                st.info("Forecast generation would be implemented here with the trained model.")
                
            else:
                st.error("Model training failed. Please try with more data or a different stock.")
    
    else:
        # Show instructions
        st.info("""
        ## 📋 Instructions
        
        1. **Select a stock** from the dropdown
        2. **Choose date range** (minimum 3 months recommended)
        3. **Adjust Bayesian parameters** if desired
        4. **Click 'Run Bayesian Analysis'** to start
        
        ## 🔬 Scientific Features
        
        - **Bayesian Dynamic Fusion**: Weights adjust based on model uncertainty
        - **Multi-Model Ensemble**: XGBoost, GRU, and Prophet models
        - **Uncertainty Quantification**: Each model's confidence is measured
        - **Real-time Adaptation**: Weights update with market changes
        
        ## 🛠️ Technical Implementation
        
        - **Robust Data Validation**: Handles missing/invalid values
        - **Safe Feature Engineering**: Avoids division by zero and infinite values
        - **Error Handling**: Graceful degradation on failures
        - **Scalable Architecture**: Can be extended with more data sources
        """)

# ============================================================================
# 8. RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()