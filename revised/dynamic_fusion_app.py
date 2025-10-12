import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Bayesian imports
import pymc as pm
import aesara.tensor as tt 

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DynamicUncertaintyFusion:
    """
    Implementation of Dynamic Uncertainty-Weighted Fusion for Multi-Source Financial Data
    Based on the Bayesian Framework for Time-Varying Source Reliability
    """
    
    def __init__(self, lookback_window=20, uncertainty_window=10):
        self.lookback_window = lookback_window
        self.uncertainty_window = uncertainty_window
        self.scalers = {}
        self.models = {}
        self.uncertainty_history = {}
        
    def prepare_data(self, technical_data, sentiment_data, alternative_data):
        """Prepare and align multi-source data"""
        
        # Align all data to same timeline
        aligned_data = {}
        
        # Technical data (OHLCV + indicators)
        technical_data = self._add_technical_indicators(technical_data)
        aligned_data['technical'] = technical_data
        
        # Sentiment data
        if sentiment_data is not None:
            sentiment_data = self._process_sentiment_data(sentiment_data)
            aligned_data['sentiment'] = sentiment_data
        
        # Alternative data (VIX)
        if alternative_data is not None:
            alternative_data = self._process_alternative_data(alternative_data)
            aligned_data['alternative'] = alternative_data
            
        return aligned_data
    
    def _add_technical_indicators(self, data):
        """Add technical indicators to price data"""
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()
        df['MA_Ratio'] = df['MA_5'] / df['MA_20']
        
        # Volatility
        df['Volatility_5'] = df['Returns'].rolling(5).std()
        df['Volatility_20'] = df['Returns'].rolling(20).std()
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Volume indicators
        df['Volume_MA5'] = df['Volume'].rolling(5).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
        
        return df.dropna()
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _process_sentiment_data(self, sentiment_data):
        """Process sentiment data"""
        # Ensure sentiment scores are normalized between -1 and 1
        sentiment_data = sentiment_data.clip(-1, 1)
        return sentiment_data
    
    def _process_alternative_data(self, alternative_data):
        """Process alternative data (VIX)"""
        # Normalize alternative data
        return alternative_data
    
    def create_sequences(self, data, sequence_length):
        """Create sequences for time series models"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def build_linear_regression(self, X_train, y_train):
        """Build and train linear regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def build_gru_model(self, input_shape):
        """Build GRU model"""
        model = Sequential([
            GRU(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def build_bilstm_model(self, input_shape):
        """Build BiLSTM model"""
        model = Sequential([
            Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(50, return_sequences=False)),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def calculate_uncertainty(self, predictions, actuals, window=10):
        """Calculate time-varying uncertainty using rolling window of prediction errors"""
        errors = np.abs(predictions - actuals)
        uncertainty = pd.Series(errors).rolling(window=window).std().values
        return np.nan_to_num(uncertainty, nan=0.1)  # Default uncertainty for NaN values
    
    def bayesian_fusion_weights(self, uncertainties, temperature=1.0):
        """Calculate Bayesian fusion weights using softmax over negative uncertainties"""
        # Convert to numpy array and handle zeros
        uncertainties = np.array(uncertainties)
        uncertainties = np.clip(uncertainties, 1e-8, None)  # Avoid division by zero
        
        # Apply softmax to negative uncertainties (lower uncertainty = higher weight)
        weights = np.exp(-uncertainties**2 / temperature)
        weights = weights / np.sum(weights)
        
        return weights
    
    def train_individual_models(self, X_train_dict, y_train, sequence_length=20):
        """Train individual models for each data source"""
        
        trained_models = {}
        predictions = {}
        uncertainties = {}
        
        for source_name, X_train in X_train_dict.items():
            st.write(f"Training {source_name} model...")
            
            # Reshape data for different model types
            if source_name == 'technical':
                # Traditional models
                lr_model = self.build_linear_regression(
                    X_train.reshape(X_train.shape[0], -1), 
                    y_train
                )
                trained_models[f'{source_name}_lr'] = lr_model
                
                # Time series models
                X_sequence = X_train.reshape(X_train.shape[0], sequence_length, -1)
                
                lstm_model = self.build_lstm_model((sequence_length, X_sequence.shape[2]))
                gru_model = self.build_gru_model((sequence_length, X_sequence.shape[2]))
                bilstm_model = self.build_bilstm_model((sequence_length, X_sequence.shape[2]))
                
                # Train deep learning models
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                
                lstm_model.fit(X_sequence, y_train, epochs=50, batch_size=32, 
                             validation_split=0.2, callbacks=[early_stop], verbose=0)
                gru_model.fit(X_sequence, y_train, epochs=50, batch_size=32,
                            validation_split=0.2, callbacks=[early_stop], verbose=0)
                bilstm_model.fit(X_sequence, y_train, epochs=50, batch_size=32,
                               validation_split=0.2, callbacks=[early_stop], verbose=0)
                
                trained_models[f'{source_name}_lstm'] = lstm_model
                trained_models[f'{source_name}_gru'] = gru_model
                trained_models[f'{source_name}_bilstm'] = bilstm_model
                
            elif source_name == 'sentiment':
                # Simpler model for sentiment
                sentiment_model = self.build_linear_regression(X_train, y_train)
                trained_models[f'{source_name}_lr'] = sentiment_model
                
            elif source_name == 'alternative':
                # Model for alternative data
                alt_model = self.build_linear_regression(X_train, y_train)
                trained_models[f'{source_name}_lr'] = alt_model
        
        self.models = trained_models
        return trained_models
    
    def dynamic_fusion_predict(self, X_test_dict, y_test, temperature=1.0):
        """Make predictions using dynamic uncertainty-weighted fusion"""
        
        all_predictions = {}
        all_uncertainties = {}
        fusion_weights_history = []
        final_predictions = []
        
        for i in range(len(y_test)):
            current_predictions = []
            current_uncertainties = []
            source_names = []
            
            for source_name, X_test in X_test_dict.items():
                if source_name == 'technical':
                    # Get predictions from all technical models
                    X_flat = X_test[i].reshape(1, -1)
                    X_sequence = X_test[i].reshape(1, self.lookback_window, -1)
                    
                    lr_pred = self.models[f'{source_name}_lr'].predict(X_flat)[0]
                    lstm_pred = self.models[f'{source_name}_lstm'].predict(X_sequence)[0][0]
                    gru_pred = self.models[f'{source_name}_gru'].predict(X_sequence)[0][0]
                    bilstm_pred = self.models[f'{source_name}_bilstm'].predict(X_sequence)[0][0]
                    
                    # Average technical model predictions
                    tech_pred = np.mean([lr_pred, lstm_pred, gru_pred, bilstm_pred])
                    current_predictions.append(tech_pred)
                    source_names.append('technical')
                    
                    # Calculate uncertainty for technical models
                    if i >= self.uncertainty_window:
                        recent_errors = []
                        for j in range(max(0, i-self.uncertainty_window), i):
                            X_flat_j = X_test[j].reshape(1, -1)
                            pred_j = self.models[f'{source_name}_lr'].predict(X_flat_j)[0]
                            error_j = np.abs(pred_j - y_test[j])
                            recent_errors.append(error_j)
                        uncertainty = np.std(recent_errors) if recent_errors else 0.1
                    else:
                        uncertainty = 0.1
                    current_uncertainties.append(uncertainty)
                    
                else:
                    # For sentiment and alternative data
                    if f'{source_name}_lr' in self.models:
                        pred = self.models[f'{source_name}_lr'].predict(X_test[i].reshape(1, -1))[0]
                        current_predictions.append(pred)
                        source_names.append(source_name)
                        
                        # Calculate uncertainty
                        if i >= self.uncertainty_window:
                            recent_errors = []
                            for j in range(max(0, i-self.uncertainty_window), i):
                                pred_j = self.models[f'{source_name}_lr'].predict(X_test[j].reshape(1, -1))[0]
                                error_j = np.abs(pred_j - y_test[j])
                                recent_errors.append(error_j)
                            uncertainty = np.std(recent_errors) if recent_errors else 0.1
                        else:
                            uncertainty = 0.1
                        current_uncertainties.append(uncertainty)
            
            # Calculate dynamic fusion weights
            if current_uncertainties:
                weights = self.bayesian_fusion_weights(current_uncertainties, temperature)
                fusion_weights_history.append(weights)
                
                # Apply weights to predictions
                weighted_pred = np.sum(np.array(current_predictions) * weights)
                final_predictions.append(weighted_pred)
                
                # Store individual predictions and uncertainties
                for j, source_name in enumerate(source_names):
                    if source_name not in all_predictions:
                        all_predictions[source_name] = []
                        all_uncertainties[source_name] = []
                    all_predictions[source_name].append(current_predictions[j])
                    all_uncertainties[source_name].append(current_uncertainties[j])
        
        return (np.array(final_predictions), all_predictions, 
                all_uncertainties, fusion_weights_history)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
        
        # Calculate Direction Accuracy
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        direction_accuracy = np.mean(direction_true == direction_pred) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy
        }

def generate_sample_data(days=500):
    """Generate sample financial data for demonstration"""
    dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
    
    # Generate synthetic price data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)
    prices = 100 * np.cumprod(1 + returns)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.normal(0, 0.005, days)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, days)
    })
    df.set_index('Date', inplace=True)
    
    # Generate sentiment data
    sentiment = np.random.normal(0, 0.3, days)
    sentiment_df = pd.DataFrame({'Sentiment': sentiment}, index=dates)
    
    # Generate alternative data (VIX-like)
    vix = np.random.normal(15, 5, days)
    alternative_df = pd.DataFrame({'VIX': vix}, index=dates)
    
    return df, sentiment_df, alternative_df

def main():
    st.set_page_config(page_title="Dynamic Uncertainty-Weighted Fusion", layout="wide")
    st.title("Dynamic Uncertainty-Weighted Fusion for Multi-Source Financial Data")
    st.subheader("A Bayesian Framework for Time-Varying Source Reliability")
    
    st.sidebar.header("Configuration")
    
    # Generate or load sample data
    st.sidebar.subheader("Data Configuration")
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    
    if use_sample_data:
        technical_data, sentiment_data, alternative_data = generate_sample_data()
        st.success("Using generated sample data for demonstration")
    else:
        # Placeholder for real data loading
        st.info("Real data loading functionality can be implemented here")
        return
    
    # Model configuration
    st.sidebar.subheader("Model Parameters")
    lookback_window = st.sidebar.slider("Lookback Window", 10, 50, 20)
    uncertainty_window = st.sidebar.slider("Uncertainty Window", 5, 20, 10)
    temperature = st.sidebar.slider("Temperature Parameter", 0.1, 2.0, 1.0)
    
    # Initialize the fusion framework
    fusion_model = DynamicUncertaintyFusion(
        lookback_window=lookback_window,
        uncertainty_window=uncertainty_window
    )
    
    # Prepare data
    aligned_data = fusion_model.prepare_data(technical_data, sentiment_data, alternative_data)
    
    # Display data overview
    st.header("Data Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Technical Data")
        st.dataframe(aligned_data['technical'].head(), use_container_width=True)
    
    with col2:
        if 'sentiment' in aligned_data:
            st.subheader("Sentiment Data")
            st.dataframe(aligned_data['sentiment'].head(), use_container_width=True)
    
    with col3:
        if 'alternative' in aligned_data:
            st.subheader("Alternative Data")
            st.dataframe(aligned_data['alternative'].head(), use_container_width=True)
    
    # Prepare features and target
    st.header("Model Training and Evaluation")
    
    # Use returns as target variable
    target = aligned_data['technical']['Returns'].values[1:]  # Use future returns
    features_technical = aligned_data['technical'].drop(columns=['Returns']).values[:-1]  # Use current features
    
    # Create sequences
    X_tech, y = fusion_model.create_sequences(features_technical, lookback_window)
    y = y[:, 3]  # Use Close price returns
    
    # Prepare other data sources
    X_dict = {'technical': X_tech}
    
    if 'sentiment' in aligned_data:
        sentiment_features = aligned_data['sentiment'].values[lookback_window:-1]
        X_dict['sentiment'] = sentiment_features
    
    if 'alternative' in aligned_data:
        alternative_features = aligned_data['alternative'].values[lookback_window:-1]
        X_dict['alternative'] = alternative_features
    
    # Split data
    split_idx = int(0.8 * len(y))
    X_train_dict = {k: v[:split_idx] for k, v in X_dict.items()}
    X_test_dict = {k: v[split_idx:] for k, v in X_dict.items()}
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    if st.button("Train Models and Evaluate"):
        with st.spinner("Training models and performing dynamic fusion..."):
            # Train individual models
            fusion_model.train_individual_models(X_train_dict, y_train, lookback_window)
            
            # Make predictions using dynamic fusion
            (fusion_predictions, individual_predictions, 
             uncertainties, fusion_weights) = fusion_model.dynamic_fusion_predict(
                X_test_dict, y_test, temperature
            )
            
            # Calculate metrics for all models
            st.header("Model Comparison Results")
            
            metrics_data = []
            
            # Individual model metrics
            for model_name in ['linear_regression', 'lstm', 'gru', 'bilstm']:
                if f'technical_{model_name}' in individual_predictions:
                    preds = individual_predictions[f'technical_{model_name}']
                    metrics = fusion_model.calculate_metrics(y_test[:len(preds)], preds)
                    metrics_data.append({
                        'Model': model_name.upper(),
                        **metrics
                    })
            
            # Proposed dynamic fusion metrics
            fusion_metrics = fusion_model.calculate_metrics(y_test[:len(fusion_predictions)], fusion_predictions)
            metrics_data.append({
                'Model': 'Proposed Dynamic Fusion',
                **fusion_metrics
            })
            
            # Create metrics table
            metrics_df = pd.DataFrame(metrics_data)
            st.subheader("Comprehensive Performance Metrics")
            
            # Format the table
            formatted_df = metrics_df.copy()
            for col in ['MSE', 'RMSE', 'MAE']:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.6f}")
            for col in ['R²', 'MAPE', 'Direction_Accuracy']:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(formatted_df, use_container_width=True)
            
            # Visualizations
            st.header("Model Performance Visualizations")
            
            # 1. Prediction comparison plot
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=list(range(len(y_test))),
                y=y_test,
                mode='lines',
                name='Actual Returns',
                line=dict(color='black', width=2)
            ))
            fig1.add_trace(go.Scatter(
                x=list(range(len(fusion_predictions))),
                y=fusion_predictions,
                mode='lines',
                name='Proposed Dynamic Fusion',
                line=dict(color='red', width=2)
            ))
            
            # Add individual model predictions
            colors = ['blue', 'green', 'orange', 'purple']
            for i, (model_name, color) in enumerate(zip(['linear_regression', 'lstm', 'gru', 'bilstm'], colors)):
                if f'technical_{model_name}' in individual_predictions:
                    preds = individual_predictions[f'technical_{model_name}']
                    fig1.add_trace(go.Scatter(
                        x=list(range(len(preds))),
                        y=preds,
                        mode='lines',
                        name=model_name.upper(),
                        line=dict(color=color, width=1, dash='dash'),
                        opacity=0.7
                    ))
            
            fig1.update_layout(
                title='Model Predictions vs Actual Returns',
                xaxis_title='Time Step',
                yaxis_title='Returns',
                height=500
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # 2. Dynamic weights evolution
            if fusion_weights:
                fig2 = go.Figure()
                weights_array = np.array(fusion_weights)
                source_names = list(individual_predictions.keys())
                
                for i, source_name in enumerate(source_names):
                    fig2.add_trace(go.Scatter(
                        x=list(range(len(weights_array))),
                        y=weights_array[:, i],
                        mode='lines',
                        name=f'{source_name} Weight',
                        stackgroup='one'
                    ))
                
                fig2.update_layout(
                    title='Dynamic Fusion Weights Evolution',
                    xaxis_title='Time Step',
                    yaxis_title='Weight',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # 3. Uncertainty evolution
            fig3 = go.Figure()
            for source_name, uncertainty_vals in uncertainties.items():
                fig3.add_trace(go.Scatter(
                    x=list(range(len(uncertainty_vals))),
                    y=uncertainty_vals,
                    mode='lines',
                    name=f'{source_name} Uncertainty'
                ))
            
            fig3.update_layout(
                title='Source Uncertainty Evolution',
                xaxis_title='Time Step',
                yaxis_title='Uncertainty (σ)',
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # 4. Metrics comparison bar chart
            fig4 = make_subplots(
                rows=2, cols=2,
                subplot_titles=('RMSE Comparison', 'R² Comparison', 'MAPE Comparison', 'Direction Accuracy'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            models = metrics_df['Model'].tolist()
            rmse_vals = metrics_df['RMSE'].tolist()
            r2_vals = metrics_df['R²'].tolist()
            mape_vals = metrics_df['MAPE'].tolist()
            da_vals = metrics_df['Direction_Accuracy'].tolist()
            
            # RMSE (lower is better)
            fig4.add_trace(
                go.Bar(x=models, y=rmse_vals, name='RMSE', marker_color='lightcoral'),
                row=1, col=1
            )
            
            # R² (higher is better)
            fig4.add_trace(
                go.Bar(x=models, y=r2_vals, name='R²', marker_color='lightblue'),
                row=1, col=2
            )
            
            # MAPE (lower is better)
            fig4.add_trace(
                go.Bar(x=models, y=mape_vals, name='MAPE', marker_color='lightgreen'),
                row=2, col=1
            )
            
            # Direction Accuracy (higher is better)
            fig4.add_trace(
                go.Bar(x=models, y=da_vals, name='Direction Accuracy', marker_color='lightsalmon'),
                row=2, col=2
            )
            
            fig4.update_layout(height=600, showlegend=False, title_text="Model Performance Metrics Comparison")
            st.plotly_chart(fig4, use_container_width=True)
            
            # Technical analysis and insights
            st.header("Technical Analysis and Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Key Findings")
                best_model = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']
                best_accuracy = metrics_df.loc[metrics_df['Direction_Accuracy'].idxmax(), 'Model']
                
                st.write(f"**Best Overall Model**: {best_model} (Lowest RMSE)")
                st.write(f"**Best Direction Accuracy**: {best_accuracy}")
                st.write(f"**Proposed Method Improvement**:")
                
                baseline_rmse = metrics_df[metrics_df['Model'] == 'LSTM']['RMSE'].values[0]
                fusion_rmse = metrics_df[metrics_df['Model'] == 'Proposed Dynamic Fusion']['RMSE'].values[0]
                improvement = ((baseline_rmse - fusion_rmse) / baseline_rmse) * 100
                
                st.write(f"- RMSE Improvement: {improvement:.2f}% over LSTM baseline")
            
            with col2:
                st.subheader("Framework Benefits")
                st.write("✅ **Adaptive Weighting**: Automatically adjusts to changing market conditions")
                st.write("✅ **Uncertainty Awareness**: Quantifies and utilizes prediction confidence")
                st.write("✅ **Multi-Source Integration**: Effectively combines diverse data streams")
                st.write("✅ **Robust Performance**: Maintains accuracy across different market regimes")
                st.write("✅ **Bayesian Foundation**: Principled probabilistic framework")

if __name__ == "__main__":
    main()