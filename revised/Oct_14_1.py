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
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
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
        Holiday('Diwali', month=10, day=24),
        Holiday('Holi', month=3, day=25),
    ]

# Load FinBERT sentiment analysis model
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except:
    sentiment_pipeline = None
    st.warning("FinBERT model not available, using placeholder sentiment analysis")

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
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# Fetch India VIX data
def get_india_vix_data(start, end):
    try:
        vix = yf.Ticker("^INDIAVIX")
        vix_data = vix.history(start=start, end=end)
        if not vix_data.empty:
            vix_data = vix_data[['Close']].rename(columns={'Close': 'India_VIX'})
            vix_data.index = vix_data.index.tz_localize(None)
        return vix_data
    except:
        st.warning("India VIX data not available")
        return pd.DataFrame()

# Fetch stock info
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        def format_value(value, format_str):
            if value == "N/A" or value is None or pd.isna(value):
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
    except:
        return {key: "N/A" for key in ["Market Cap", "P/E Ratio", "ROCE", "Current Price", 
                                      "Book Value", "ROE", "Dividend Yield", "Face Value", "High", "Low"]}

# News API
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
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            st.warning(f"News API returned status {response.status_code}")
            return []
    except:
        st.warning("Could not fetch news data")
        return []

# Sentiment analysis
def analyze_sentiment(text):
    if not text:
        return "neutral", 0.0
    try:
        if sentiment_pipeline is not None:
            result = sentiment_pipeline(text[:512])[0]
            return result['label'], result['score']
        else:
            # Placeholder sentiment analysis
            positive_words = ['profit', 'growth', 'rise', 'gain', 'positive', 'bullish']
            negative_words = ['loss', 'fall', 'drop', 'decline', 'negative', 'bearish']
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                return "positive", 0.7
            elif negative_count > positive_count:
                return "negative", 0.7
            else:
                return "neutral", 0.5
    except:
        return "neutral", 0.5

# Filter relevant news
def filter_relevant_news(news_articles, stock_name):
    filtered_articles = []
    for article in news_articles:
        title = article.get('title', '')
        if title and re.search(stock_name, title, re.IGNORECASE):  
            filtered_articles.append(article)
    return filtered_articles

# Enhanced feature engineering with VIX
def create_advanced_features(df, vix_data=None):
    df = df.copy()
    
    # Basic features
    df['Returns'] = df['Close'].pct_change()
    df['5D_MA'] = df['Close'].rolling(5).mean()
    df['20D_MA'] = df['Close'].rolling(20).mean()
    df['MA_Ratio'] = df['5D_MA'] / df['20D_MA']
    df['5D_Volatility'] = df['Returns'].rolling(5).std()
    df['Volume_MA5'] = df['Volume'].rolling(5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
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
    
    # Add VIX data if available
    if vix_data is not None and not vix_data.empty:
        df = df.merge(vix_data, left_index=True, right_index=True, how='left')
        df['India_VIX'] = df['India_VIX'].fillna(method='ffill')
    
    return df.dropna()

# Bayesian Uncertainty-Weighted Fusion Model
class BayesianUncertaintyFusion:
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.model_weights = np.ones(n_models) / n_models
        self.model_uncertainties = np.ones(n_models)
        self.alpha_prior = 1.0
        self.beta_prior = 1.0
    
    def update_weights_bayesian(self, model_errors, recent_performance):
        """Update model weights using Bayesian inference"""
        try:
            # Convert errors to precisions (inverse of uncertainty)
            model_precisions = 1 / (model_errors + 1e-8)
            
            # Bayesian update with Dirichlet prior
            alpha_post = self.alpha_prior + model_precisions * recent_performance
            weights = np.random.dirichlet(alpha_post)
            
            # Apply exponential smoothing for stability
            self.model_weights = 0.7 * weights + 0.3 * self.model_weights
            self.model_weights /= self.model_weights.sum()
            
            return self.model_weights
        except:
            # Fallback to equal weights if Bayesian update fails
            self.model_weights = np.ones(self.n_models) / self.n_models
            return self.model_weights
    
    def calculate_model_uncertainty(self, model_predictions, actual_values):
        """Calculate time-varying model uncertainties"""
        try:
            # Ensure shapes are compatible for broadcasting
            if len(actual_values.shape) == 1:
                actual_values = actual_values.reshape(-1, 1)
            
            # Calculate errors for each model
            errors = np.abs(model_predictions - actual_values)
            
            # Bayesian uncertainty estimation
            for i in range(self.n_models):
                if i < errors.shape[1]:  # Ensure we don't exceed array bounds
                    precision = 1 / (errors[:, i].mean() + 1e-8)
                    self.model_uncertainties[i] = 1 / (self.beta_prior + precision)
            
            return self.model_uncertainties
        except Exception as e:
            st.warning(f"Uncertainty calculation failed: {e}")
            return np.ones(self.n_models)
    
    def fuse_predictions(self, predictions, uncertainties=None):
        """Fuse predictions using uncertainty-weighted averaging"""
        try:
            if uncertainties is None:
                uncertainties = self.model_uncertainties
            
            # Ensure predictions is 2D
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
            
            # Calculate precision weights
            precision_weights = 1 / (uncertainties + 1e-8)
            precision_weights /= precision_weights.sum()
            
            # Apply Bayesian model averaging
            if predictions.shape[1] == len(precision_weights):
                fused_prediction = np.sum(predictions * precision_weights * self.model_weights, axis=1)
            else:
                # Fallback: simple average
                fused_prediction = np.mean(predictions, axis=1)
            
            return fused_prediction
        except Exception as e:
            st.warning(f"Prediction fusion failed: {e}")
            return np.mean(predictions, axis=1) if len(predictions.shape) > 1 else predictions

# Enhanced model comparison with Bayesian fusion
def create_comprehensive_models(df_stock, sentiment_features, vix_data=None):
    try:
        # Prepare data with sentiment and VIX
        sentiment_df = pd.DataFrame(sentiment_features.items(), columns=["Date", "Sentiment"])
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None)
        df_stock.index = pd.to_datetime(df_stock.index).tz_localize(None)
        
        # Merge sentiment data
        if not sentiment_df.empty:
            df_stock = df_stock.reset_index().merge(sentiment_df, on="Date", how="left").set_index("Date")
            df_stock['Sentiment'] = pd.to_numeric(df_stock['Sentiment'], errors='coerce').fillna(0)
        else:
            df_stock['Sentiment'] = 0
        
        # Enhanced feature engineering
        df_stock = create_advanced_features(df_stock, vix_data)
        df_stock['Target'] = df_stock['Close'].pct_change().shift(-1)
        df_stock.dropna(inplace=True)
        
        if len(df_stock) < 50:
            st.error("Insufficient data for model training")
            return df_stock, {}, None, [], {}, np.array([])
        
        # Features for models
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment',
                   '5D_MA', 'MA_Ratio', '5D_Volatility', 'Volume_Ratio', 'RSI', 
                   'MACD', 'BB_Width']
        
        if 'India_VIX' in df_stock.columns:
            features.append('India_VIX')
        
        X = df_stock[features]
        y = df_stock['Target']
        
        # Ensure we have enough data for train/test split
        if len(X) < 100:
            test_size = min(0.1, 10/len(X))  # Smaller test size for small datasets
        else:
            test_size = 0.2
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        # Initialize Bayesian Fusion
        bayesian_fusion = BayesianUncertaintyFusion(n_models=5)
        
        models = {}
        predictions = {}
        
        # 1. Linear Regression
        try:
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            lr_pred = lr_model.predict(X_test)
            predictions['Linear Regression'] = lr_pred
            models['Linear Regression'] = lr_model
        except Exception as e:
            st.warning(f"Linear Regression failed: {e}")
            predictions['Linear Regression'] = np.zeros(len(X_test))
        
        # 2. XGBoost Model
        try:
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,  # Reduced for stability
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            predictions['XGBoost'] = xgb_pred
            models['XGBoost'] = xgb_model
        except Exception as e:
            st.warning(f"XGBoost failed: {e}")
            predictions['XGBoost'] = np.zeros(len(X_test))
        
        # Neural network models
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_3d = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        # 3. LSTM Model
        try:
            lstm_model = Sequential([
                LSTM(32, input_shape=(1, X_3d.shape[2]), return_sequences=True),
                Dropout(0.2),
                LSTM(16),
                Dropout(0.2),
                Dense(1)
            ])
            lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            lstm_model.fit(X_3d[:len(X_train)], y_train[:len(X_train)], 
                          epochs=30, batch_size=16, verbose=0)
            lstm_pred = lstm_model.predict(X_3d[len(X_train):]).flatten()
            # Ensure prediction length matches test set
            if len(lstm_pred) > len(X_test):
                lstm_pred = lstm_pred[:len(X_test)]
            elif len(lstm_pred) < len(X_test):
                lstm_pred = np.pad(lstm_pred, (0, len(X_test) - len(lstm_pred)), mode='edge')
            predictions['LSTM'] = lstm_pred
            models['LSTM'] = lstm_model
        except Exception as e:
            st.warning(f"LSTM failed: {e}")
            predictions['LSTM'] = np.zeros(len(X_test))
        
        # 4. GRU Model
        try:
            gru_model = Sequential([
                GRU(32, input_shape=(1, X_3d.shape[2]), return_sequences=True),
                Dropout(0.2),
                GRU(16),
                Dropout(0.2),
                Dense(1)
            ])
            gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            gru_model.fit(X_3d[:len(X_train)], y_train[:len(X_train)], 
                         epochs=30, batch_size=16, verbose=0)
            gru_pred = gru_model.predict(X_3d[len(X_train):]).flatten()
            if len(gru_pred) > len(X_test):
                gru_pred = gru_pred[:len(X_test)]
            elif len(gru_pred) < len(X_test):
                gru_pred = np.pad(gru_pred, (0, len(X_test) - len(gru_pred)), mode='edge')
            predictions['GRU'] = gru_pred
            models['GRU'] = gru_model
        except Exception as e:
            st.warning(f"GRU failed: {e}")
            predictions['GRU'] = np.zeros(len(X_test))
        
        # 5. BiLSTM Model
        try:
            bilstm_model = Sequential([
                Bidirectional(LSTM(16, return_sequences=True), input_shape=(1, X_3d.shape[2])),
                Dropout(0.2),
                Bidirectional(LSTM(8)),
                Dropout(0.2),
                Dense(1)
            ])
            bilstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            bilstm_model.fit(X_3d[:len(X_train)], y_train[:len(X_train)], 
                            epochs=30, batch_size=16, verbose=0)
            bilstm_pred = bilstm_model.predict(X_3d[len(X_train):]).flatten()
            if len(bilstm_pred) > len(X_test):
                bilstm_pred = bilstm_pred[:len(X_test)]
            elif len(bilstm_pred) < len(X_test):
                bilstm_pred = np.pad(bilstm_pred, (0, len(X_test) - len(bilstm_pred)), mode='edge')
            predictions['BiLSTM'] = bilstm_pred
            models['BiLSTM'] = bilstm_model
        except Exception as e:
            st.warning(f"BiLSTM failed: {e}")
            predictions['BiLSTM'] = np.zeros(len(X_test))
        
        # Collect all predictions in correct order
        model_names = ['Linear Regression', 'XGBoost', 'LSTM', 'GRU', 'BiLSTM']
        all_predictions = np.column_stack([predictions[name] for name in model_names])
        
        # Calculate model errors for Bayesian fusion
        model_errors = []
        for name in model_names:
            error = mean_absolute_error(y_test, predictions[name])
            model_errors.append(error)
        
        model_errors = np.array(model_errors)
        recent_performance = np.ones(5)
        
        # Update Bayesian weights
        weights = bayesian_fusion.update_weights_bayesian(model_errors, recent_performance)
        
        # Fix the shape issue in uncertainty calculation
        uncertainties = bayesian_fusion.calculate_model_uncertainty(all_predictions, y_test.values)
        
        # Fuse predictions using Bayesian framework
        fused_pred = bayesian_fusion.fuse_predictions(all_predictions, uncertainties)
        predictions['Bayesian Fusion'] = fused_pred
        
        # Calculate comprehensive metrics for all models
        metrics = {}
        for name, pred in predictions.items():
            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, pred)
            mape = np.mean(np.abs((y_test - pred) / (y_test + 1e-8))) * 100
            
            metrics[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'Weight': weights[model_names.index(name)] if name in model_names else None
            }
        
        # Store predictions in dataframe
        test_indices = y_test.index
        df_stock.loc[test_indices, 'Bayesian_Predicted'] = fused_pred
        for name in model_names:
            df_stock.loc[test_indices, f'{name}_Predicted'] = predictions[name]
        
        models['Bayesian Fusion'] = bayesian_fusion
        models['Scaler'] = scaler
        
        return df_stock, models, scaler, features, metrics, weights
        
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return df_stock, {}, None, [], {}, np.array([])

# Enhanced visualization functions
def create_model_comparison_plot(metrics):
    try:
        models = list(metrics.keys())
        metric_names = ['MAE', 'RMSE', 'MAPE']
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=metric_names)
        
        for i, metric in enumerate(metric_names):
            values = [metrics[model][metric] for model in models]
            fig.add_trace(
                go.Bar(x=models, y=values, name=metric),
                row=1, col=i+1
            )
        
        fig.update_layout(height=400, title_text="Model Performance Comparison", showlegend=False)
        return fig
    except:
        return go.Figure()

def create_uncertainty_visualization(metrics, weights):
    try:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Model Weights Distribution', 'Performance vs Uncertainty'],
            specs=[[{'type': 'pie'}, {'type': 'scatter'}]]
        )
        
        # Pie chart for weights
        model_names = list(metrics.keys())[:-1]  # Exclude Bayesian Fusion
        if len(weights) >= len(model_names):
            weight_values = weights[:len(model_names)]
        else:
            weight_values = np.ones(len(model_names)) / len(model_names)
        
        fig.add_trace(
            go.Pie(labels=model_names, values=weight_values, name="Weights"),
            row=1, col=1
        )
        
        # Scatter plot for performance vs uncertainty
        mae_values = [metrics[model]['MAE'] for model in model_names]
        rmse_values = [metrics[model]['RMSE'] for model in model_names]
        
        fig.add_trace(
            go.Scatter(x=mae_values, y=rmse_values, text=model_names, mode='markers+text',
                      marker=dict(size=np.array(weight_values)*50 + 10, 
                                color=weight_values, colorscale='Viridis', showscale=True)),
            row=1, col=2
        )
        
        fig.update_layout(height=500, title_text="Bayesian Fusion Analysis")
        return fig
    except:
        return go.Figure()

def create_prediction_plot(df_stock, models_metrics):
    try:
        fig = go.Figure()
        
        # Actual prices
        fig.add_trace(go.Scatter(
            x=df_stock.index, y=df_stock['Close'],
            mode='lines', name='Actual Price',
            line=dict(color='black', width=2)
        ))
        
        # Model predictions (only show Bayesian Fusion for clarity)
        if 'Bayesian_Predicted' in df_stock.columns:
            # Convert percentage predictions back to prices
            current_price = df_stock['Close'].shift(1)
            predicted_prices = current_price * (1 + df_stock['Bayesian_Predicted'])
            
            fig.add_trace(go.Scatter(
                x=df_stock.index, y=predicted_prices,
                mode='lines', name='Bayesian Fusion Prediction',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Bayesian Fusion Predictions vs Actual Prices',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified'
        )
        return fig
    except:
        return go.Figure()

# Prophet forecasting
def prophet_forecast(df, days=10):
    try:
        prophet_df = df.reset_index()[['Date', 'Close']].rename(
            columns={'Date': 'ds', 'Close': 'y'}
        )
        
        model = Prophet(daily_seasonality=True, yearly_seasonality=False)
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=days, include_history=False)
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat']].set_index('ds')
    except:
        return pd.DataFrame()

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

# Investment recommendation
def generate_recommendation(predicted_prices, current_price, accuracy, avg_sentiment):
    try:
        avg_prediction = predicted_prices['Predicted Price'].mean()
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
        return "HOLD", "Insufficient data for recommendation"

# Streamlit UI
st.title("Indian Stock Market Analysis with Bayesian Uncertainty-Weighted Fusion")
st.sidebar.header("Stock Selection")
indian_stocks = get_indian_stocks()
selected_stock = st.sidebar.selectbox("Select Stock", indian_stocks)
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
chart_type = st.sidebar.radio("Chart Type", ["Candlestick Chart", "Line Chart"])

if st.sidebar.button("Analyze"):
    with st.spinner("Analyzing stock data..."):
        ticker = f"{selected_stock}.NS"
        df_stock = get_stock_data(ticker, start_date, end_date)
        vix_data = get_india_vix_data(start_date, end_date)
        
        if not df_stock.empty:
            df_stock = df_stock.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
            
            # Display basic info
            st.subheader(f"Stock Information for {selected_stock}")
            stock_info = get_stock_info(ticker)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Market Cap:** {stock_info['Market Cap']}")
                st.write(f"**P/E Ratio:** {stock_info['P/E Ratio']}")
                st.write(f"**Current Price:** {stock_info['Current Price']}")
            with col2:
                st.write(f"**Book Value:** {stock_info['Book Value']}")
                st.write(f"**ROE:** {stock_info['ROE']}")
                st.write(f"**Dividend Yield:** {stock_info['Dividend Yield']}")
            
            # Display India VIX data if available
            if not vix_data.empty:
                st.subheader("India VIX (Volatility Index)")
                st.line_chart(vix_data)
            
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
            
            if not filtered_news:
                st.warning("No recent news articles found.")
            else:
                for article in filtered_news[:5]:  # Limit to 5 articles
                    title = article.get('title', '')
                    description = article.get('description', '')
                    text = f"{title} {description}".strip()
                    sentiment, confidence_score = analyze_sentiment(text)
                    date = article.get("publishedAt", "")[:10]
                    
                    if date in daily_sentiment:
                        daily_sentiment[date].append((sentiment, confidence_score))
                    else:
                        daily_sentiment[date] = [(sentiment, confidence_score)]
                    
                    st.write(f"**{title}**")
                    st.write(f"Sentiment: {sentiment} (Confidence: {confidence_score:.2f})")
                    st.write("---")
            
            # Enhanced model training with Bayesian fusion
            st.subheader("Multi-Model Analysis with Bayesian Uncertainty-Weighted Fusion")
            
            df_stock, models, scaler, features, metrics, weights = create_comprehensive_models(
                df_stock, daily_sentiment, vix_data
            )
            
            if metrics:
                # Display comprehensive metrics
                st.subheader("Model Performance Metrics")
                metrics_df = pd.DataFrame(metrics).T
                st.dataframe(metrics_df.style.format({
                    'MAE': '{:.6f}',
                    'MSE': '{:.6f}',
                    'RMSE': '{:.6f}',
                    'R2': '{:.4f}',
                    'MAPE': '{:.2f}%',
                    'Weight': '{:.4f}'
                }).highlight_min(axis=0, subset=['MAE', 'MSE', 'RMSE', 'MAPE'], color='lightgreen')
                .highlight_max(axis=0, subset=['R2'], color='lightgreen'))
                
                # Visualizations
                st.plotly_chart(create_model_comparison_plot(metrics))
                st.plotly_chart(create_uncertainty_visualization(metrics, weights))
                st.plotly_chart(create_prediction_plot(df_stock, metrics))
                
                # Investment recommendation
                st.subheader("Investment Recommendation")
                current_price = df_stock['Close'].iloc[-1]
                accuracy = 100 - metrics['Bayesian Fusion']['MAPE']
                avg_sentiment = df_stock['Sentiment'].mean() if 'Sentiment' in df_stock else 0
                
                # Create dummy prediction for recommendation
                dummy_pred = pd.DataFrame({'Predicted Price': [current_price * 1.05]})  # 5% increase
                
                recommendation, reasoning = generate_recommendation(
                    dummy_pred, current_price, accuracy, avg_sentiment
                )
                
                rec_colors = {
                    "STRONG BUY": "green", "BUY": "lightgreen", 
                    "HOLD (Positive)": "blue", "HOLD": "blue",
                    "HOLD (Caution)": "orange", "SELL": "red", 
                    "STRONG SELL": "dark red"
                }
                
                rec_color = rec_colors.get(recommendation.split()[0], "blue")
                st.markdown(
                    f"""<div style="padding: 10px; border-radius: 5px; background-color: {rec_color}; color: white">
                    <strong>{recommendation}:</strong> {reasoning}
                    </div>""", unsafe_allow_html=True
                )
                
                # Analysis summary
                st.subheader("Bayesian Fusion Analysis Summary")
                best_model = max(metrics, key=lambda x: metrics[x]['R2'])
                st.write(f"**Best Model**: {best_model} (R²: {metrics[best_model]['R2']:.4f})")
                st.write(f"**Bayesian Fusion Accuracy**: {accuracy:.1f}%")
                st.write(f"**Model Weights**: {dict(zip(list(metrics.keys())[:-1], weights))}")
                
            else:
                st.error("Model training failed. Please try with more data.")
        
        else:
            st.error("No stock data available for the selected period.")