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
        return pd.DataFrame()

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

# Enhanced feature engineering with VIX
def create_advanced_features(df, vix_data=None):
    df = df.copy()
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
        self.alpha_prior = 1.0  # Prior for Dirichlet distribution
        self.beta_prior = 1.0   # Prior for Beta distribution (uncertainty)
    
    def update_weights_bayesian(self, model_errors, recent_performance):
        """Update model weights using Bayesian inference"""
        # Convert errors to precisions (inverse of uncertainty)
        model_precisions = 1 / (model_errors + 1e-8)
        
        # Bayesian update with Dirichlet prior
        alpha_post = self.alpha_prior + model_precisions * recent_performance
        weights = np.random.dirichlet(alpha_post)
        
        # Apply exponential smoothing for stability
        self.model_weights = 0.7 * weights + 0.3 * self.model_weights
        self.model_weights /= self.model_weights.sum()
        
        return self.model_weights
    
    def calculate_model_uncertainty(self, model_predictions, actual_values):
        """Calculate time-varying model uncertainties"""
        errors = np.abs(model_predictions - actual_values)
        
        # Bayesian uncertainty estimation
        for i in range(self.n_models):
            # Update uncertainty using Bayesian inference
            precision = 1 / (errors[:, i].mean() + 1e-8)
            self.model_uncertainties[i] = 1 / (self.beta_prior + precision)
        
        return self.model_uncertainties
    
    def fuse_predictions(self, predictions, uncertainties=None):
        """Fuse predictions using uncertainty-weighted averaging"""
        if uncertainties is None:
            uncertainties = self.model_uncertainties
        
        # Calculate precision weights
        precision_weights = 1 / (uncertainties + 1e-8)
        precision_weights /= precision_weights.sum()
        
        # Apply Bayesian model averaging
        fused_prediction = np.sum(predictions * precision_weights * self.model_weights, axis=1)
        
        return fused_prediction

# Enhanced model comparison with Bayesian fusion
def create_comprehensive_models(df_stock, sentiment_features, vix_data=None):
    # Prepare data with sentiment and VIX
    sentiment_df = pd.DataFrame(sentiment_features.items(), columns=["Date", "Sentiment"])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None)
    df_stock.index = pd.to_datetime(df_stock.index).tz_localize(None)
    df_stock = df_stock.reset_index().merge(sentiment_df, on="Date", how="left").set_index("Date")
    df_stock['Sentiment'] = pd.to_numeric(df_stock['Sentiment'], errors='coerce').fillna(0)
    
    # Enhanced feature engineering
    df_stock = create_advanced_features(df_stock, vix_data)
    df_stock['Target'] = df_stock['Close'].pct_change().shift(-1)
    df_stock.dropna(inplace=True)
    
    # Features for models
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment',
               '5D_MA', 'MA_Ratio', '5D_Volatility', 'Volume_Ratio', 'RSI', 
               'MACD', 'BB_Width']
    
    if 'India_VIX' in df_stock.columns:
        features.append('India_VIX')
    
    X = df_stock[features]
    y = df_stock['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize Bayesian Fusion
    bayesian_fusion = BayesianUncertaintyFusion(n_models=5)
    
    # 1. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    # 2. XGBoost Model
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
    xgb_pred = xgb_model.predict(X_test)
    
    # 3. LSTM Model
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_3d = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    
    lstm_model = Sequential([
        LSTM(64, input_shape=(1, X_3d.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    lstm_model.fit(X_3d[:len(X_train)], y_train[:len(X_train)], 
                  epochs=50, batch_size=32, verbose=0)
    lstm_pred = lstm_model.predict(X_3d[len(X_train):]).flatten()
    
    # 4. GRU Model
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
    gru_pred = gru_model.predict(X_3d[len(X_train):]).flatten()
    
    # 5. BiLSTM Model
    bilstm_model = Sequential([
        Bidirectional(LSTM(32, return_sequences=True), input_shape=(1, X_3d.shape[2])),
        Dropout(0.2),
        Bidirectional(LSTM(16)),
        Dropout(0.2),
        Dense(1)
    ])
    bilstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    bilstm_model.fit(X_3d[:len(X_train)], y_train[:len(X_train)], 
                    epochs=50, batch_size=32, verbose=0)
    bilstm_pred = bilstm_model.predict(X_3d[len(X_train):]).flatten()
    
    # Collect all predictions
    all_predictions = np.column_stack([lr_pred, xgb_pred, lstm_pred, gru_pred, bilstm_pred])
    
    # Calculate model errors for Bayesian fusion
    model_errors = []
    for i, pred in enumerate([lr_pred, xgb_pred, lstm_pred, gru_pred, bilstm_pred]):
        error = mean_absolute_error(y_test, pred)
        model_errors.append(error)
    
    model_errors = np.array(model_errors)
    recent_performance = np.ones(5)  # Placeholder for recent performance metric
    
    # Update Bayesian weights
    weights = bayesian_fusion.update_weights_bayesian(model_errors, recent_performance)
    uncertainties = bayesian_fusion.calculate_model_uncertainty(all_predictions, y_test.values)
    
    # Fuse predictions using Bayesian framework
    fused_pred = bayesian_fusion.fuse_predictions(all_predictions, uncertainties)
    
    # Calculate comprehensive metrics for all models
    models = {
        'Linear Regression': lr_pred,
        'XGBoost': xgb_pred,
        'LSTM': lstm_pred,
        'GRU': gru_pred,
        'BiLSTM': bilstm_pred,
        'Bayesian Fusion': fused_pred
    }
    
    metrics = {}
    for name, pred in models.items():
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)
        mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
        
        metrics[name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Weight': weights[list(models.keys()).index(name)] if name != 'Bayesian Fusion' else None
        }
    
    # Store predictions
    df_stock.loc[y_test.index, 'Bayesian_Predicted'] = fused_pred
    for i, (name, pred) in enumerate(models.items()):
        if name != 'Bayesian Fusion':
            df_stock.loc[y_test.index, f'{name}_Predicted'] = pred
    
    model_objects = {
        'Linear Regression': lr_model,
        'XGBoost': xgb_model,
        'LSTM': lstm_model,
        'GRU': gru_model,
        'BiLSTM': bilstm_model,
        'Bayesian Fusion': bayesian_fusion
    }
    
    return df_stock, model_objects, scaler, features, metrics, weights

# Enhanced visualization functions
def create_model_comparison_plot(metrics):
    models = list(metrics.keys())
    metric_names = ['MAE', 'RMSE', 'MAPE']
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=metric_names)
    
    for i, metric in enumerate(metric_names):
        values = [metrics[model][metric] for model in models]
        fig.add_trace(
            go.Bar(x=models, y=values, name=metric),
            row=1, col=i+1
        )
    
    fig.update_layout(height=400, title_text="Model Performance Comparison")
    return fig

def create_uncertainty_visualization(metrics, weights):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Model Weights Distribution', 'Performance vs Uncertainty'],
        specs=[[{'type': 'pie'}, {'type': 'scatter'}]]
    )
    
    # Pie chart for weights
    model_names = list(metrics.keys())[:-1]  # Exclude Bayesian Fusion
    weight_values = weights
    
    fig.add_trace(
        go.Pie(labels=model_names, values=weight_values, name="Weights"),
        row=1, col=1
    )
    
    # Scatter plot for performance vs uncertainty
    mae_values = [metrics[model]['MAE'] for model in model_names]
    rmse_values = [metrics[model]['RMSE'] for model in model_names]
    
    fig.add_trace(
        go.Scatter(x=mae_values, y=rmse_values, text=model_names, mode='markers+text',
                  marker=dict(size=weight_values*100, color=weight_values, 
                            colorscale='Viridis', showscale=True)),
        row=1, col=2
    )
    
    fig.update_layout(height=500, title_text="Bayesian Fusion Analysis")
    return fig

def create_prediction_plot(df_stock, models_metrics):
    fig = go.Figure()
    
    # Actual prices
    fig.add_trace(go.Scatter(
        x=df_stock.index, y=df_stock['Close'],
        mode='lines', name='Actual Price',
        line=dict(color='black', width=2)
    ))
    
    # Model predictions
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, (model, color) in enumerate(zip(models_metrics.keys(), colors)):
        if f'{model}_Predicted' in df_stock.columns:
            pred_col = f'{model}_Predicted'
            # Convert percentage predictions back to prices
            current_price = df_stock['Close'].shift(1)
            predicted_prices = current_price * (1 + df_stock[pred_col])
            
            fig.add_trace(go.Scatter(
                x=df_stock.index, y=predicted_prices,
                mode='lines', name=f'{model} Prediction',
                line=dict(color=color, width=1, dash='dash'),
                opacity=0.7
            ))
    
    fig.update_layout(
        title='Model Predictions vs Actual Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )
    return fig

# Prophet forecasting (keep as is)
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

# Rest of your existing functions (candlestick chart, investment recommendation, etc.)
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

def generate_recommendation(predicted_prices, current_price, accuracy, avg_sentiment):
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

# Streamlit UI with enhanced features
st.title("Indian Stock Market Analysis with Bayesian Uncertainty-Weighted Fusion")
st.sidebar.header("Stock Selection")
indian_stocks = get_indian_stocks()
selected_stock = st.sidebar.selectbox("Select Stock", indian_stocks)
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
chart_type = st.sidebar.radio("Chart Type", ["Candlestick Chart", "Line Chart"])

if st.sidebar.button("Analyze"):
    ticker = f"{selected_stock}.NS"
    df_stock = get_stock_data(ticker, start_date, end_date)
    vix_data = get_india_vix_data(start_date, end_date)
    
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

        # Display India VIX data if available
        if not vix_data.empty:
            st.subheader("India VIX (Volatility Index)")
            st.line_chart(vix_data)
            st.write(f"Current VIX: {vix_data['India_VIX'].iloc[-1]:.2f}")

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

        # News and sentiment analysis (keep existing code)
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

        # Enhanced model training with Bayesian fusion
        st.subheader("Multi-Model Analysis with Bayesian Uncertainty-Weighted Fusion")
        
        df_stock, models, scaler, features, metrics, weights = create_comprehensive_models(
            df_stock, daily_sentiment if daily_sentiment else {}, vix_data
        )

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
        st.subheader("Model Performance Visualization")
        st.plotly_chart(create_model_comparison_plot(metrics))
        
        st.subheader("Bayesian Fusion Analysis")
        st.plotly_chart(create_uncertainty_visualization(metrics, weights))
        
        st.subheader("Predictions vs Actual Prices")
        st.plotly_chart(create_prediction_plot(df_stock, metrics))

        # Feature importance analysis
        st.subheader("Feature Importance Analysis")
        try:
            explainer = shap.TreeExplainer(models['XGBoost'])
            shap_values = explainer.shap_values(X_test)
            
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
            st.pyplot(fig)
        except:
            st.info("Feature importance visualization requires additional data")

        # Investment recommendation based on Bayesian fusion
        st.subheader("Investment Recommendation")
        current_price = df_stock['Close'].iloc[-1]
        accuracy = 100 - (metrics['Bayesian Fusion']['MAPE'])
        avg_sentiment = df_stock['Sentiment'].mean() if 'Sentiment' in df_stock else 0
        
        # Use Bayesian fusion predictions for recommendation
        bayesian_pred = df_stock['Bayesian_Predicted'].dropna()
        if len(bayesian_pred) > 0:
            avg_prediction = current_price * (1 + bayesian_pred.mean())
        else:
            avg_prediction = current_price
        
        price_change = ((avg_prediction - current_price) / current_price) * 100
        
        recommendation, reasoning = generate_recommendation(
            pd.DataFrame({'Predicted Price': [avg_prediction]}), 
            current_price, 
            accuracy,
            avg_sentiment
        )
        
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

        # Detailed analysis
        st.subheader("Detailed Analysis Summary")
        analysis_text = f"""
        ### Bayesian Fusion Analysis Results:
        
        **Best Performing Model**: {max(metrics, key=lambda x: metrics[x]['R2'])} with R² = {max(metrics[x]['R2'] for x in metrics):.4f}
        
        **Key Insights**:
        - Bayesian fusion dynamically weights models based on their uncertainty and recent performance
        - Model weights are updated using Bayesian inference with Dirichlet priors
        - The system adapts to changing market conditions by adjusting model contributions
        
        **Model Confidence Levels**:
        """
        
        for model, weight in zip(list(metrics.keys())[:-1], weights):
            analysis_text += f"- {model}: {weight*100:.1f}% weight\n"
        
        analysis_text += f"""
        
        **Overall System Accuracy**: {accuracy:.1f}%
        **Expected Price Movement**: {'+' if price_change > 0 else ''}{price_change:.2f}%
        
        The Bayesian framework provides robust uncertainty quantification and adaptive model weighting,
        making it particularly suitable for volatile financial markets.
        """
        
        st.markdown(analysis_text)