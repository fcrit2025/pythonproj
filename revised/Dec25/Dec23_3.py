import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')
import re
import os
from scipy import stats

# Set page configuration
st.set_page_config(
    page_title="Dynamic Uncertainty-Weighted Fusion for Financial Forecasting",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Dynamic Uncertainty-Weighted Fusion for Multi-Source Financial Data")
st.markdown("""
A Bayesian Framework for Time-Varying Source Reliability in Stock Price Forecasting
""")

# Custom Indian holiday calendar
class IndiaHolidayCalendar:
    @staticmethod
    def is_holiday(date):
        holidays = {
            '2024-01-26': 'Republic Day',
            '2024-08-15': 'Independence Day',
            '2024-10-02': 'Gandhi Jayanti',
            '2024-10-24': 'Diwali',
            '2024-03-25': 'Holi',
        }
        return date.strftime('%Y-%m-%d') in holidays

# Load FinBERT sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finbert",
            device=-1  # Use CPU
        )
        return sentiment_pipeline
    except:
        # Fallback to simpler sentiment analysis
        from textblob import TextBlob
        return lambda x: [{'label': 'positive' if TextBlob(x).sentiment.polarity > 0 else 'negative', 
                         'score': abs(TextBlob(x).sentiment.polarity)}]

# Initialize sentiment model
sentiment_pipeline = load_sentiment_model()

# Load Indian stock symbols
@st.cache_data
def get_indian_stocks():
    # Create a comprehensive list of Indian stocks
    indian_stocks = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "ITC.NS", "BHARTIARTL.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "LT.NS", "BAJFINANCE.NS", "WIPRO.NS", "HINDUNILVR.NS", "ASIANPAINT.NS",
        "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ONGC.NS", "NTPC.NS"
    ]
    return indian_stocks

# Fetch stock data
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            # Try without .NS suffix
            if ticker.endswith('.NS'):
                ticker = ticker[:-3]
                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            return None
        
        # Clean the data
        data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        data.index = pd.to_datetime(data.index)
        data.index = data.index.tz_localize(None)
        
        # Handle missing values
        data = data.fillna(method='ffill')
        data = data.fillna(method='bfill')
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Fetch FII/DII data from Groww
@st.cache_data
def get_fii_dii_data():
    try:
        # Create synthetic FII/DII data for demonstration
        # In production, replace with actual API call to https://groww.in/fii-dii-data
        
        # Generate date range for last 2 years
        dates = pd.date_range(start='2022-01-01', end=datetime.now().strftime('%Y-%m-%d'), freq='B')  # Business days
        n_days = len(dates)
        
        # Create realistic FII/DII data patterns
        np.random.seed(42)
        
        # Base trends with some randomness
        fii_trend = np.cumsum(np.random.randn(n_days) * 50 + 20)
        dii_trend = np.cumsum(np.random.randn(n_days) * 40 + 15)
        
        # Add seasonal patterns
        seasonal = 100 * np.sin(np.linspace(0, 4 * np.pi, n_days))
        fii_trend += seasonal
        dii_trend -= seasonal * 0.8
        
        # Ensure positive values
        fii_trend = np.maximum(fii_trend, 100)
        dii_trend = np.maximum(dii_trend, 100)
        
        fii_dii_data = pd.DataFrame({
            'Date': dates,
            'FII_Net_Investment': fii_trend,
            'DII_Net_Investment': dii_trend,
            'FII_Buy_Value': np.random.randint(1000, 5000, n_days),
            'FII_Sell_Value': np.random.randint(800, 4500, n_days),
            'DII_Buy_Value': np.random.randint(800, 4000, n_days),
            'DII_Sell_Value': np.random.randint(700, 3500, n_days),
        })
        
        fii_dii_data.set_index('Date', inplace=True)
        return fii_dii_data
    except Exception as e:
        st.warning(f"Could not fetch FII/DII data: {str(e)}")
        return None

# Fetch stock info
@st.cache_data
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        def format_value(value, format_str):
            if value is None or pd.isna(value):
                return "N/A"
            try:
                return format_str.format(value)
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
    except:
        return {}

# News API functions
NEWS_API_KEY = "563215a35c1a47968f46271e04083ea3"
NEWS_API_URL = "https://newsapi.org/v2/everything"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_news(stock_symbol):
    try:
        stock_name_mapping = {
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "Tata Consultancy Services",
            "INFY.NS": "Infosys",
            "HDFCBANK.NS": "HDFC Bank",
            "ICICIBANK.NS": "ICICI Bank",
            "RELIANCE": "Reliance Industries",
            "TCS": "Tata Consultancy Services",
            "INFY": "Infosys",
            "HDFCBANK": "HDFC Bank",
            "ICICIBANK": "ICICI Bank",
        }
        
        query = stock_name_mapping.get(stock_symbol, stock_symbol.replace('.NS', ''))
        
        params = {
            "q": f"{query} stock",
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20
        }
        
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            return articles[:10]  # Return top 10 articles
        else:
            st.warning(f"News API returned status {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"Could not fetch news: {str(e)}")
        return []

# Sentiment analysis
def analyze_sentiment(text):
    if not text or pd.isna(text):
        return "neutral", 0.0
    
    try:
        # Clean text
        text = str(text)[:512]
        result = sentiment_pipeline(text)[0]
        label = result['label']
        score = result['score']
        
        # Map to our labels
        if label == 'positive':
            return 'positive', score
        elif label == 'negative':
            return 'negative', score
        else:
            return 'neutral', 0.5
    except:
        # Fallback to simple sentiment analysis
        text_lower = str(text).lower()
        positive_words = ['profit', 'growth', 'gain', 'rise', 'up', 'bullish', 'positive']
        negative_words = ['loss', 'fall', 'down', 'bearish', 'negative', 'decline']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'positive', min(0.5 + pos_count * 0.1, 0.95)
        elif neg_count > pos_count:
            return 'negative', min(0.5 + neg_count * 0.1, 0.95)
        else:
            return 'neutral', 0.5

# Feature engineering - FIXED VERSION
def create_advanced_features(df):
    df = df.copy()
    
    # Basic features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    for window in [5, 20, 50]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    
    # Moving average ratios
    df['MA_Ratio_5_20'] = df['MA_5'] / df['MA_20']
    
    # Volatility measures
    df['Volatility_5'] = df['Returns'].rolling(window=5).std()
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()
    
    # Volume features
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    
    # Price momentum
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Drop NaN values
    df = df.dropna()
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    
    return df

# Create sentiment features
def create_sentiment_features(news_articles, date):
    """Create daily sentiment features"""
    if not news_articles:
        return pd.Series([0.0, 0.0, 0.0], index=['Sentiment_Score', 'News_Count', 'Sentiment_Trend'])
    
    sentiments = []
    scores = []
    
    for article in news_articles:
        title = article.get('title', '')
        description = article.get('description', '')
        
        text = f"{title} {description}"
        sentiment, score = analyze_sentiment(text)
        
        # Convert sentiment to numerical value
        if sentiment == 'positive':
            sentiment_value = score
        elif sentiment == 'negative':
            sentiment_value = -score
        else:
            sentiment_value = 0
        
        sentiments.append(sentiment_value)
        scores.append(score)
    
    if sentiments:
        return pd.Series([
            np.mean(sentiments),  # Average sentiment
            len(sentiments),      # News count
            np.mean(scores)       # Average confidence
        ], index=['Sentiment_Score', 'News_Count', 'Sentiment_Trend'])
    else:
        return pd.Series([0.0, 0.0, 0.0], index=['Sentiment_Score', 'News_Count', 'Sentiment_Trend'])

# Create FII/DII features
def create_fii_dii_features(fii_dii_data, date):
    if fii_dii_data is None:
        # Return default values if no FII/DII data
        return pd.Series([0.0, 0.0, 0.0], index=['FII_Net', 'DII_Net', 'Net_Flow'])
    
    # Find the closest available date
    if date in fii_dii_data.index:
        data_date = date
    else:
        # Find nearest previous business day
        available_dates = fii_dii_data.index[fii_dii_data.index <= date]
        if len(available_dates) > 0:
            data_date = available_dates[-1]
        else:
            # Return default values
            return pd.Series([0.0, 0.0, 0.0], index=['FII_Net', 'DII_Net', 'Net_Flow'])
    
    try:
        # Get data for the specific date
        data = fii_dii_data.loc[data_date]
        
        # Calculate features
        fii_net = float(data.get('FII_Net_Investment', 0))
        dii_net = float(data.get('DII_Net_Investment', 0))
        net_flow = fii_net + dii_net
        
        return pd.Series([
            fii_net,
            dii_net,
            net_flow
        ], index=['FII_Net', 'DII_Net', 'Net_Flow'])
    except:
        return pd.Series([0.0, 0.0, 0.0], index=['FII_Net', 'DII_Net', 'Net_Flow'])

# Neural Network Models
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        transformer_out = self.transformer(x)
        out = self.fc(transformer_out[:, -1, :])
        return out

# Dynamic Uncertainty-Weighted Fusion Model
class DynamicUncertaintyFusion:
    def __init__(self, n_sources=3, window_size=10, temperature=1.0):
        self.n_sources = n_sources
        self.window_size = window_size
        self.temperature = temperature
        self.uncertainties = []
        self.weights_history = []
        
    def calculate_uncertainty(self, predictions, actuals):
        """Calculate time-varying uncertainty for each source"""
        uncertainties = []
        for i in range(self.n_sources):
            if len(predictions[i]) > 0 and len(actuals) > 0:
                # Calculate prediction errors
                errors = predictions[i] - actuals
                
                # Calculate uncertainty as rolling variance of errors
                if len(errors) >= self.window_size:
                    error_variance = np.var(errors[-self.window_size:])
                else:
                    error_variance = np.var(errors) if len(errors) > 1 else 1.0
                
                # Add small epsilon to avoid division by zero
                uncertainties.append(error_variance + 1e-8)
            else:
                uncertainties.append(1.0)  # Default high uncertainty
        
        return np.array(uncertainties)
    
    def calculate_weights(self, uncertainties):
        """Calculate dynamic weights using softmax over negative uncertainties"""
        # Convert to numpy array
        uncertainties = np.array(uncertainties)
        
        # Apply softmax over negative uncertainties (higher uncertainty = lower weight)
        exp_values = np.exp(-uncertainties / self.temperature)
        weights = exp_values / np.sum(exp_values)
        
        return weights
    
    def fuse_predictions(self, predictions, uncertainties=None):
        """Fuse predictions using uncertainty-weighted averaging"""
        if uncertainties is None:
            # If no uncertainties provided, use equal weights
            weights = np.ones(self.n_sources) / self.n_sources
        else:
            weights = self.calculate_weights(uncertainties)
        
        # Store weights history
        self.weights_history.append(weights.copy())
        
        # Weighted average of predictions
        min_length = min(len(p) for p in predictions)
        fused_prediction = np.zeros(min_length)
        for i in range(self.n_sources):
            fused_prediction += weights[i] * predictions[i][:min_length]
        
        return fused_prediction, weights

# Create dataset for training - FIXED VERSION
class FinancialDataset(Dataset):
    def __init__(self, features, targets, sequence_length=20):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        features_seq = self.features[idx:idx+self.sequence_length]
        target = self.targets[idx+self.sequence_length]
        return torch.FloatTensor(features_seq), torch.FloatTensor([target])

# Train model
def train_model(model, train_loader, val_loader, n_epochs=30, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                predictions = model(batch_features)
                loss = criterion(predictions, batch_targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
    
    return train_losses, val_losses

# Prepare data for models - FIXED VERSION
def prepare_data(features_df, target_series, sequence_length=20, test_size=0.2):
    """Prepare data for model training with proper alignment"""
    # Ensure features and targets are aligned
    common_index = features_df.index.intersection(target_series.index)
    features_aligned = features_df.loc[common_index]
    targets_aligned = target_series.loc[common_index]
    
    # Scale features
    scaler_X = StandardScaler()
    features_scaled = scaler_X.fit_transform(features_aligned.values)
    
    # Scale targets
    scaler_y = StandardScaler()
    targets_scaled = scaler_y.fit_transform(targets_aligned.values.reshape(-1, 1)).flatten()
    
    # Split data
    n_train = int(len(features_scaled) * (1 - test_size))
    X_train = features_scaled[:n_train]
    y_train = targets_scaled[:n_train]
    X_test = features_scaled[n_train:]
    y_test = targets_scaled[n_train:]
    
    # Create datasets
    train_dataset = FinancialDataset(X_train, y_train, sequence_length)
    test_dataset = FinancialDataset(X_test, y_test, sequence_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return train_loader, test_loader, scaler_X, scaler_y, common_index

# Evaluate model
def evaluate_model(model, test_loader, scaler_y):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_predictions = model(batch_features)
            predictions.extend(batch_predictions.numpy().flatten())
            actuals.extend(batch_targets.numpy().flatten())
    
    # Inverse transform predictions and actuals
    predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100  # Add epsilon to avoid division by zero
    r2 = r2_score(actuals, predictions)
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

# Main application
def main():
    # Sidebar for user inputs
    st.sidebar.header("Configuration")
    
    # Stock selection
    indian_stocks = get_indian_stocks()
    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        indian_stocks,
        index=0
    )
    
    # Date range selection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date_input = st.date_input("Start Date", start_date)
    with col2:
        end_date_input = st.date_input("End Date", end_date)
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    sequence_length = st.sidebar.slider("Sequence Length", 10, 60, 20)
    n_epochs = st.sidebar.slider("Training Epochs", 10, 100, 30)
    
    # Fusion parameters
    st.sidebar.subheader("Fusion Parameters")
    uncertainty_window = st.sidebar.slider("Uncertainty Window", 5, 30, 10)
    temperature = st.sidebar.slider("Temperature", 0.1, 5.0, 1.0, 0.1)
    
    # Fetch data button
    if st.sidebar.button("Fetch Data and Run Analysis", type="primary"):
        with st.spinner("Fetching data and running analysis..."):
            # Main content area
            st.header(f"Analysis for {selected_stock}")
            
            # Fetch stock data
            stock_data = get_stock_data(selected_stock, start_date_input, end_date_input)
            
            if stock_data is None or stock_data.empty:
                st.error("No data available for the selected stock and date range.")
                return
            
            # Display stock info
            st.subheader("Stock Information")
            stock_info = get_stock_info(selected_stock)
            
            if stock_info:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Market Cap", stock_info.get("Market Cap", "N/A"))
                    st.metric("Current Price", stock_info.get("Current Price", "N/A"))
                with col2:
                    st.metric("P/E Ratio", stock_info.get("P/E Ratio", "N/A"))
                    st.metric("Book Value", stock_info.get("Book Value", "N/A"))
                with col3:
                    st.metric("ROCE", stock_info.get("ROCE", "N/A"))
                    st.metric("ROE", stock_info.get("ROE", "N/A"))
                with col4:
                    st.metric("Dividend Yield", stock_info.get("Dividend Yield", "N/A"))
                    st.metric("Face Value", stock_info.get("Face Value", "N/A"))
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Data Overview", 
                "🤖 Model Training", 
                "🔮 Predictions", 
                "📊 Fusion Analysis",
                "📰 News Sentiment"
            ])
            
            with tab1:
                # Display candlestick chart
                st.subheader("Price Chart")
                
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('Price and Volume', 'Technical Indicators', 'Returns Distribution'),
                    row_heights=[0.5, 0.25, 0.25]
                )
                
                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=stock_data.index,
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        name='OHLC'
                    ),
                    row=1, col=1
                )
                
                # Add moving averages
                stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
                stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=stock_data['MA_20'],
                        name='20-Day MA',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=stock_data['MA_50'],
                        name='50-Day MA',
                        line=dict(color='red', width=1)
                    ),
                    row=1, col=1
                )
                
                # Volume chart
                colors = ['green' if stock_data['Close'].iloc[i] > stock_data['Open'].iloc[i] 
                         else 'red' for i in range(len(stock_data))]
                
                fig.add_trace(
                    go.Bar(
                        x=stock_data.index,
                        y=stock_data['Volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.5
                    ),
                    row=2, col=1
                )
                
                # RSI
                delta = stock_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                stock_data['RSI'] = 100 - (100 / (1 + rs))
                
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=stock_data['RSI'],
                        name='RSI',
                        line=dict(color='purple', width=1)
                    ),
                    row=3, col=1
                )
                
                # Add RSI bands
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
                fig.update_layout(
                    height=800,
                    showlegend=True,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display data statistics
                st.subheader("Data Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Days", len(stock_data))
                    st.metric("Average Volume", f"{stock_data['Volume'].mean():,.0f}")
                    
                with col2:
                    returns = stock_data['Close'].pct_change().dropna()
                    st.metric("Average Daily Return", f"{returns.mean()*100:.2f}%")
                    st.metric("Volatility (Std Dev)", f"{returns.std()*100:.2f}%")
                    
                with col3:
                    st.metric("Sharpe Ratio", f"{returns.mean()/returns.std():.2f}" if returns.std() > 0 else "N/A")
                    st.metric("Max Drawdown", f"{(stock_data['Close']/stock_data['Close'].cummax()-1).min()*100:.2f}%")
            
            # Feature engineering - FIXED
            stock_data_features = create_advanced_features(stock_data)
            
            # Fetch additional data
            fii_dii_data = get_fii_dii_data()
            news_articles = get_news(selected_stock)
            
            # Create multi-source dataset - FIXED
            st.subheader("Creating Multi-Source Dataset...")
            
            # Get common dates (after feature engineering)
            aligned_dates = stock_data_features.index
            
            # Create source-specific dataframes
            source_dfs = {}
            
            # Source 1: Technical features
            source_dfs['Technical'] = stock_data_features
            
            # Source 2: News sentiment features
            sentiment_features_list = []
            for date in aligned_dates:
                sentiment_features = create_sentiment_features(news_articles, date)
                sentiment_features_list.append(sentiment_features)
            
            source_dfs['Sentiment'] = pd.DataFrame(
                sentiment_features_list,
                index=aligned_dates,
                columns=['Sentiment_Score', 'News_Count', 'Sentiment_Trend']
            )
            
            # Source 3: FII/DII features
            fii_dii_features_list = []
            for date in aligned_dates:
                fii_dii_features = create_fii_dii_features(fii_dii_data, date)
                fii_dii_features_list.append(fii_dii_features)
            
            source_dfs['FII/DII'] = pd.DataFrame(
                fii_dii_features_list,
                index=aligned_dates,
                columns=['FII_Net', 'DII_Net', 'Net_Flow']
            )
            
            # Prepare target variable (next day's close price)
            target_series = stock_data['Close'].shift(-1)  # Predict next day's close
            
            # Align target with features (remove last value which has no target)
            target_series = target_series.loc[aligned_dates]
            
            with tab2:
                st.subheader("Model Training")
                
                # Train individual models
                models = {}
                results = {}
                
                progress_bar = st.progress(0)
                
                for i, (source_name, source_df) in enumerate(source_dfs.items()):
                    st.write(f"Training {source_name} Model...")
                    
                    # Align source data with target
                    common_dates = source_df.index.intersection(target_series.index)
                    if len(common_dates) < sequence_length + 10:  # Need enough data
                        st.warning(f"Not enough aligned data for {source_name} model. Skipping...")
                        continue
                    
                    source_aligned = source_df.loc[common_dates]
                    target_aligned = target_series.loc[common_dates]
                    
                    # Prepare data for this source
                    train_loader, test_loader, scaler_X, scaler_y, _ = prepare_data(
                        source_aligned,
                        target_aligned,
                        sequence_length=sequence_length
                    )
                    
                    # Select model architecture based on source
                    input_size = source_aligned.shape[1]
                    if source_name == 'Technical':
                        model = GRUModel(input_size=input_size)
                    elif source_name == 'Sentiment':
                        model = TransformerModel(input_size=input_size)
                    else:  # FII/DII
                        model = LSTMModel(input_size=input_size)
                    
                    # Train model
                    train_losses, val_losses = train_model(
                        model, train_loader, test_loader, n_epochs=n_epochs
                    )
                    
                    # Evaluate model
                    evaluation = evaluate_model(model, test_loader, scaler_y)
                    
                    # Store results
                    models[source_name] = {
                        'model': model,
                        'scaler_X': scaler_X,
                        'scaler_y': scaler_y,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'evaluation': evaluation
                    }
                    
                    results[source_name] = evaluation
                    
                    # Plot training history
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(
                        y=train_losses, mode='lines', name='Training Loss'
                    ))
                    fig_loss.add_trace(go.Scatter(
                        y=val_losses, mode='lines', name='Validation Loss'
                    ))
                    fig_loss.update_layout(
                        title=f'{source_name} Model Training History',
                        xaxis_title='Epoch',
                        yaxis_title='Loss',
                        height=300
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                    
                    progress_bar.progress((i + 1) / len(source_dfs))
            
            with tab3:
                st.subheader("Model Predictions")
                
                if not models:
                    st.warning("No models were successfully trained. Please check your data.")
                    return
                
                # Make predictions with all models
                all_predictions = {}
                all_actuals = None
                
                for source_name, model_info in models.items():
                    predictions = model_info['evaluation']['predictions']
                    actuals = model_info['evaluation']['actuals']
                    all_predictions[source_name] = predictions
                    
                    if all_actuals is None:
                        all_actuals = actuals
                    elif len(actuals) < len(all_actuals):
                        all_actuals = all_actuals[:len(actuals)]
                    elif len(actuals) > len(all_actuals):
                        predictions = predictions[:len(all_actuals)]
                        all_predictions[source_name] = predictions
                
                # Create dynamic uncertainty-weighted fusion
                fusion_model = DynamicUncertaintyFusion(
                    n_sources=len(all_predictions),
                    window_size=uncertainty_window,
                    temperature=temperature
                )
                
                # Calculate uncertainties for each source
                uncertainties = []
                for i, (source_name, predictions) in enumerate(all_predictions.items()):
                    if len(predictions) == len(all_actuals):
                        errors = predictions - all_actuals
                        uncertainty = np.var(errors[-uncertainty_window:]) if len(errors) >= uncertainty_window else np.var(errors)
                        uncertainties.append(uncertainty + 1e-8)
                    else:
                        uncertainties.append(1.0)
                
                # Fuse predictions
                predictions_list = list(all_predictions.values())
                fused_predictions, weights = fusion_model.fuse_predictions(predictions_list, uncertainties)
                
                # Evaluate fused predictions
                fused_actuals = all_actuals[:len(fused_predictions)]
                fused_mse = mean_squared_error(fused_actuals, fused_predictions)
                fused_mae = mean_absolute_error(fused_actuals, fused_predictions)
                fused_rmse = np.sqrt(fused_mse)
                fused_r2 = r2_score(fused_actuals, fused_predictions)
                
                # Display comparison
                st.subheader("Model Performance Comparison")
                
                comparison_data = []
                for source_name, eval_results in results.items():
                    comparison_data.append({
                        'Model': source_name,
                        'RMSE': eval_results['rmse'],
                        'MAE': eval_results['mae'],
                        'R² Score': eval_results['r2'],
                        'MAPE': eval_results['mape']
                    })
                
                # Add fused model
                comparison_data.append({
                    'Model': 'Dynamic Fusion',
                    'RMSE': fused_rmse,
                    'MAE': fused_mae,
                    'R² Score': fused_r2,
                    'MAPE': np.mean(np.abs((fused_actuals - fused_predictions) / (fused_actuals + 1e-8))) * 100
                })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE'], color='lightgreen')
                                     .highlight_max(subset=['R² Score'], color='lightblue'), 
                                     use_container_width=True)
                
                # Plot predictions
                fig_predictions = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Predictions vs Actuals', 'Prediction Errors'),
                    vertical_spacing=0.15
                )
                
                # Create dates for predictions
                pred_dates = pd.date_range(end=aligned_dates[-1], periods=len(fused_actuals), freq='D')
                
                # Plot actual vs predictions
                fig_predictions.add_trace(
                    go.Scatter(x=pred_dates, y=fused_actuals, 
                              name='Actual', mode='lines', line=dict(color='blue')),
                    row=1, col=1
                )
                
                fig_predictions.add_trace(
                    go.Scatter(x=pred_dates, y=fused_predictions, 
                              name='Dynamic Fusion', mode='lines', line=dict(color='red', dash='dash')),
                    row=1, col=1
                )
                
                # Add individual model predictions
                colors = ['green', 'orange', 'purple']
                for i, (source_name, predictions) in enumerate(all_predictions.items()):
                    if len(predictions) == len(fused_actuals):
                        fig_predictions.add_trace(
                            go.Scatter(x=pred_dates, y=predictions, 
                                      name=source_name, mode='lines', 
                                      line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                                      opacity=0.5),
                            row=1, col=1
                        )
                
                # Plot errors
                errors = fused_actuals - fused_predictions
                fig_predictions.add_trace(
                    go.Scatter(x=pred_dates, y=errors, 
                              name='Error', mode='lines', 
                              line=dict(color='gray')),
                    row=2, col=1
                )
                
                fig_predictions.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
                
                fig_predictions.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig_predictions, use_container_width=True)
            
            with tab4:
                st.subheader("Dynamic Fusion Analysis")
                
                # Display fusion weights
                if hasattr(fusion_model, 'weights_history') and len(fusion_model.weights_history) > 0:
                    weights_array = np.array(fusion_model.weights_history)
                    
                    # Create DataFrame for weights
                    weights_df = pd.DataFrame(
                        weights_array,
                        columns=list(all_predictions.keys()),
                        index=pred_dates[-len(weights_array):]
                    )
                    
                    # Plot weights evolution
                    fig_weights = go.Figure()
                    for source in weights_df.columns:
                        fig_weights.add_trace(go.Scatter(
                            x=weights_df.index,
                            y=weights_df[source],
                            name=source,
                            mode='lines'
                        ))
                    
                    fig_weights.update_layout(
                        title='Dynamic Fusion Weights Over Time',
                        xaxis_title='Date',
                        yaxis_title='Weight',
                        height=400
                    )
                    st.plotly_chart(fig_weights, use_container_width=True)
                    
                    # Display weight statistics
                    st.subheader("Weight Statistics")
                    weight_stats = weights_df.describe()
                    st.dataframe(weight_stats, use_container_width=True)
                    
                    # Display uncertainty analysis
                    st.subheader("Uncertainty Analysis")
                    
                    uncertainty_data = []
                    for source_name, predictions in all_predictions.items():
                        if len(predictions) == len(fused_actuals):
                            errors = predictions - fused_actuals
                            uncertainty = np.var(errors)
                            mae = np.mean(np.abs(errors))
                            
                            uncertainty_data.append({
                                'Source': source_name,
                                'Uncertainty (Variance)': uncertainty,
                                'MAE': mae,
                                'Std Dev': np.std(errors),
                                'Max Error': np.max(np.abs(errors))
                            })
                    
                    uncertainty_df = pd.DataFrame(uncertainty_data)
                    st.dataframe(uncertainty_df.style.highlight_min(subset=['Uncertainty (Variance)', 'MAE', 'Std Dev', 'Max Error'], 
                                                                    color='lightgreen'), 
                                 use_container_width=True)
            
            with tab5:
                st.subheader("News Sentiment Analysis")
                
                if news_articles:
                    # Display news articles with sentiment
                    sentiment_scores = []
                    
                    for i, article in enumerate(news_articles[:10]):  # Show top 10 articles
                        title = article.get('title', 'No title')
                        description = article.get('description', '')
                        url = article.get('url', '#')
                        published_at = article.get('publishedAt', '')
                        
                        # Analyze sentiment
                        text = f"{title} {description}"
                        sentiment, score = analyze_sentiment(text)
                        
                        # Determine color based on sentiment
                        if sentiment == 'positive':
                            color = 'green'
                            emoji = '📈'
                        elif sentiment == 'negative':
                            color = 'red'
                            emoji = '📉'
                        else:
                            color = 'gray'
                            emoji = '➖'
                        
                        # Display article
                        with st.expander(f"{emoji} {title}", expanded=i==0):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**Published:** {published_at}")
                                if description:
                                    st.write(description)
                            with col2:
                                st.metric("Sentiment", sentiment, 
                                         delta=f"Score: {score:.2f}" if sentiment != 'neutral' else None)
                                if url != '#':
                                    st.markdown(f"[Read more]({url})")
                        
                        sentiment_scores.append(score if sentiment == 'positive' else -score)
                    
                    # Plot sentiment distribution
                    if sentiment_scores:
                        fig_sentiment = go.Figure()
                        fig_sentiment.add_trace(go.Histogram(
                            x=sentiment_scores,
                            nbinsx=10,
                            name='Sentiment Distribution',
                            marker_color='lightblue'
                        ))
                        
                        fig_sentiment.update_layout(
                            title='Sentiment Score Distribution',
                            xaxis_title='Sentiment Score',
                            yaxis_title='Count',
                            height=400
                        )
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                        # Calculate overall sentiment
                        avg_sentiment = np.mean(sentiment_scores)
                        st.metric("Overall Sentiment", 
                                 "Bullish" if avg_sentiment > 0 else "Bearish" if avg_sentiment < 0 else "Neutral",
                                 delta=f"{avg_sentiment:.2f}")
                else:
                    st.info("No news articles found for this stock.")
            
            # Display conclusion
            st.success("Analysis Complete! The Dynamic Uncertainty-Weighted Fusion model successfully integrated multiple data sources with adaptive weighting based on time-varying source reliability.")

if __name__ == "__main__":
    main()