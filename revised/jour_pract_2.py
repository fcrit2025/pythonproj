import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set page configuration
st.set_page_config(
    page_title="Nifty 50 AI Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .news-card {
        background-color: #ffffff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

class Nifty50DataCollector:
    def __init__(self):
        self.nifty_symbol = "^NSEI"
        self.vix_symbol = "^INDIAVIX"
        
    def fetch_market_data(self, start_date, end_date):
        """Fetch Nifty 50 and India VIX data"""
        try:
            # Fetch Nifty 50 data
            nifty_data = yf.download(self.nifty_symbol, start=start_date, end=end_date, progress=False)
            # Fetch India VIX data
            vix_data = yf.download(self.vix_symbol, start=start_date, end=end_date, progress=False)
            
            return nifty_data, vix_data
        except Exception as e:
            st.error(f"Error fetching market data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def calculate_technical_indicators(self, df):
    """Calculate technical indicators for Nifty 50"""
    try:
        df = df.copy()
        
        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_Ratio_5_20'] = df['MA_5'] / df['MA_20']
        df['MA_Ratio_20_50'] = df['MA_20'] / df['MA_50']
        
        # Volatility indicators
        df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
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
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands - FIXED: Calculate each column separately
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        
        # Safe BB_Position calculation to avoid division by zero
        bb_range = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = np.where(bb_range != 0, 
                                   (df['Close'] - df['BB_Lower']) / bb_range, 
                                   0.5)  # Default to middle if range is zero
        
        # Volume indicators
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
        
        # Price ranges
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Handle missing values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return df

    def apply_distribution_aware_normalization(self, df):
    """Apply different scalers based on data distribution"""
        try:
            df_scaled = df.copy()
        
        # Technical indicators - StandardScaler
        tech_columns = ['MA_5', 'MA_20', 'MA_50', 'MA_Ratio_5_20', 'MA_Ratio_20_50', 
                       'Volatility_5', 'Volatility_20', 'RSI', 'MACD', 'MACD_Signal', 
                       'MACD_Histogram', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position']
        
        # Check which technical columns actually exist in the dataframe
        available_tech_columns = [col for col in tech_columns if col in df.columns]
        
        if available_tech_columns:
            tech_scaler = StandardScaler()
            df_scaled[available_tech_columns] = tech_scaler.fit_transform(df[available_tech_columns])
            self.scalers['technical'] = tech_scaler
        
        # VIX data - RobustScaler (handles outliers well)
        vix_columns = ['VIX_Close', 'VIX_High', 'VIX_Low']
        available_vix_columns = [col for col in vix_columns if col in df.columns]
        
        if available_vix_columns:
            vix_scaler = RobustScaler()
            df_scaled[available_vix_columns] = vix_scaler.fit_transform(df[available_vix_columns])
            self.scalers['vix'] = vix_scaler
        
        # Sentiment scores - MinMaxScaler to [-1, 1]
        if 'Sentiment_Score' in df.columns:
            sentiment_scaler = MinMaxScaler(feature_range=(-1, 1))
            df_scaled[['Sentiment_Score']] = sentiment_scaler.fit_transform(df[['Sentiment_Score']])
            self.scalers['sentiment'] = sentiment_scaler
        
        # Price and volume - StandardScaler
        price_volume_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Volume_MA_5', 'Volume_Ratio']
        available_price_volume_columns = [col for col in price_volume_columns if col in df.columns]
        
        if available_price_volume_columns:
            price_scaler = StandardScaler()
            df_scaled[available_price_volume_columns] = price_scaler.fit_transform(df[available_price_volume_columns])
            self.scalers['price_volume'] = price_scaler
        
        return df_scaled
    except Exception as e:
        st.error(f"Error in normalization: {e}")
        return df

class NewsSentimentAnalyzer:
    def __init__(self):
        self.api_key = "563215a35c1a47968f46271e04083ea3"
        self.base_url = "https://newsapi.org/v2/everything"
        
    def analyze_sentiment(self, text):
        """Simplified sentiment analysis"""
        if not text or pd.isna(text):
            return "neutral", 0.5
        
        positive_words = ['up', 'rise', 'gain', 'bullish', 'positive', 'growth', 'profit', 
                         'increase', 'strong', 'beat', 'surge', 'rally', 'soar', 'higher']
        negative_words = ['down', 'fall', 'drop', 'bearish', 'negative', 'loss', 'decline', 
                         'decrease', 'weak', 'miss', 'plunge', 'slide', 'tumble', 'lower']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            confidence = min(0.5 + (positive_count * 0.08), 0.95)
            return "positive", confidence
        elif negative_count > positive_count:
            confidence = min(0.5 + (negative_count * 0.08), 0.95)
            return "negative", confidence
        else:
            return "neutral", 0.5
    
    def get_financial_news(self, days_back=30):
        """Fetch financial news for India and global markets"""
        try:
            queries = [
                "Nifty 50 India",
                "Indian stock market", 
                "Sensex India",
                "RBI policy",
                "Indian economy",
                "global markets",
                "US Federal Reserve",
                "crude oil prices",
                "inflation rates"
            ]
            
            all_articles = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            for query in queries:
                params = {
                    "q": query,
                    "apiKey": self.api_key,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "from": start_date.strftime("%Y-%m-%d"),
                    "to": end_date.strftime("%Y-%m-%d"),
                    "pageSize": 10
                }
                
                response = requests.get(self.base_url, params=params, timeout=10)
                if response.status_code == 200:
                    articles = response.json().get("articles", [])
                    all_articles.extend(articles)
            
            # Remove duplicates
            unique_articles = []
            seen_titles = set()
            
            for article in all_articles:
                title = article.get('title', '')
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_articles.append(article)
            
            return unique_articles
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            return []
    
    def create_daily_sentiment_scores(self, news_articles):
        """Create daily sentiment scores from news articles"""
        daily_sentiment = {}
        
        for article in news_articles:
            title = article.get('title', '')
            description = article.get('description', '')
            content = f"{title} {description}".strip()
            published_at = article.get('publishedAt', '')
            
            if not published_at:
                continue
                
            try:
                # Extract date from publishedAt
                date_str = published_at.split('T')[0]
                sentiment, confidence = self.analyze_sentiment(content)
                
                # Convert sentiment to numerical score
                if sentiment == "positive":
                    score = confidence
                elif sentiment == "negative":
                    score = -confidence
                else:
                    score = 0
                
                if date_str in daily_sentiment:
                    daily_sentiment[date_str].append(score)
                else:
                    daily_sentiment[date_str] = [score]
                    
            except Exception:
                continue
        
        # Calculate daily average sentiment
        daily_avg_sentiment = {}
        for date, scores in daily_sentiment.items():
            daily_avg_sentiment[date] = np.mean(scores)
        
        return daily_avg_sentiment

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        
    def integrate_all_data(self, nifty_data, vix_data, sentiment_scores):
        """Integrate all data sources with proper alignment"""
        try:
            # Create base dataframe from Nifty data
            df = nifty_data.copy()
            
            # Add VIX data
            vix_columns = ['Close', 'High', 'Low', 'Volume']
            for col in vix_columns:
                df[f'VIX_{col}'] = vix_data[col]
            
            # Add sentiment scores with date alignment
            df['Date_Str'] = df.index.strftime('%Y-%m-%d')
            df['Sentiment_Score'] = df['Date_Str'].map(sentiment_scores)
            df = df.drop('Date_Str', axis=1)
            
            # Fill missing sentiment scores with 0 (neutral)
            df['Sentiment_Score'] = df['Sentiment_Score'].fillna(0)
            
            # Create target variable (next day's return)
            df['Target'] = df['Close'].shift(-1)
            df['Target_Return'] = df['Target'].pct_change()
            
            # Remove rows with missing values
            df = df.dropna()
            
            return df
        except Exception as e:
            st.error(f"Error integrating data: {e}")
            return pd.DataFrame()
    
    def apply_distribution_aware_normalization(self, df):
        """Apply different scalers based on data distribution"""
        try:
            df_scaled = df.copy()
            
            # Technical indicators - StandardScaler
            tech_columns = ['MA_5', 'MA_20', 'MA_50', 'MA_Ratio_5_20', 'MA_Ratio_20_50', 
                           'Volatility_5', 'Volatility_20', 'RSI', 'MACD', 'MACD_Signal', 
                           'MACD_Histogram', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position']
            
            tech_scaler = StandardScaler()
            df_scaled[tech_columns] = tech_scaler.fit_transform(df[tech_columns])
            self.scalers['technical'] = tech_scaler
            
            # VIX data - RobustScaler (handles outliers well)
            vix_columns = ['VIX_Close', 'VIX_High', 'VIX_Low']
            vix_scaler = RobustScaler()
            df_scaled[vix_columns] = vix_scaler.fit_transform(df[vix_columns])
            self.scalers['vix'] = vix_scaler
            
            # Sentiment scores - MinMaxScaler to [-1, 1]
            sentiment_scaler = MinMaxScaler(feature_range=(-1, 1))
            df_scaled[['Sentiment_Score']] = sentiment_scaler.fit_transform(df[['Sentiment_Score']])
            self.scalers['sentiment'] = sentiment_scaler
            
            # Price and volume - StandardScaler
            price_volume_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Volume_MA_5', 'Volume_Ratio']
            price_scaler = StandardScaler()
            df_scaled[price_volume_columns] = price_scaler.fit_transform(df[price_volume_columns])
            self.scalers['price_volume'] = price_scaler
            
            return df_scaled
        except Exception as e:
            st.error(f"Error in normalization: {e}")
            return df

class SourceSpecificModels:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        
    def create_market_model(self, input_dim):
        """GRU-based market data model"""
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(self.sequence_length, input_dim)),
            Dropout(0.3),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def create_sentiment_model(self, input_dim):
        """Transformer-based sentiment model"""
        inputs = Input(shape=(self.sequence_length, input_dim))
        
        # Multi-head attention
        attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
        attention_output = LayerNormalization()(attention_output + inputs)
        
        # Feed forward
        ff_output = Dense(32, activation='relu')(attention_output)
        ff_output = Dropout(0.2)(ff_output)
        ff_output = Dense(16, activation='relu')(ff_output)
        ff_output = LayerNormalization()(ff_output + attention_output)
        
        # Global average pooling and output
        pooled = GlobalAveragePooling1D()(ff_output)
        outputs = Dense(1, activation='linear')(pooled)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def create_vix_model(self, input_dim):
        """MLP-based VIX model"""
        model = Sequential([
            Dense(32, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

class DynamicUncertaintyFusion:
    def __init__(self, lookback_window=10, temperature=1.0):
        self.lookback_window = lookback_window
        self.temperature = temperature
        
    def calculate_uncertainty(self, predictions, actuals):
        """Calculate time-varying uncertainty using recent prediction errors"""
        if len(predictions) < self.lookback_window:
            return 1.0
            
        recent_preds = predictions[-self.lookback_window:]
        recent_actuals = actuals[-self.lookback_window:]
        
        errors = np.abs(np.array(recent_preds) - np.array(recent_actuals))
        uncertainty = np.var(errors)
        
        return max(uncertainty, 1e-6)
    
    def compute_dynamic_weights(self, uncertainties):
        """Compute Bayesian softmax weights based on uncertainties"""
        if not uncertainties:
            return {}
            
        unc_array = np.array(list(uncertainties.values()))
        weights = np.exp(-unc_array**2 / self.temperature)
        weights = weights / np.sum(weights)
        
        return dict(zip(uncertainties.keys(), weights))

class ModelTrainer:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.models = {}
        self.histories = {}
        
    def prepare_sequences(self, data, features, target_col='Target'):
        """Prepare sequences for time series models"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[features].iloc[i:(i + self.sequence_length)].values)
            y.append(data[target_col].iloc[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def train_models(self, data, market_features, sentiment_features, vix_features):
        """Train all models"""
        try:
            # Prepare sequences for each source
            X_market, y = self.prepare_sequences(data, market_features)
            X_sentiment, _ = self.prepare_sequences(data, sentiment_features)
            X_vix_flat = data[vix_features].values[self.sequence_length:]
            
            # Split data
            split_idx = int(0.8 * len(X_market))
            
            X_market_train, X_market_test = X_market[:split_idx], X_market[split_idx:]
            X_sentiment_train, X_sentiment_test = X_sentiment[:split_idx], X_sentiment[split_idx:]
            X_vix_train, X_vix_test = X_vix_flat[:split_idx], X_vix_flat[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Create and train models
            source_models = SourceSpecificModels(sequence_length=self.sequence_length)
            
            # Market Model (GRU)
            with st.spinner("Training Market Model (GRU)..."):
                market_model = source_models.create_market_model(len(market_features))
                market_history = market_model.fit(
                    X_market_train, y_train,
                    epochs=30, batch_size=32, verbose=0,
                    validation_data=(X_market_test, y_test),
                    callbacks=[
                        EarlyStopping(patience=5, restore_best_weights=True),
                        ReduceLROnPlateau(patience=3, factor=0.5)
                    ]
                )
                self.models['market'] = market_model
                self.histories['market'] = market_history
            
            # Sentiment Model (Transformer)
            with st.spinner("Training Sentiment Model (Transformer)..."):
                sentiment_model = source_models.create_sentiment_model(len(sentiment_features))
                sentiment_history = sentiment_model.fit(
                    X_sentiment_train, y_train,
                    epochs=30, batch_size=32, verbose=0,
                    validation_data=(X_sentiment_test, y_test),
                    callbacks=[
                        EarlyStopping(patience=5, restore_best_weights=True),
                        ReduceLROnPlateau(patience=3, factor=0.5)
                    ]
                )
                self.models['sentiment'] = sentiment_model
                self.histories['sentiment'] = sentiment_history
            
            # VIX Model (MLP)
            with st.spinner("Training VIX Model (MLP)..."):
                vix_model = source_models.create_vix_model(len(vix_features))
                vix_history = vix_model.fit(
                    X_vix_train, y_train,
                    epochs=50, batch_size=32, verbose=0,
                    validation_data=(X_vix_test, y_test),
                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
                )
                self.models['vix'] = vix_model
                self.histories['vix'] = vix_history
            
            # Store test data
            self.test_data = {
                'X_market': X_market_test,
                'X_sentiment': X_sentiment_test, 
                'X_vix': X_vix_test,
                'y_true': y_test
            }
            
            return self.test_data
        except Exception as e:
            st.error(f"Error training models: {e}")
            return None
    
    def train_comparison_models(self, data, features):
        """Train traditional ML models for comparison"""
        try:
            # Prepare features
            X = data[features].iloc[self.sequence_length:].copy()
            y = data['Target'].iloc[self.sequence_length:].copy()
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Models to compare
            comparison_models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42)
            }
            
            # Train models
            self.comparison_models = {}
            self.comparison_predictions = {}
            
            for name, model in comparison_models.items():
                with st.spinner(f"Training {name}..."):
                    model.fit(X_train, y_train)
                    self.comparison_models[name] = model
                    self.comparison_predictions[name] = model.predict(X_test)
            
            self.comparison_y_test = y_test
            return self.comparison_y_test
        except Exception as e:
            st.error(f"Error training comparison models: {e}")
            return None

def main():
    # Main header
    st.markdown('<h1 class="main-header">📈 Nifty 50 AI Predictor with Dynamic Uncertainty Fusion</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Date selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    sequence_length = st.sidebar.slider("Sequence Length", 10, 50, 20)
    lookback_window = st.sidebar.slider("Uncertainty Lookback Window", 5, 20, 10)
    
    # Initialize classes
    data_collector = Nifty50DataCollector()
    sentiment_analyzer = NewsSentimentAnalyzer()
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer(sequence_length=sequence_length)
    
    # Feature groups
    market_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_20', 'MA_50', 
                      'MA_Ratio_5_20', 'MA_Ratio_20_50', 'Volatility_5', 'Volatility_20', 
                      'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Middle', 
                      'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position', 'Volume_MA_5', 
                      'Volume_Ratio', 'High_Low_Ratio', 'Close_Open_Ratio']
    
    sentiment_features = ['Sentiment_Score', 'Returns', 'Volatility_5', 'VIX_Close', 'BB_Position']
    vix_features = ['VIX_Close', 'VIX_High', 'VIX_Low', 'Volatility_5', 'Volatility_20', 'BB_Width', 'Sentiment_Score']
    all_features = market_features + sentiment_features + vix_features
    
    # Main content
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Collecting and processing data..."):
            # Fetch market data
            nifty_data, vix_data = data_collector.fetch_market_data(start_date, end_date)
            
            if nifty_data.empty or vix_data.empty:
                st.error("Failed to fetch market data. Please check your dates and try again.")
                return
            
            # Calculate technical indicators
            nifty_data = data_collector.calculate_technical_indicators(nifty_data)
            
            # Fetch and analyze news
            news_articles = sentiment_analyzer.get_financial_news(days_back=90)
            daily_sentiment_scores = sentiment_analyzer.create_daily_sentiment_scores(news_articles)
            
            # Integrate all data
            integrated_data = preprocessor.integrate_all_data(nifty_data, vix_data, daily_sentiment_scores)
            
            if integrated_data.empty:
                st.error("Data integration failed. Please try different date range.")
                return
            
            # Apply normalization
            normalized_data = preprocessor.apply_distribution_aware_normalization(integrated_data)
        
        # Display market data
        st.header("📊 Market Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = nifty_data['Close'].iloc[-1]
            prev_close = nifty_data['Close'].iloc[-2] if len(nifty_data) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            st.metric("Nifty 50 Current Price", f"₹{current_price:.2f}", 
                     f"{change:+.2f} ({change_pct:+.2f}%)")
        
        with col2:
            vix_current = vix_data['Close'].iloc[-1] if not vix_data.empty else 0
            st.metric("India VIX", f"{vix_current:.2f}")
        
        with col3:
            st.metric("Total Trading Days", len(nifty_data))
        
        with col4:
            avg_volume = nifty_data['Volume'].mean()
            st.metric("Average Volume", f"{avg_volume:,.0f}")
        
        # Price chart
        st.subheader("Nifty 50 Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=nifty_data.index,
            open=nifty_data['Open'],
            high=nifty_data['High'],
            low=nifty_data['Low'],
            close=nifty_data['Close'],
            name='Nifty 50'
        ))
        fig.update_layout(
            title="Nifty 50 Price Movement",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # News and Sentiment Analysis
        st.header("📰 News Sentiment Analysis")
        
        if news_articles:
            # Display recent news
            st.subheader("Recent Financial News")
            for i, article in enumerate(news_articles[:10]):
                title = article.get('title', 'No title')
                description = article.get('description', 'No description')
                sentiment, confidence = sentiment_analyzer.analyze_sentiment(f"{title} {description}")
                
                # Sentiment color coding
                sentiment_color = {
                    'positive': '#4CAF50',
                    'negative': '#F44336',
                    'neutral': '#FF9800'
                }
                
                with st.container():
                    st.markdown(f"""
                    <div class="news-card">
                        <h4>{title}</h4>
                        <p>{description}</p>
                        <p><strong>Sentiment:</strong> <span style="color: {sentiment_color[sentiment]}">{sentiment.upper()}</span> 
                        (Confidence: {confidence:.2f})</p>
                        <p><small>Source: {article.get('source', {}).get('name', 'Unknown')} | 
                        Published: {article.get('publishedAt', '')[:10]}</small></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Sentiment distribution
            st.subheader("Sentiment Distribution")
            sentiments = []
            for article in news_articles:
                title = article.get('title', '')
                description = article.get('description', '')
                sentiment, _ = sentiment_analyzer.analyze_sentiment(f"{title} {description}")
                sentiments.append(sentiment)
            
            sentiment_df = pd.DataFrame({'sentiment': sentiments})
            sentiment_counts = sentiment_df['sentiment'].value_counts()
            
            fig_sentiment = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="News Sentiment Distribution",
                color=sentiment_counts.index,
                color_discrete_map={'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#FF9800'}
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        else:
            st.warning("No news articles found for the selected period.")
        
        # Model Training
        st.header("🤖 AI Model Training")
        
        # Train source models
        test_data = trainer.train_models(normalized_data, market_features, sentiment_features, vix_features)
        
        if test_data is None:
            st.error("Model training failed. Please try again.")
            return
        
        # Train comparison models
        comparison_test = trainer.train_comparison_models(normalized_data, all_features)
        
        # Model Evaluation
        st.header("📈 Model Evaluation")
        
        # Get predictions
        market_pred = trainer.models['market'].predict(test_data['X_market']).flatten()
        sentiment_pred = trainer.models['sentiment'].predict(test_data['X_sentiment']).flatten()
        vix_pred = trainer.models['vix'].predict(test_data['X_vix']).flatten()
        y_true = test_data['y_true']
        
        # Calculate metrics for each model
        metrics_data = []
        
        def calculate_all_metrics(y_true, y_pred, model_name):
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            r2 = r2_score(y_true, y_pred)
            
            # Direction accuracy
            if len(y_true) > 1:
                direction_true = np.diff(y_true) > 0
                direction_pred = np.diff(y_pred) > 0
                direction_accuracy = np.mean(direction_true == direction_pred) * 100
            else:
                direction_accuracy = 0
            
            return {
                'Model': model_name,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape,
                'R²': r2,
                'Direction Accuracy': direction_accuracy
            }
        
        # Source models metrics
        metrics_data.append(calculate_all_metrics(y_true, market_pred, 'Market Model (GRU)'))
        metrics_data.append(calculate_all_metrics(y_true, sentiment_pred, 'Sentiment Model (Transformer)'))
        metrics_data.append(calculate_all_metrics(y_true, vix_pred, 'VIX Model (MLP)'))
        
        # Comparison models metrics
        for name, y_pred in trainer.comparison_predictions.items():
            metrics_data.append(calculate_all_metrics(trainer.comparison_y_test, y_pred, name))
        
        # Dynamic Fusion
        fusion_model = DynamicUncertaintyFusion(lookback_window=lookback_window)
        
        uncertainties = {}
        uncertainties['market'] = fusion_model.calculate_uncertainty(market_pred.tolist(), y_true.tolist())
        uncertainties['sentiment'] = fusion_model.calculate_uncertainty(sentiment_pred.tolist(), y_true.tolist())
        uncertainties['vix'] = fusion_model.calculate_uncertainty(vix_pred.tolist(), y_true.tolist())
        
        weights = fusion_model.compute_dynamic_weights(uncertainties)
        
        fused_predictions = (
            weights['market'] * market_pred +
            weights['sentiment'] * sentiment_pred +
            weights['vix'] * vix_pred
        )
        
        metrics_data.append(calculate_all_metrics(y_true, fused_predictions, 'Dynamic Fusion'))
        
        # Display metrics table
        metrics_df = pd.DataFrame(metrics_data)
        st.subheader("Model Performance Metrics")
        
        # Format metrics for display
        display_metrics = metrics_df.copy()
        display_metrics['MAE'] = display_metrics['MAE'].map('{:.4f}'.format)
        display_metrics['MSE'] = display_metrics['MSE'].map('{:.4f}'.format)
        display_metrics['RMSE'] = display_metrics['RMSE'].map('{:.4f}'.format)
        display_metrics['MAPE'] = display_metrics['MAPE'].map('{:.2f}%'.format)
        display_metrics['R²'] = display_metrics['R²'].map('{:.4f}'.format)
        display_metrics['Direction Accuracy'] = display_metrics['Direction Accuracy'].map('{:.2f}%'.format)
        
        st.dataframe(display_metrics, use_container_width=True)
        
        # Fusion Weights
        st.subheader("Dynamic Fusion Weights")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Market Model Weight", f"{weights['market']:.3f}")
        with col2:
            st.metric("Sentiment Model Weight", f"{weights['sentiment']:.3f}")
        with col3:
            st.metric("VIX Model Weight", f"{weights['vix']:.3f}")
        
        # Visualizations
        st.header("📊 Performance Visualizations")
        
        # Predictions vs Actual
        st.subheader("Predictions vs Actual Values")
        fig_comparison = go.Figure()
        
        # Add traces for each model
        fig_comparison.add_trace(go.Scatter(
            y=y_true, name='Actual', line=dict(color='black', width=3)
        ))
        fig_comparison.add_trace(go.Scatter(
            y=market_pred, name='Market Model', line=dict(color='blue', dash='dot')
        ))
        fig_comparison.add_trace(go.Scatter(
            y=sentiment_pred, name='Sentiment Model', line=dict(color='green', dash='dot')
        ))
        fig_comparison.add_trace(go.Scatter(
            y=fused_predictions, name='Dynamic Fusion', line=dict(color='red', width=2)
        ))
        
        fig_comparison.update_layout(
            title="Model Predictions Comparison",
            xaxis_title="Time Steps",
            yaxis_title="Price",
            height=400
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Feature Importance (for tree-based models)
        st.subheader("Feature Importance (Random Forest)")
        if 'Random Forest' in trainer.comparison_models:
            rf_model = trainer.comparison_models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': all_features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance - Random Forest"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Final Insights
        st.header("💡 Insights & Recommendations")
        
        best_model = metrics_df.loc[metrics_df['R²'].idxmax()]
        best_model_name = best_model['Model']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Findings")
            st.markdown(f"""
            - **Best Performing Model**: {best_model_name}
            - **Highest R² Score**: {best_model['R²']:.4f}
            - **Lowest RMSE**: {metrics_df['RMSE'].min():.4f}
            - **Fusion Improvement**: {((best_model['R²'] - metrics_df[metrics_df['Model'] == 'Market Model (GRU)']['R²'].iloc[0]) / metrics_df[metrics_df['Model'] == 'Market Model (GRU)']['R²'].iloc[0] * 100):.2f}% over base market model
            """)
        
        with col2:
            st.subheader("Recommendations")
            st.markdown("""
            - Use Dynamic Fusion for robust predictions
            - Monitor model uncertainties regularly
            - Combine technical indicators with sentiment analysis
            - Consider market regimes in decision making
            - Regular model retraining recommended
            """)

if __name__ == "__main__":
    main()