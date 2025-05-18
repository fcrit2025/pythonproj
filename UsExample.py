import streamlit as st
st.set_option('server.fileWatcherType', 'none')  # Disable file watcher
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage explicitly
import xgboost as xgb
import shap
import plotly.graph_objects as go
from prophet import Prophet
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# Set page config
st.set_page_config(page_title="US Stock Predictor", layout="wide")

# Custom US business day calendar
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# Load FinBERT sentiment analysis model
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception as e:
    st.error(f"Error loading sentiment model: {str(e)}")
    sentiment_pipeline = None

# List of major US stocks to analyze
US_STOCKS = {
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "META": "Meta (Facebook)",
    "AAPL": "Apple",
    "GOOGL": "Alphabet (Google)",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "WMT": "Walmart"
}

# Fetch stock data with error handling
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker, start, end):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end)
        if not data.empty:
            data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
            data.index = data.index.tz_localize(None)
            return data
        else:
            st.warning(f"No data found for {ticker}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

# Fetch stock info with error handling
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        def format_value(value, format_str):
            if value == "N/A" or value is None:
                return "N/A"
            return format_str.format(value)
        
        return {
            "Market Cap": format_value(info.get("marketCap"), "${:,}"),
            "P/E Ratio": format_value(info.get("trailingPE"), "{}"),
            "ROE": format_value(info.get("returnOnEquity"), "{:.2f}%"),
            "Current Price": format_value(info.get("currentPrice"), "${:.2f}"),
            "Book Value": format_value(info.get("bookValue"), "${:.2f}"),
            "52 Week High": format_value(info.get("fiftyTwoWeekHigh"), "${:.2f}"),
            "52 Week Low": format_value(info.get("fiftyTwoWeekLow"), "${:.2f}"),
            "Dividend Yield": format_value(info.get("dividendYield"), "{:.2f}%"),
            "Volume": format_value(info.get("volume"), "{:,}"),
        }
    except Exception as e:
        st.error(f"Error fetching info for {ticker}: {str(e)}")
        return {}

# News API with error handling
NEWS_API_KEY = "563215a35c1a47968f46271e04083ea3"  # Replace with your actual key
NEWS_API_URL = "https://newsapi.org/v2/everything"

def get_news(stock_symbol):
    try:
        query = US_STOCKS.get(stock_symbol, stock_symbol)
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 5
        }
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            st.error(f"News API error: {response.json().get('message', 'Unknown error')}")
            return []
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

# Sentiment analysis with error handling
def analyze_sentiment(text):
    if not text or sentiment_pipeline is None:
        return "neutral", 0.0
    try:
        result = sentiment_pipeline(text[:512])[0]
        return result['label'], result['score']
    except Exception as e:
        st.error(f"Sentiment analysis error: {str(e)}")
        return "neutral", 0.0

# Feature engineering
def create_advanced_features(df):
    try:
        df = df.copy()
        df['Returns'] = df['Close'].pct_change()
        df['5D_MA'] = df['Close'].rolling(5).mean()
        df['20D_MA'] = df['Close'].rolling(20).mean()
        df['MA_Ratio'] = df['5D_MA'] / df['20D_MA']
        df['5D_Volatility'] = df['Returns'].rolling(5).std()
        df['Volume_MA5'] = df['Volume'].rolling(5).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
        return df.dropna()
    except Exception as e:
        st.error(f"Feature engineering error: {str(e)}")
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
        st.error(f"Prophet forecast error: {str(e)}")
        return pd.DataFrame()

# Adjust predictions for market closures
def adjust_predictions_for_market_closures(predictions_df):
    try:
        # Generate business days in the prediction range
        business_days = pd.date_range(
            start=predictions_df.index.min(),
            end=predictions_df.index.max(),
            freq=us_bd
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
        
        # Calculate daily changes
        predictions_df['Daily Change (%)'] = predictions_df['adjusted_prediction'].pct_change().fillna(0) * 100
        
        return predictions_df[['adjusted_prediction', 'Daily Change (%)']].rename(
            columns={'adjusted_prediction': 'Predicted Price'}
        )
    except Exception as e:
        st.error(f"Market closure adjustment error: {str(e)}")
        return predictions_df

# Hybrid XGBoost-GRU-Prophet Model
def create_hybrid_model(df_stock, sentiment_features):
    try:
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
        
        # Calculate model weights
        xgb_pred = xgb_model.predict(X_test)
        gru_pred = gru_model.predict(X_3d[len(X_train):]).flatten()
        
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        gru_mae = mean_absolute_error(y_test, gru_pred)
        
        total = (1/xgb_mae + 1/gru_mae)
        xgb_weight = (1/xgb_mae)/total
        gru_weight = (1/gru_mae)/total
        
        # Final predictions
        final_pred = (xgb_weight * xgb_pred) + (gru_weight * gru_pred)
        df_stock.loc[y_test.index, 'Predicted'] = final_pred
        
        # Calculate accuracy
        mae = mean_absolute_error(y_test, final_pred)
        accuracy = max(0, 100 - (mae * 100))
        
        return df_stock, {'xgb': xgb_model, 'gru': gru_model}, scaler, features, accuracy, {'xgb_weight': xgb_weight, 'gru_weight': gru_weight}
    except Exception as e:
        st.error(f"Hybrid model training error: {str(e)}")
        return None, None, None, None, 0, None

# Generate predictions with realistic bounds
def hybrid_predict_prices(models, scaler, last_known_data, features, days=10, weights=None):
    try:
        if weights is None:
            weights = {'xgb_weight': 0.6, 'gru_weight': 0.4}
            
        # Get Prophet forecast
        prophet_forecast_df = prophet_forecast(last_known_data, days=days)
        future_dates = prophet_forecast_df.index
        
        # Initialize results
        future_prices = pd.DataFrame(index=future_dates, 
                                   columns=['Predicted Price', 'Daily Change (%)'])
        
        current_data = last_known_data.copy()
        last_close = current_data['Close'].iloc[-1]
        recent_volatility = current_data['5D_Volatility'].iloc[-1]
        
        for i, date in enumerate(future_dates):
            # 1. Get individual model predictions
            xgb_input = current_data[features].iloc[-1:].copy()
            xgb_pred = models['xgb'].predict(xgb_input)[0]
            
            input_scaled = scaler.transform(xgb_input)
            input_3d = input_scaled.reshape(1, 1, input_scaled.shape[1])
            gru_pred = models['gru'].predict(input_3d)[0][0]
            
            prophet_pred = (prophet_forecast_df['yhat'].iloc[i] - last_close) / last_close
            
            # 2. Create weighted ensemble prediction
            combined_pred = (
                weights['xgb_weight'] * xgb_pred + 
                weights['gru_weight'] * gru_pred
            )
            
            # Blend with Prophet
            final_pred = 0.8 * combined_pred + 0.2 * prophet_pred
            
            # 3. Apply realistic noise
            max_daily_change = 0.05  # 5% max daily change
            noise = np.random.normal(0, min(recent_volatility, 0.03))
            adj_pred = np.clip(final_pred + noise, -max_daily_change, max_daily_change)
            
            # 4. Calculate new price with bounds
            new_close = last_close * (1 + adj_pred)
            
            # Price bounds
            price_bound_low = last_close * 0.90
            price_bound_high = last_close * 1.10
            new_close = np.clip(new_close, price_bound_low, price_bound_high)
            
            future_prices.loc[date, 'Predicted Price'] = new_close
            last_close = new_close
            
            # 5. Update simulated data
            new_row = {
                'Open': new_close * (0.998 + np.random.uniform(-0.002, 0.002)),
                'High': new_close * (1 + np.random.uniform(0, 0.01)),
                'Low': new_close * (1 - np.random.uniform(0, 0.01)),
                'Close': new_close,
                'Volume': current_data['Volume'].iloc[-1] * (0.95 + np.random.uniform(-0.05, 0.05)),
                'Sentiment': current_data['Sentiment'].iloc[-1] * (0.95 + np.random.uniform(-0.1, 0.1)),
                '5D_MA': current_data['5D_MA'].iloc[-1] * (1 + np.random.uniform(-0.005, 0.005)),
                '20D_MA': current_data['20D_MA'].iloc[-1] * (1 + np.random.uniform(-0.002, 0.002)),
                'MA_Ratio': current_data['MA_Ratio'].iloc[-1] * (1 + np.random.uniform(-0.01, 0.01)),
                '5D_Volatility': recent_volatility * (1 + np.random.uniform(-0.1, 0.1)),
                'Volume_MA5': current_data['Volume_MA5'].iloc[-1] * (1 + np.random.uniform(-0.05, 0.05)),
                'Volume_Ratio': current_data['Volume_Ratio'].iloc[-1] * (1 + np.random.uniform(-0.1, 0.1))
            }
            current_data = pd.concat([current_data.iloc[1:], pd.DataFrame(new_row, index=[date])])
        
        # Calculate daily changes
        future_prices['Daily Change (%)'] = future_prices['Predicted Price'].pct_change().fillna(0) * 100
        
        # Adjust for market closures
        future_prices = adjust_predictions_for_market_closures(future_prices)
        
        return future_prices
    
    except Exception as e:
        st.error(f"Prediction generation error: {str(e)}")
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
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False
    )
    return fig

# Generate investment recommendation
def generate_recommendation(predicted_prices, current_price, accuracy, avg_sentiment):
    try:
        avg_prediction = predicted_prices['Predicted Price'].mean()
        price_change = ((avg_prediction - current_price) / current_price) * 100
        
        sentiment_factor = 1 + (avg_sentiment * 0.5)
        adjusted_change = price_change * sentiment_factor
        
        if adjusted_change > 7 and accuracy > 72:
            return "STRONG BUY", "High confidence in significant price increase"
        elif adjusted_change > 3 and accuracy > 65:
            return "BUY", "Good confidence in moderate price increase"
        elif adjusted_change > 0 and accuracy > 60:
            return "HOLD (Positive)", "Potential for slight growth"
        elif adjusted_change < -7 and accuracy > 72:
            return "STRONG SELL", "High confidence in significant price drop"
        elif adjusted_change < -3 and accuracy > 65:
            return "SELL", "Good confidence in moderate price drop"
        elif adjusted_change < 0 and accuracy > 60:
            return "HOLD (Caution)", "Potential for slight decline"
        else:
            return "HOLD", "Unclear direction - consider other factors"
    except Exception as e:
        return "HOLD", f"Could not generate recommendation: {str(e)}"

# Streamlit UI
st.title("US Stock Market Predictor")
st.sidebar.header("Stock Selection")

# User inputs
selected_stock = st.sidebar.selectbox("Select Stock", list(US_STOCKS.keys()), format_func=lambda x: f"{x} - {US_STOCKS[x]}")
start_date = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 10)
chart_type = st.sidebar.radio("Chart Type", ["Candlestick Chart", "Line Chart"])

if st.sidebar.button("Analyze"):
    with st.spinner("Analyzing stock data..."):
        # Fetch stock data
        ticker = selected_stock
        df_stock = get_stock_data(ticker, start_date, end_date)
        
        if not df_stock.empty:
            df_stock = df_stock.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
            
            # Display stock info
            st.subheader(f"Stock Information for {ticker} - {US_STOCKS[ticker]}")
            stock_info = get_stock_info(ticker)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Market Cap:** {stock_info['Market Cap']}")
                st.write(f"**P/E Ratio:** {stock_info['P/E Ratio']}")
                st.write(f"**Current Price:** {stock_info['Current Price']}")
                st.write(f"**52 Week High:** {stock_info['52 Week High']}")
            with col2:
                st.write(f"**Book Value:** {stock_info['Book Value']}")
                st.write(f"**ROE:** {stock_info['ROE']}")
                st.write(f"**Dividend Yield:** {stock_info['Dividend Yield']}")
                st.write(f"**52 Week Low:** {stock_info['52 Week Low']}")
            
            # Display historical data
            st.subheader("Historical Price Data")
            st.dataframe(df_stock[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index(ascending=False).style.format({
                'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}', 
                'Close': '{:.2f}', 'Volume': '{:,}'
            }))
            
            # Display chart
            st.subheader(f"{chart_type} for {ticker}")
            if chart_type == "Candlestick Chart":
                st.plotly_chart(create_candlestick_chart(df_stock))
            else:
                st.line_chart(df_stock["Close"])
            
            # News and sentiment analysis
            st.subheader(f"Latest News for {US_STOCKS[ticker]}")
            news_articles = get_news(ticker)
            daily_sentiment = {}
            
            if not news_articles:
                st.warning("No recent news articles found for this stock.")
            else:
                for article in news_articles:
                    title = article.get('title', '')
                    description = article.get('description', '')
                    text = f"{title} {description}".strip()
                    sentiment, confidence_score = analyze_sentiment(text)
                    date = article.get("publishedAt", "")[0:10]
                    
                    st.write(f"**{title}**")
                    st.write(description)
                    st.write(f"Sentiment: {sentiment} (Confidence: {confidence_score:.2f})")
                    st.write(f"[Read more]({article['url']})")
                    st.write("---")
                    
                    if date in daily_sentiment:
                        daily_sentiment[date].append((sentiment, confidence_score))
                    else:
                        daily_sentiment[date] = [(sentiment, confidence_score)]
            
            # Train hybrid model
            df_stock, models, scaler, features, accuracy, weights = create_hybrid_model(
                df_stock, 
                daily_sentiment if daily_sentiment else {}
            )
            
            if models is not None:
                # AI Analysis Conclusion
                st.subheader("AI Analysis Conclusion")
                current_price = df_stock['Close'].iloc[-1]
                avg_sentiment = df_stock['Sentiment'].mean() if 'Sentiment' in df_stock else 0
                
                explanation = f"""
                Our analysis of {ticker} reveals:
                
                1. **Price Movement**: Currently at {stock_info['Current Price']}. Model confidence: {accuracy:.1f}%
                2. **Technical Indicators**:
                   - Trading {'above' if df_stock['MA_Ratio'].iloc[-1] > 1 else 'below'} 20-day MA
                   - Volatility: {'high' if df_stock['5D_Volatility'].iloc[-1] > 0.02 else 'moderate' if df_stock['5D_Volatility'].iloc[-1] > 0.01 else 'low'}
                   - Volume: {'increasing' if df_stock['Volume_Ratio'].iloc[-1] > 1 else 'decreasing' if df_stock['Volume_Ratio'].iloc[-1] < 1 else 'stable'}
                """
                st.write(explanation)
                
                # Generate forecast
                st.subheader(f"{forecast_days}-Day Price Forecast")
                last_data = df_stock.iloc[-30:]
                future_prices = hybrid_predict_prices(
                    models, scaler, last_data, features, 
                    days=forecast_days, weights=weights
                )
                
                if not future_prices.empty:
                    # Format forecast table
                    def format_forecast_table(df):
                        return df[['Predicted Price', 'Daily Change (%)']].rename(columns={
                            'Predicted Price': 'Price ($)',
                            'Daily Change (%)': 'Daily Change (%)'
                        }).style.format({
                            'Price ($)': '${:,.2f}',
                            'Daily Change (%)': '{:+.2f}%'
                        }).applymap(
                            lambda x: 'color: #4CAF50' if x > 0 else 'color: #F44336',
                            subset=['Daily Change (%)']
                        )
                    
                    st.dataframe(
                        format_forecast_table(future_prices),
                        use_container_width=True
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
                        "STRONG SELL": "darkred"
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
                        hovertemplate='$%{y:.2f}<extra>%{x|%b %d, %Y}</extra>'
                    ))
                    fig.add_trace(go.Scatter(
                        x=future_data.index,
                        y=future_data['Price'],
                        mode='lines+markers',
                        name='AI Forecast',
                        line=dict(color='#FF7F0E', width=2, dash='dot'),
                        marker=dict(size=8, color='#FF7F0E'),
                        hovertemplate='$%{y:.2f}<extra>%{x|%b %d, %Y}</extra>'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[historical_data.index[-1]],
                        y=[current_price],
                        mode='markers',
                        name='Current Price',
                        marker=dict(size=12, color='#DC3912'),
                        hovertemplate='$%{y:.2f}<extra>Current Price</extra>'
                    ))
                    fig.add_vline(
                        x=historical_data.index[-1],
                        line_width=1,
                        line_dash="dash",
                        line_color="grey"
                    )
                    fig.update_layout(
                        title=f"{ticker} Price Trend (Historical vs Forecast)",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode="x unified",
                        plot_bgcolor='rgba(240,240,240,0.8)',
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)