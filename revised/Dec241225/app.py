# finapp_fingpt.py
import streamlit as st
import requests
import yfinance as yf
import re
from transformers import pipeline
import torch
import os
from dotenv import load_dotenv  # For environment variables

# --- Load environment variables ---
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "563215a35c1a47968f46271e04083ea3")
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Indian stock dictionary (must be defined before use)
STOCKS = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services", 
    "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank"
}

# --- Model Setup (Using a Lightweight, Compatible Model) ---
@st.cache_resource
def load_fingpt_pipeline():
    """Load a lightweight, compatible financial model for analysis"""
    try:
        # SWITCH TO A SMALLER, MORE RELIABLE MODEL
        # Option 1: A small but capable general financial model (Recommended)
        model_name = "microsoft/phi-2"  # 2.7B params, good reasoning
        
        # Option 2: A distilled financial sentiment model (Fastest)
        # model_name = "ahmedrachid/FinancialBERT-Sentiment-Analysis"
        
        # Option 3: Another capable small model
        # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        print(f"Loading model: {model_name}")
        
        # Load tokenizer and model
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # For phi-2, add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Create text generation pipeline
        fingpt_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            temperature=0.4,
            do_sample=True
        )
        
        print("✅ Model loaded successfully!")
        return fingpt_pipeline
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Fallback: Use a simple sentiment pipeline if primary model fails
        try:
            print("Attempting to load fallback sentiment model...")
            return pipeline("text-classification", model="ProsusAI/finbert")
        except:
            return None

# --- News Functions (Reused from your code) ---
def get_news(stock_symbol, page_size=10):
    stock_name_mapping = STOCKS
    query = stock_name_mapping.get(stock_symbol, stock_symbol)
    
    params = {
        "q": query, 
        "apiKey": NEWS_API_KEY, 
        "language": "en", 
        "sortBy": "publishedAt",
        "pageSize": page_size
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            st.error(f"News API Error {response.status_code}: {response.text}")
            return []
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
        return []

def filter_relevant_news(news_articles, search_term):
    filtered_articles = []
    for article in news_articles:
        title = article.get('title', '')
        description = article.get('description', '')
        content = f"{title} {description}"
        if content and re.search(search_term, content, re.IGNORECASE):
            filtered_articles.append(article)
    return filtered_articles

# --- FinGPT Analysis Function (Updated) ---
def analyze_with_fingpt(ticker, news_articles, stock_data):
    """Use the loaded model to analyze news and stock data"""
    
    if not fingpt_pipeline:
        return "Financial model not available.", "Error"
    
    # Prepare news summary (limit tokens)
    news_summary = "\n".join([
        f"{i+1}. {article['title'][:80]}"
        for i, article in enumerate(news_articles[:3])  # Only 3 articles
    ])
    
    # Create optimized prompt
    prompt = f"""Analyze stock sentiment based on news:

Stock: {ticker} ({STOCKS.get(ticker, ticker)})
Price: ₹{stock_data.get('current_price', 'N/A'):.2f}

Recent News:
{news_summary}

Provide a concise analysis with:
1. Overall sentiment (Positive/Neutral/Negative)
2. Recommendation: BUY, HOLD, or SELL
3. One-line reason

Analysis:"""
    
    try:
        # Generate analysis
        response = fingpt_pipeline(
            prompt,
            max_new_tokens=100,  # Reduced for smaller models
            temperature=0.3,
            do_sample=True,
            top_p=0.9
        )
        
        generated_text = response[0]['generated_text']
        
        # Extract just the new analysis part (remove prompt)
        if "Analysis:" in generated_text:
            analysis = generated_text.split("Analysis:")[-1].strip()
        else:
            analysis = generated_text[len(prompt):].strip()
        
        return analysis[:500], "Success"  # Limit output length
        
    except Exception as e:
        return f"Analysis failed: {str(e)[:100]}", "Error"

# --- Main Analysis Function ---
def make_decision_and_display(ticker):
    st.subheader(f"📊 Analyzing {STOCKS.get(ticker, ticker)} ({ticker})")
    
    # Get stock data
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        hist_data = stock.history(period="1d")
        if not hist_data.empty:
            current_price = hist_data['Close'].iloc[-1]
            st.metric("Current Price (INR)", f"₹{current_price:.2f}")
            
            # Store stock data for FinGPT
            stock_data = {
                'current_price': current_price,
                'ticker': ticker
            }
        else:
            st.warning("Could not fetch live price data.")
            stock_data = {'current_price': 0, 'ticker': ticker}
            
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        stock_data = {'current_price': 0, 'ticker': ticker}
    
    # Get and filter news
    with st.spinner("Fetching recent news..."):
        raw_news = get_news(ticker, page_size=15)
        search_term = STOCKS.get(ticker, ticker).split()[0]
        relevant_news = filter_relevant_news(raw_news, search_term)
        news_to_display = relevant_news[:10]
    
    if not news_to_display:
        st.info("No relevant news articles found for analysis.")
        return
    
    # Display news articles first
    st.markdown("---")
    st.subheader(f"📰 Top {len(news_to_display)} News Articles")
    
    news_texts = []
    for i, article in enumerate(news_to_display):
        with st.expander(f"{i+1}. {article['title'][:80]}..."):
            st.markdown(f"**Source:** {article['source']['name']}")
            st.markdown(f"**Published:** {article['publishedAt'][:10]}")
            st.markdown(f"**Description:** {article.get('description', 'No description')}")
            st.markdown(f"[Read full article]({article['url']})")
        
        # Collect text for FinGPT analysis
        news_texts.append(f"{article['title']}. {article.get('description', '')}")
    
    # FinGPT Analysis Section
    st.markdown("---")
    st.subheader("🤖 FinGPT AI Analysis")
    
    with st.spinner("Generating AI analysis with FinGPT..."):
        analysis_result, status = analyze_with_fingpt(
            ticker, 
            news_to_display, 
            stock_data
        )
    
    if status == "Success":
        # Display FinGPT analysis
        st.markdown("### FinGPT Analysis Result")
        st.write(analysis_result.split("Analysis:")[-1].strip())
        
        # Extract recommendation if possible
        analysis_lower = analysis_result.lower()
        if "buy" in analysis_lower:
            st.success("**FinGPT Recommendation: BUY**")
        elif "sell" in analysis_lower:
            st.error("**FinGPT Recommendation: SELL**")
        elif "hold" in analysis_lower:
            st.warning("**FinGPT Recommendation: HOLD**")
    else:
        st.error("Could not generate FinGPT analysis")

# --- Streamlit UI ---
st.set_page_config(page_title="FinGPT Stock Analyst", page_icon="📈", layout="wide")

st.title("🤖 FinGPT-Powered Indian Stock News & Sentiment Analyzer")
st.markdown("---")

# Initialize FinGPT model
with st.spinner("Loading FinGPT model (this may take a minute)..."):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    fingpt_pipeline = load_fingpt_pipeline()

if fingpt_pipeline:
    st.success("✅ FinGPT model loaded successfully!")
else:
    st.warning("⚠️ FinGPT model failed to load. News analysis will proceed without AI.")

# Stock selection dropdown[citation:4][citation:10]
col1, col2 = st.columns([1, 2])
with col1:
    selected_stock = st.selectbox(
        "Choose a stock to analyze:",
        options=list(STOCKS.keys()),
        format_func=lambda x: f"{x} - {STOCKS[x]}",
        help="Select an Indian stock for news and sentiment analysis"
    )

st.markdown("---")

if st.button("🚀 Analyze Stock", type="primary", use_container_width=True):
    make_decision_and_display(selected_stock)

# Footer
st.markdown("---")
st.markdown("""
**Note:** 
- This tool uses FinGPT open-source financial LLM for analysis
- Stock data from Yahoo Finance (add .NS for NSE symbols)
- News from NewsAPI (requires valid API key)
- Analysis is for informational purposes only
""")