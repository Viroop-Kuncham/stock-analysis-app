import os
import streamlit as st
import requests
import yfinance as yf
from yahooquery import search
from datetime import datetime, timedelta
from transformers import pipeline
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression

# --- CONFIGURATION ---
NEWS_API_KEY = "Your News API Key Here"
SENTIMENT_MODEL = pipeline("sentiment-analysis")
MAX_NEWS_ARTICLES = 15

# --- TICKER RESOLUTION ---
def resolve_ticker(user_input):
    user_input = user_input.strip()
    if user_input.isupper() and len(user_input) <= 5:
        try:
            stock = yf.Ticker(user_input)
            info = stock.info
            if 'shortName' in info:
                return user_input, info['shortName']
        except Exception:
            pass
    try:
        results = search(user_input)
        if not results:
            return None, None
        for res in results:
            if 'longname' in res and res['longname']:
                return res['symbol'], res['longname']
        return results[0]['symbol'], results[0]['symbol']
    except Exception:
        return None, None

# --- ENHANCED NEWS API FETCH ---
def get_news(company_name, ticker):
    # Add terms that broaden search: CEO, products, board, etc.
    keywords = [
        company_name,
        ticker,
        f"{company_name} CEO",
        f"{company_name} board",
        f"{company_name} earnings",
        f"{company_name} product",
        f"{ticker} stock",
        f"{ticker} outlook",
        f"{company_name} future",
    ]
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    query = " OR ".join(keywords)
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q=({query})&from={from_date}&language=en&sortBy=relevancy&pageSize={MAX_NEWS_ARTICLES}&apiKey={NEWS_API_KEY}"
    )
    res = requests.get(url)
    if res.status_code != 200:
        st.error("Error fetching news data.")
        return []
    return res.json().get('articles', [])

# --- SENTIMENT ANALYSIS ---
def analyze_sentiments(articles):
    sentiments = []
    for art in articles:
        text = art.get('description') or art.get('title') or ""
        if not text.strip():
            sentiments.append(("NEUTRAL", 0.5))
            continue
        result = SENTIMENT_MODEL(text[:512])[0]
        label = result['label'].upper()
        score = result['score']
        sentiments.append((label, score))
    return sentiments

# --- SENTIMENT PLOT ---
def plot_sentiment_chart(sentiments):
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for label, _ in sentiments:
        if label not in counts:
            counts["NEUTRAL"] += 1
        else:
            counts[label] += 1
    fig = go.Figure(
        go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=['green', 'red', 'gray'],
            text=list(counts.values()),
            textposition='auto'
        )
    )
    fig.update_layout(title="Sentiment Distribution", yaxis_title="Article Count")
    st.plotly_chart(fig, use_container_width=True)

# --- STOCK DATA ---
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        info = stock.info
        return info, hist
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None, None

def display_stock_chart(hist, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines+markers', name='Close Price'))
    fig.update_layout(
        title=f"{ticker} Stock Price (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- 30-DAY SENTIMENT AND PRICE ---
def get_30_day_sentiment(ticker):
    scores = []
    for i in range(30):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={ticker}&from={date}&to={date}&language=en&sortBy=relevancy&pageSize=3&apiKey={NEWS_API_KEY}"
        )
        try:
            response = requests.get(url)
            if response.status_code != 200:
                scores.append(0.0)
                continue
            articles = response.json().get("articles", [])
            day_scores = []
            for article in articles:
                text = article.get("description") or article.get("title") or ""
                if text.strip():
                    result = SENTIMENT_MODEL(text[:512])[0]
                    label = result["label"]
                    score = result["score"] if label == "POSITIVE" else -result["score"]
                    day_scores.append(score)
            avg_score = np.mean(day_scores) if day_scores else 0.0
            scores.append(avg_score)
        except Exception:
            scores.append(0.0)
    return list(reversed(scores))

def get_30_day_price_change(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="35d")
    hist = hist["Close"].dropna()
    pct_changes = hist.pct_change().dropna()[-30:]
    return pct_changes.tolist()

# --- LINEAR MODEL ---
def compute_sentiment_sensitivity(ticker):
    sentiment_scores = get_30_day_sentiment(ticker)
    price_changes = get_30_day_price_change(ticker)
    if len(sentiment_scores) != 30 or len(price_changes) != 30:
        raise ValueError("Data mismatch.")
    X = np.array(sentiment_scores).reshape(-1, 1)
    y = np.array(price_changes)
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0]

def estimate_price(current_price, sentiments, sensitivity):
    sentiment_score = 0
    total = 0
    for label, score in sentiments:
        if label == "POSITIVE":
            sentiment_score += score
        elif label == "NEGATIVE":
            sentiment_score -= score
        total += 1
    if total == 0:
        return current_price, 0.0
    avg_score = sentiment_score / total
    estimated_pct_change = avg_score * sensitivity
    estimated_price = current_price * (1 + estimated_pct_change)
    return round(estimated_price, 2), round(estimated_pct_change * 100, 2)

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Stock Sentiment Tool", layout="wide")
st.title("📈 AI Market Research & Sentiment Analysis Tool")

user_input = st.text_input("Enter company name or stock ticker symbol:")

if user_input:
    ticker, company_name = resolve_ticker(user_input)
    if not ticker:
        st.error("No matching ticker found.")
        st.stop()

    st.header(f"{company_name} ({ticker})")

    info, hist = get_stock_data(ticker)
    if info is None:
        st.stop()
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

        # --- MODIFIED STOCK METRICS DISPLAY ---
    col1.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
    col2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
    mcap = info.get('marketCap', None)
    mcap_str = f"${mcap/1e9:.2f}B" if mcap else "N/A"
    col3.metric("Market Cap", mcap_str)
    col4.metric("52-Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
    col5.metric("52-Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
    prev_close = info.get('previousClose', None)
    col6.metric("Previous Close", f"${prev_close}" if prev_close else "N/A")




    st.markdown(f"**Industry:** {info.get('industry', 'N/A')}  |  **Sector:** {info.get('sector', 'N/A')}")
    if info.get("website"):
        st.markdown(f"**Website:** [{info.get('website')}]({info.get('website')})")

    with st.expander("📄 Business Summary"):
        st.write(info.get('longBusinessSummary', 'No summary available.'))

    if hist is not None and not hist.empty:
        display_stock_chart(hist, ticker)
    else:
        st.warning("No price history available.")

    st.subheader("🗞️ News & Sentiment (Past 7 Days)")
    articles = get_news(f"{company_name} OR {ticker}")
    if not articles:
        st.warning("No recent news found.")
    else:
        sentiments = analyze_sentiments(articles)
        for idx, article in enumerate(articles):
            st.markdown(f"**[{article.get('title', 'No title')}]({article.get('url', '#')})**")
            st.write(article.get("description", ""))
            label, score = sentiments[idx]
            st.markdown(f"*Sentiment:* **{label}**  |  *Confidence:* {score:.2f}")
            st.markdown("---")

        plot_sentiment_chart(sentiments)

        current_price = info.get("currentPrice", None)
        if current_price:
            try:
                sensitivity = compute_sentiment_sensitivity(ticker)
                est_price, pct_change = estimate_price(current_price, sentiments, sensitivity)
                sign = "+" if pct_change >= 0 else "-"
                st.subheader("💡 Estimated Price Impact Based on Sentiment")
                st.markdown(
                    f"**Current:** ${current_price} → Estimated: ${est_price} ({sign}{abs(pct_change)}%)"
                )
            except Exception as e:
                st.warning(f"Could not estimate price: {e}")
        else:
            st.warning("Price data unavailable for estimation.")

    st.markdown(
        """
        <hr style="margin-top: 2em;">
        <div style='font-size: 0.8em; color: gray;'>
        ⚠️ This tool uses public news and statistical modeling to suggest price influence.
        It is not financial advice. Always do your own research before making investment decisions.
        </div>
        """,
        unsafe_allow_html=True
    )
