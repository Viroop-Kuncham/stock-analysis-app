# AI Market Research & Sentiment Analysis Tool

A Streamlit web application for advanced stock analytics, combining real-time price data, news sentiment analysis, and AI-driven price impact estimation.

---

## 🚀 Features

- **Ticker Resolution:** Enter a company name or ticker symbol to auto-resolve the correct stock.
- **Stock Metrics:** View current price, P/E ratio, market cap, 52-week high/low, and more.
- **Interactive Price Chart:** Visualize the last 30 days of price history.
- **News Aggregation:** Fetches recent news from NewsAPI for the selected stock.
- **AI Sentiment Analysis:** Uses a transformer model to analyze news sentiment.
- **Sentiment Distribution Chart:** Visualizes sentiment breakdown of recent news.
- **Price Impact Estimation:** Estimates potential price movement based on sentiment and historical data.
- **Business Summary:** Displays company overview and key information.
- **Industry & Sector Info:** Quick reference for the company’s sector and industry.

---

## Requirements

- Python 3.8 or higher
- [Streamlit](https://streamlit.io/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [yahooquery](https://github.com/dpguthrie/yahooquery)
- [transformers](https://huggingface.co/transformers/)
- [plotly](https://plotly.com/python/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [requests](https://docs.python-requests.org/)
- [NewsAPI key](https://newsapi.org/)

---

##  Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/stock-analysis-app.git
   cd stock-analysis-app
   ```

2. **(Optional but recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your NewsAPI key:**
   - Open `Stock_Analysis_App.py`
   - Replace:
     ```python
     NEWS_API_KEY = "Your News API Key Here"
     ```
     with your actual NewsAPI key.

---

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run Stock_Analysis_App.py
   ```
2. Open the provided local URL in your browser (usually http://localhost:8501).
3. Enter a company name or ticker symbol to begin analysis.

---

## Example

![App Screenshot](screenshot.png) <!-- Add a screenshot of your app here -->

---

## Configuration

- **API Keys:**  
  - You must provide your own [NewsAPI](https://newsapi.org/) key.
- **Model:**  
  - The app uses HuggingFace’s `transformers` pipeline for sentiment analysis. The default model is `distilbert-base-uncased-finetuned-sst-2-english`.

---

## File Structure

```
Stock_Analysis_App.py
requirements.txt
README.md
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [yahooquery](https://github.com/dpguthrie/yahooquery)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Plotly](https://plotly.com/python/)
- [NewsAPI](https://newsapi.org/)

---

## Contact

For questions or suggestions, please contact [kunchamviroop@gmail.com](mailto:kunchamviroop@gmail.com).
