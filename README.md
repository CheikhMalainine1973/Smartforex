# Smart Forex Prediction System

A comprehensive forex market prediction system that combines technical analysis and sentiment analysis to generate trading signals.

## Features

- **Technical Analysis**: Uses LSTM model and various technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands)
- **Sentiment Analysis**: Analyzes news articles and social media using FinBERT
- **Fusion Model**: Combines technical and sentiment analysis using XGBoost
- **Automated Pipeline**: End-to-end data collection, analysis, and prediction
- **Visualization**: Plots and charts for analysis results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CheikhMalainine1973/smartforex.git
cd smartforex
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys:
   - Create a `.env` file in the project root
   - Add your API keys for News API and Twitter API:
```
NEWS_API_KEY=your_news_api_key
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
```

## Usage

1. Run the main analysis pipeline:
```bash
python scripts/main.py
```

2. For specific currency pairs:
```python
from scripts.main import SmartForex

smart_forex = SmartForex()
smart_forex.run_pipeline(symbol="EURUSD=X", days_back=30)
```

## Project Structure

```
smartforex/
├── data/               # Data storage
├── models/            # Trained models
├── results/           # Analysis results
├── scripts/           # Python modules
│   ├── data_collector.py
│   ├── technical_analysis.py
│   ├── sentiment_analysis.py
│   ├── fusion_model.py
│   └── main.py
├── requirements.txt   # Dependencies
└── README.md         # Documentation
```

## Components

### 1. Data Collection
- Fetches forex data using yfinance
- Collects news articles using News API
- Gathers tweets using Twitter API

### 2. Technical Analysis
- Calculates technical indicators
- Trains LSTM model for price prediction
- Analyzes market trends

### 3. Sentiment Analysis
- Uses FinBERT for financial sentiment analysis
- Analyzes news and social media sentiment
- Generates market sentiment scores

### 4. Fusion Model
- Combines technical and sentiment features
- Uses XGBoost for final prediction
- Generates trading signals with confidence scores

## Output

The system generates:
- Technical analysis charts
- Sentiment analysis results
- Trading signals with confidence levels
- JSON files with detailed analysis results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Disclaimer

This system is for educational purposes only. Trading forex involves significant risk of loss and is not suitable for all investors. Past performance is not indicative of future results.
