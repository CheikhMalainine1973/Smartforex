import os
import json
from datetime import datetime, timedelta
import pandas as pd
from data_collector import DataCollector
from technical_analysis import TechnicalAnalyzer
from sentiment_analysis import SentimentAnalyzer
from fusion_model import FusionModel

class SmartForex:
    def __init__(self):
        # Initialize components
        self.data_collector = DataCollector()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.fusion_model = FusionModel()
        
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)

    def collect_data(self, symbol: str, days_back: int = 30):
        """
        Collect all necessary data
        """
        print("Collecting data...")
        
        # Get forex data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        forex_data = self.data_collector.get_forex_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if forex_data is not None:
            self.data_collector.save_data(forex_data, f'{symbol.lower()}_data.csv')
        
        # Get news data
        news_data = self.data_collector.get_news_data(
            query="forex OR currency market",
            from_date=start_date.strftime('%Y-%m-%d'),
            to_date=end_date.strftime('%Y-%m-%d')
        )
        
        if news_data is not None:
            self.data_collector.save_data(news_data, 'forex_news.json')
        
        # Get tweets
        tweets = self.data_collector.get_twitter_sentiment(
            query="forex OR currency market",
            count=100
        )
        
        if tweets is not None:
            self.data_collector.save_data(tweets, 'forex_tweets.json')

    def analyze_data(self, symbol: str):
        """
        Perform technical and sentiment analysis
        """
        print("\nAnalyzing data...")
        
        # Technical Analysis
        print("Performing technical analysis...")
        tech_data = pd.read_csv(f'data/{symbol.lower()}_data.csv', index_col=0, parse_dates=True)
        tech_data = self.technical_analyzer.add_technical_indicators(tech_data)
        
        # Train technical model
        X_tech, y_tech = self.technical_analyzer.prepare_data(tech_data)
        train_size = int(len(X_tech) * 0.8)
        X_train_tech, X_test_tech = X_tech[:train_size], X_tech[train_size:]
        y_train_tech, y_test_tech = y_tech[:train_size], y_tech[train_size:]
        
        history = self.technical_analyzer.train_model(X_train_tech, y_train_tech)
        tech_predictions = self.technical_analyzer.predict(X_test_tech)
        
        # Sentiment Analysis
        print("Performing sentiment analysis...")
        with open('data/forex_news.json', 'r') as f:
            news_data = json.load(f)
        
        with open('data/forex_tweets.json', 'r') as f:
            tweets_data = json.load(f)
        
        news_sentiment = self.sentiment_analyzer.analyze_news(news_data)
        tweets_sentiment = self.sentiment_analyzer.analyze_tweets(tweets_data)
        market_sentiment = self.sentiment_analyzer.get_market_sentiment(news_sentiment, tweets_sentiment)
        
        # Save sentiment results
        with open('data/market_sentiment.json', 'w') as f:
            json.dump(market_sentiment, f)
        
        return tech_data, market_sentiment

    def make_prediction(self, tech_data: pd.DataFrame, sentiment_data: dict):
        """
        Make final prediction using fusion model
        """
        print("\nMaking final prediction...")
        
        # Prepare features
        X = self.fusion_model.prepare_features(tech_data, sentiment_data)
        y = self.fusion_model.create_labels(tech_data)
        
        # Train fusion model
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        self.fusion_model.train(X_train, y_train)
        
        # Evaluate model
        metrics = self.fusion_model.evaluate_model(X_test, y_test)
        print("\nModel Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Make prediction for latest data
        latest_features = X.iloc[[-1]]
        prediction, probability = self.fusion_model.predict(latest_features)
        signal = self.fusion_model.get_trading_signal(prediction[0], probability[0][1])
        
        return signal, probability[0][1]

    def run_pipeline(self, symbol: str = "EURUSD=X", days_back: int = 30):
        """
        Run the complete prediction pipeline
        """
        print(f"Starting Smart Forex analysis for {symbol}...")
        
        # Step 1: Collect data
        self.collect_data(symbol, days_back)
        
        # Step 2: Analyze data
        tech_data, sentiment_data = self.analyze_data(symbol)
        
        # Step 3: Make prediction
        signal, confidence = self.make_prediction(tech_data, sentiment_data)
        
        # Save results
        results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'confidence': float(confidence),
            'technical_trend': self.technical_analyzer.analyze_trend(tech_data),
            'market_sentiment': sentiment_data['sentiment_category']
        }
        
        with open(f'results/{symbol.lower()}_prediction.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print("\nAnalysis Complete!")
        print(f"Trading Signal: {signal}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Technical Trend: {results['technical_trend']}")
        print(f"Market Sentiment: {results['market_sentiment']}")

if __name__ == "__main__":
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""# API Keys
NEWS_API_KEY=your_news_api_key
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
""")
        print("Created .env file. Please add your API keys before running the analysis.")
    else:
        # Run the analysis
        smart_forex = SmartForex()
        smart_forex.run_pipeline() 