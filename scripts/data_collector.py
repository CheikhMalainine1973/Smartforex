import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
import tweepy
import json

class DataCollector:
    def __init__(self):
        load_dotenv()
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        self.twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.twitter_access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        # Initialize news API client
        self.news_client = NewsApiClient(api_key=self.news_api_key)
        
        # Initialize Twitter API client
        auth = tweepy.OAuthHandler(self.twitter_api_key, self.twitter_api_secret)
        auth.set_access_token(self.twitter_access_token, self.twitter_access_token_secret)
        self.twitter_client = tweepy.API(auth)

    def get_forex_data(self, symbol, start_date, end_date, interval='1d'):
        """
        Fetch forex data using yfinance
        """
        try:
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            return data
        except Exception as e:
            print(f"Error fetching forex data: {str(e)}")
            return None

    def get_news_data(self, query, from_date, to_date):
        """
        Fetch news articles related to forex market
        """
        try:
            news = self.news_client.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='relevancy'
            )
            return news['articles']
        except Exception as e:
            print(f"Error fetching news data: {str(e)}")
            return None

    def get_twitter_sentiment(self, query, count=100):
        """
        Fetch tweets related to forex market
        """
        try:
            tweets = self.twitter_client.search_tweets(q=query, count=count, tweet_mode='extended')
            return [tweet.full_text for tweet in tweets]
        except Exception as e:
            print(f"Error fetching tweets: {str(e)}")
            return None

    def save_data(self, data, filename, directory='data'):
        """
        Save data to CSV or JSON file
        """
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f)
        
        print(f"Data saved to {filepath}")

if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Get forex data
    forex_data = collector.get_forex_data(
        symbol="EURUSD=X",
        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    if forex_data is not None:
        collector.save_data(forex_data, 'eurusd_data.csv')
    
    # Get news data
    news_data = collector.get_news_data(
        query="forex OR currency market",
        from_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
        to_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    if news_data is not None:
        collector.save_data(news_data, 'forex_news.json') 