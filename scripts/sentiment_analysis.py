from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from textblob import TextBlob
import json
import pandas as pd
from typing import List, Dict, Union

class SentimentAnalyzer:
    def __init__(self):
        # Initialize FinBERT model
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Sentiment labels
        self.labels = ['positive', 'negative', 'neutral']

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text using FinBERT
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            
        return {
            'positive': scores[0][0].item(),
            'negative': scores[0][1].item(),
            'neutral': scores[0][2].item()
        }

    def analyze_news(self, news_data: List[Dict]) -> pd.DataFrame:
        """
        Analyze sentiment of news articles
        """
        results = []
        
        for article in news_data:
            # Combine title and description for analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            
            # Get FinBERT sentiment
            sentiment = self.analyze_text(text)
            
            # Get TextBlob sentiment as additional metric
            blob = TextBlob(text)
            textblob_sentiment = blob.sentiment.polarity
            
            results.append({
                'title': article.get('title', ''),
                'published_at': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', ''),
                'positive': sentiment['positive'],
                'negative': sentiment['negative'],
                'neutral': sentiment['neutral'],
                'textblob_sentiment': textblob_sentiment
            })
        
        return pd.DataFrame(results)

    def analyze_tweets(self, tweets: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment of tweets
        """
        results = []
        
        for tweet in tweets:
            # Get FinBERT sentiment
            sentiment = self.analyze_text(tweet)
            
            # Get TextBlob sentiment
            blob = TextBlob(tweet)
            textblob_sentiment = blob.sentiment.polarity
            
            results.append({
                'tweet': tweet,
                'positive': sentiment['positive'],
                'negative': sentiment['negative'],
                'neutral': sentiment['neutral'],
                'textblob_sentiment': textblob_sentiment
            })
        
        return pd.DataFrame(results)

    def get_market_sentiment(self, news_df: pd.DataFrame, tweets_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate overall market sentiment score
        """
        # Calculate weighted sentiment scores
        news_weight = 0.7
        tweets_weight = 0.3
        
        # News sentiment
        news_sentiment = (
            news_df['positive'].mean() - news_df['negative'].mean()
        ) * news_weight
        
        # Twitter sentiment
        tweets_sentiment = (
            tweets_df['positive'].mean() - tweets_df['negative'].mean()
        ) * tweets_weight
        
        # Overall sentiment
        overall_sentiment = news_sentiment + tweets_sentiment
        
        # Determine sentiment category
        if overall_sentiment > 0.3:
            sentiment_category = "Strongly Bullish"
        elif overall_sentiment > 0.1:
            sentiment_category = "Moderately Bullish"
        elif overall_sentiment > -0.1:
            sentiment_category = "Neutral"
        elif overall_sentiment > -0.3:
            sentiment_category = "Moderately Bearish"
        else:
            sentiment_category = "Strongly Bearish"
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_category': sentiment_category,
            'news_sentiment': news_sentiment,
            'tweets_sentiment': tweets_sentiment
        }

    def plot_sentiment_distribution(self, df: pd.DataFrame, title: str):
        """
        Plot sentiment distribution
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='textblob_sentiment', bins=20)
        plt.title(title)
        plt.xlabel('Sentiment Score')
        plt.ylabel('Count')
        plt.show()

if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Load news data
    with open('data/forex_news.json', 'r') as f:
        news_data = json.load(f)
    
    # Analyze news
    news_sentiment = analyzer.analyze_news(news_data)
    
    # Load tweets
    with open('data/forex_tweets.json', 'r') as f:
        tweets_data = json.load(f)
    
    # Analyze tweets
    tweets_sentiment = analyzer.analyze_tweets(tweets_data)
    
    # Get market sentiment
    market_sentiment = analyzer.get_market_sentiment(news_sentiment, tweets_sentiment)
    
    print(f"Market Sentiment: {market_sentiment['sentiment_category']}")
    print(f"Overall Sentiment Score: {market_sentiment['overall_sentiment']:.2f}")
    
    # Plot sentiment distributions
    analyzer.plot_sentiment_distribution(news_sentiment, "News Sentiment Distribution")
    analyzer.plot_sentiment_distribution(tweets_sentiment, "Twitter Sentiment Distribution") 