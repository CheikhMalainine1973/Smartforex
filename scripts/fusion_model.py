import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
from typing import Dict, Tuple, Union
import json

class FusionModel:
    def __init__(self):
        self.technical_scaler = StandardScaler()
        self.sentiment_scaler = StandardScaler()
        self.model = None
        self.feature_importance = None

    def prepare_features(self, technical_data: pd.DataFrame, sentiment_data: Dict) -> pd.DataFrame:
        """
        Prepare features from both technical and sentiment analysis
        """
        # Technical features
        tech_features = technical_data[['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'MACD', 'RSI']].copy()
        
        # Calculate price changes
        tech_features['price_change'] = tech_features['Close'].pct_change()
        
        # Scale technical features
        tech_features_scaled = pd.DataFrame(
            self.technical_scaler.fit_transform(tech_features),
            columns=tech_features.columns,
            index=tech_features.index
        )
        
        # Create sentiment features
        sentiment_features = pd.DataFrame({
            'news_sentiment': [sentiment_data['news_sentiment']] * len(tech_features),
            'tweets_sentiment': [sentiment_data['tweets_sentiment']] * len(tech_features),
            'overall_sentiment': [sentiment_data['overall_sentiment']] * len(tech_features)
        }, index=tech_features.index)
        
        # Scale sentiment features
        sentiment_features_scaled = pd.DataFrame(
            self.sentiment_scaler.fit_transform(sentiment_features),
            columns=sentiment_features.columns,
            index=sentiment_features.index
        )
        
        # Combine features
        combined_features = pd.concat([tech_features_scaled, sentiment_features_scaled], axis=1)
        
        return combined_features

    def create_labels(self, data: pd.DataFrame, window: int = 5) -> pd.Series:
        """
        Create labels based on future price movement
        """
        # Calculate future returns
        future_returns = data['Close'].shift(-window) / data['Close'] - 1
        
        # Create binary labels (1 for positive return, 0 for negative)
        labels = (future_returns > 0).astype(int)
        
        return labels

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the fusion model
        """
        # Initialize and train XGBoost model
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X, y)
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions and get probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities

    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions, _ = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions),
            'recall': recall_score(y, predictions),
            'f1': f1_score(y, predictions)
        }
        
        return metrics

    def save_model(self, path: str):
        """
        Save the trained model and scalers
        """
        model_data = {
            'model': self.model,
            'technical_scaler': self.technical_scaler,
            'sentiment_scaler': self.sentiment_scaler,
            'feature_importance': self.feature_importance.to_dict()
        }
        
        joblib.dump(model_data, path)

    def load_model(self, path: str):
        """
        Load a trained model and scalers
        """
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.technical_scaler = model_data['technical_scaler']
        self.sentiment_scaler = model_data['sentiment_scaler']
        self.feature_importance = pd.DataFrame(model_data['feature_importance'])

    def get_trading_signal(self, prediction: int, probability: float) -> str:
        """
        Generate trading signal based on prediction and confidence
        """
        if prediction == 1:
            if probability > 0.7:
                return "Strong Buy"
            elif probability > 0.6:
                return "Buy"
            else:
                return "Weak Buy"
        else:
            if probability > 0.7:
                return "Strong Sell"
            elif probability > 0.6:
                return "Sell"
            else:
                return "Weak Sell"

if __name__ == "__main__":
    # Example usage
    fusion_model = FusionModel()
    
    # Load technical data
    tech_data = pd.read_csv('data/eurusd_data.csv', index_col=0, parse_dates=True)
    
    # Load sentiment data
    with open('data/market_sentiment.json', 'r') as f:
        sentiment_data = json.load(f)
    
    # Prepare features
    X = fusion_model.prepare_features(tech_data, sentiment_data)
    
    # Create labels
    y = fusion_model.create_labels(tech_data)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    fusion_model.train(X_train, y_train)
    
    # Evaluate model
    metrics = fusion_model.evaluate_model(X_test, y_test)
    print("Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Get feature importance
    print("\nFeature Importance:")
    print(fusion_model.feature_importance)
    
    # Save model
    fusion_model.save_model('models/fusion_model.joblib')
    
    # Make prediction for latest data
    latest_features = X.iloc[[-1]]
    prediction, probability = fusion_model.predict(latest_features)
    signal = fusion_model.get_trading_signal(prediction[0], probability[0][1])
    
    print(f"\nLatest Trading Signal: {signal}")
    print(f"Confidence: {probability[0][1]:.2%}") 