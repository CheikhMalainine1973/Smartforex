import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

class TechnicalAnalyzer:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.sequence_length = 60  # Number of days to look back

    def add_technical_indicators(self, df):
        """
        Add various technical indicators to the dataframe
        """
        # Moving Averages
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        
        # MACD
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # RSI
        df['RSI'] = RSIIndicator(close=df['Close']).rsi()
        
        # Bollinger Bands
        bollinger = BollingerBands(close=df['Close'])
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Lower'] = bollinger.bollinger_lband()
        
        return df

    def prepare_data(self, df):
        """
        Prepare data for LSTM model
        """
        # Drop NaN values
        df = df.dropna()
        
        # Select features
        features = ['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'MACD', 'MACD_Signal', 'RSI', 'BB_Upper', 'BB_Lower']
        data = df[features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length, 0])  # Predict Close price
        
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """
        Build LSTM model
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Train the LSTM model
        """
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        return history

    def predict(self, X):
        """
        Make predictions using the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        predictions = self.model.predict(X)
        predictions = self.scaler.inverse_transform(
            np.concatenate((predictions, np.zeros((len(predictions), X.shape[2]-1))), axis=1)
        )[:, 0]
        
        return predictions

    def plot_predictions(self, actual, predicted, title="Price Prediction"):
        """
        Plot actual vs predicted prices
        """
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual Price')
        plt.plot(predicted, label='Predicted Price')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def analyze_trend(self, df):
        """
        Analyze the trend based on technical indicators
        """
        # Calculate trend strength
        trend_strength = 0
        
        # Check moving averages
        if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1]:
            trend_strength += 1
        if df['Close'].iloc[-1] > df['SMA_50'].iloc[-1]:
            trend_strength += 1
            
        # Check MACD
        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
            trend_strength += 1
            
        # Check RSI
        if df['RSI'].iloc[-1] > 50:
            trend_strength += 1
            
        # Determine trend
        if trend_strength >= 3:
            return "Strong Uptrend"
        elif trend_strength >= 2:
            return "Moderate Uptrend"
        elif trend_strength == 1:
            return "Neutral"
        else:
            return "Downtrend"

if __name__ == "__main__":
    # Example usage
    analyzer = TechnicalAnalyzer()
    
    # Load and prepare data
    df = pd.read_csv('data/eurusd_data.csv', index_col=0, parse_dates=True)
    df = analyzer.add_technical_indicators(df)
    
    # Prepare data for LSTM
    X, y = analyzer.prepare_data(df)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    history = analyzer.train_model(X_train, y_train)
    
    # Make predictions
    predictions = analyzer.predict(X_test)
    
    # Plot results
    analyzer.plot_predictions(df['Close'].values[-len(predictions):], predictions)
    
    # Analyze trend
    trend = analyzer.analyze_trend(df)
    print(f"Current Trend: {trend}") 