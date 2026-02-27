import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def build_trading_model(ticker_symbol):
    print(f"Fetching data for {ticker_symbol}...")

    data = yf.download(ticker_symbol, period="2y")
    
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    
    data['Target_Next_Close'] = data['Close'].shift(-1)
      
    data = data.dropna()
    
    features = ['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50']
    X = data[features]
    y = data['Target_Next_Close']    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
     
    print("Training the Machine Learning Model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
   
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    print(f"Model Mean Absolute Error: ${error:.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label="Actual Price", color='blue')
    plt.plot(y_test.index, predictions, label="Predicted Price", color='red', linestyle='dashed')
    plt.title(f"{ticker_symbol} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    build_trading_model("AAPL")