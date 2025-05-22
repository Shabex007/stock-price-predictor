import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Configuration
STOCK_LIST = {
    'AAPL': 'Apple Inc.',
    'GOOGL': 'Alphabet Inc. (Google)',
    'MSFT': 'Microsoft Corporation',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.'
}

def get_user_selection():
    print("Available Stocks:")
    for ticker, name in STOCK_LIST.items():
        print(f"{ticker}: {name}")
    
    selections = input("\nEnter tickers (comma-separated, e.g., AAPL,MSFT): ").upper().split(',')
    return [s.strip() for s in selections if s.strip() in STOCK_LIST]

def get_stock_data(ticker, days=365*5):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return None, None, None
    
    original_close = data[['Close']].copy()
    data['Target'] = data['Close'].shift(-1)
    
    # Prepare future prediction data
    future_data = data.tail(1).copy()
    data = data[:-1].dropna()
    
    return data, original_close, future_data

def train_and_predict(ticker, company_name):
    # Get data
    data, original_close, future_data = get_stock_data(ticker)
    if data is None:
        print(f"\n⚠ Could not retrieve data for {ticker}. Skipping...")
        return
    
    # Prepare features
    X = data.drop('Target', axis=1)
    y = data['Target']
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    # Prepare future prediction
    scaled_future = scaler.transform(future_data.drop('Target', axis=1))
    future_pred = model.predict(scaled_future)[0]
    
    # Get recent prices
    last_5_days = original_close.tail(5)
    current_price = original_close.iloc[-1, 0]
    
    # Display results
    print(f"\n📈 {company_name} ({ticker}) Analysis:")
    print("\nLast 5 Trading Days:")
    print(last_5_days.round(2))
    print(f"\nCurrent Closing Price: ${current_price:.2f}")
    print(f"Predicted Next Closing Price: ${future_pred:.2f}")
    print(f"Model R² Score: {r2:.2f}")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual Prices', alpha=0.7)
    plt.plot(y_pred, label='Predicted Prices', alpha=0.7)
    plt.title(f'{company_name} Price Predictions vs Actual')
    plt.xlabel('Trading Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()

def main():
    selected = get_user_selection()
    if not selected:
        print("\n⚠ No valid stocks selected. Exiting...")
        return
    
    print("\n🚀 Processing your stocks...")
    for ticker in selected:
        company_name = STOCK_LIST.get(ticker, ticker)
        train_and_predict(ticker, company_name)

# ✅ Call directly in Jupyter, not with _name_ == "_main_"
main()
