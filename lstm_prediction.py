import yfinance as yf
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential


rcParams['figure.figsize'] = 20, 10

def prediction(ticker,period):
    df = yf.download(ticker, period=period)
    df1 = df['Adj Close']
    df1.rename('Close', inplace=True)
    prices = df1.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
    train_len = math.ceil(len(prices) * 0.8)
    train_data = scaled_prices[0:train_len, :]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(scaled_prices[i-60:i, 0])
        y_train.append(scaled_prices[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    test_data = scaled_prices[train_len-60:, :]
    x_test = []
    y_test = prices[train_len:]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=3)
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    print(f"The RMSE score is {rmse}")
    data = df.filter(['Adj Close'])
    train = data[:train_len].copy()
    validation = data[train_len:].copy()
    validation['Predictions'] = predictions
    plt.figure(figsize=(16, 8))
    plt.title('Model Predictions vs Real Prices')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.plot(train)
    plt.plot(validation[['Adj Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

ticker = input("Enter the stock price : ")
print("\n yfinance stock format ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'] \n")
period = input("Enter the period : ")
prediction(ticker,period)


