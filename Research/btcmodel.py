
import datetime
from binance.client import Client
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Instanciate Binance client
client = Client('fiTNkE2yhe0z8jYuyHMVbTb7hxrfjfB0oRb7HXC9FX4aXHg8qUhfxApLx9xrgH7r', 'gqwsC91n0nS4VNZ7HhsnUcXSx2Dxnl8gX40RILrTohA7K9lSqa1S5kBFPNfPNsHH')

# get data
symbol = 'BTCUSDT'
# BTC = client.get_historical_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, start_str="1546300800")
# BTC = pd.DataFrame(BTC,
#                    columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
#                             'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume',
#                             'Ignore'])
BTC = pd.read_csv('binance_Data.csv')
# exit()
BTC['Open time'] = pd.to_datetime(BTC['Open time'], unit='ms')

BTC.set_index('Open time', inplace=True)

BTC['Close'] = BTC['Close'].astype(float)

data = BTC.iloc[:, 3:4].astype(float).values

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

training_set = data[:10000]
test_set = data[len(data)-1:]
# Data preprocessing (Dividing datasets to training and testing data)
# X_train = training_set[0:len(training_set) - 1]
# y_train = training_set[1:len(training_set)]

# X_test = test_set[0:len(test_set) - 1]
# y_test = test_set[1:len(test_set)]
X_test = test_set

# X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))

# Init the model
# model = Sequential()
# model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(LSTM(256))
# model.add(Dense(1))
# # Compile the model
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(X_train, y_train, epochs=50, batch_size=16, shuffle=False)
# Save the `model
# model.save('bitcoin_model.h5')
model = load_model('bitcoin_model.h5')
# Perform predictions on test data
pred = []
for i in range(20):

    predicted_price = model.predict(X_test)
    X_test = np.reshape(predicted_price, (len(predicted_price), 1, predicted_price.shape[1]))
    predicted_price = scaler.inverse_transform(predicted_price)
    pred.append(predicted_price)
print(pred)
exit()
# real_price = scaler.inverse_transform(y_test)

# Display graph of our prediction
plt.figure(figsize=(10, 4))
red_patch = mpatches.Patch(color='red', label='Predicted Price of Bitcoin')
blue_patch = mpatches.Patch(color='blue', label='Real Price of Bitcoin')
plt.legend(handles=[blue_patch, red_patch])
plt.plot(pred, color='red', label='Predicted Price of Bitcoin')
# plt.plot(real_price, color='blue', label='Real Price of Bitcoin')
plt.title('Predicted vs. Real Price of Bitcoin')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
plt.waitforbuttonpress()

