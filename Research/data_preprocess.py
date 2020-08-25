import os
import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('/home/manoj/Downloads/office/trading_bot/data/bitcoin_data.csv')
print(df.head())


#
# plt.figure()
# plt.plot(df["open"])
# plt.plot(df["high"])
# plt.plot(df["Volume"])
# # plt.plot(df["close"])
# plt.title('GE stock price history')
# plt.ylabel('Price (USD)')
# plt.xlabel('Days')
# plt.legend(['Open','High','Low'], loc='upper left')
# plt.show()
# plt.waitforbuttonpress()

df_use = df[['open','high','low','close','Volume']]
print(df_use.head())
print("checking if any null values are present\n", df_use.isna().sum())


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

train_cols = ["Open","High","Low","Close","Volume"]
df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
print("Train and Test size", len(df_train), len(df_test))
# scale the feature MinMax, build array
x = df_train.loc[:,train_cols].values
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])