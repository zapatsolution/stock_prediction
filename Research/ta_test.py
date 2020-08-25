# import numpy as np
# from nose.tools import assert_equals, assert_true, assert_raises
#
import talib
from talib import stream
# import cryptowatch as cw
import pandas as pd
# Set your API Key, it is by default read from  your ~/.cw/credentials.yml file
# cw.api_key = "IJPIPMBLVOH5F45QO38D"
#
#
#
# convert time to date

dff = pd.read_csv('output_hourly_BTCPERP.csv')
dff['Date'] = pd.to_datetime(dff['time'],unit='s')
dff.to_csv('output_hourly_BTCPERP.csv',index=False)
exit()


# candles = cw.markets.get("FTX:btcusd", ohlc=True)

# This will return a list of 1 minute candles, each candle being a list with:
# [close_timestamp, open, high, low, close, volume_base, volume_quote].
# The oldest candle will be the first one in the list, the most recent will be the last one.
# print(candles.of_1m[:-30])
# df = pd.DataFrame(candles.of_1h)
df = pd.read_csv('FTX_BTCPERP, 60.csv')
df.columns = ['time', 'open', 'high', 'low', 'close', 'Volume', 'Volume MA']
candle_names  = talib.get_function_groups()['Pattern Recognition']
op = df['open']
hi = df['high']
lo = df['low']
cl = df['close']
for candle in candle_names:
    # below is same as;
    # df["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(op, hi, lo, cl)
    df[candle] = getattr(talib, candle)(op, hi, lo, cl)
# r = stream.CDL3BLACKCROWS(df[1], df[2], df[3], df[4])

df.to_csv('output_hourly_BTCPERP.csv',index=False)

