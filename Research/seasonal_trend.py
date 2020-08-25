from statsmodels.tsa.seasonal import STL
import statsmodels as sm
import pandas as pd
import os

def add_seasonal_data(df):
    resopen = sm.tsa.seasonal.seasonal_decompose(df['open'], model='additive', freq=24)
    df['open_seasonal']  = resopen.seasonal
    resclose = sm.tsa.seasonal.seasonal_decompose(df['close'], model='additive', freq=24)
    df['close_seasonal'] = resclose.seasonal
    reslow = sm.tsa.seasonal.seasonal_decompose(df['low'], model='additive', freq=24)
    df['low_seasonal']= reslow.seasonal
    reshigh = sm.tsa.seasonal.seasonal_decompose(df['high'], model='additive', freq=24)
    df['high_seasonal'] =reshigh.seasonal

    del resopen
    del resclose
    del reslow
    del reshigh
    return df




df_ge = pd.read_csv('/home/manoj/Downloads/office/trading_bot/data/bitcoin_data_v1.csv')
df_ge = add_seasonal_data(df_ge)
df_ge.to_csv('/home/manoj/Downloads/office/trading_bot/api_src/Data/bitcoin_seasonal_Data.csv',index=False)