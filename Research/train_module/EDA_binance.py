import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('binance_Data.csv')
df.head()

df.plot(x='Open time',y=['Close','Number of trades'],kind='line')

plt.show()
plt.waitforbuttonpress()