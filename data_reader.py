#This script grabs data from the internet for a given ticker price
#It allows the user to store the data as a CSV for later user
#It allows the user to plot the data and apply functions to the data generating new columns

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

#start at 1/1/00
start = dt.datetime(2015,1,1)
#end at 12/31/16
end = dt.datetime(2016,12,31)


#grab data from yahoo finance for the given ticker over period of start to end
df = web.DataReader('CAT', 'yahoo', start, end)

#prints first 5 rows of data that was grabbed
print(df.head())

#convert collected data into csv and saves data in given file name
df.to_csv('cat.csv')

#reads csv file into data frame parsed by date
#df = pd.read_csv('cat.csv', parse_dates=True, index_col=0)


#plots the data 
df.plot()
plt.show()

#plots desired entry
df['Volume'].plot()

#how to address individual columns
print(df[['Open','High']])

#adds 100 day moving average to data frame
df['100movAvg'] = df['Adj Close'].rolling(window=100).mean()

print(df.tail())
