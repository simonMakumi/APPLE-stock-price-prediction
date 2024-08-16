import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.express as px
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from PIL import Image


st.title('Apple Stock Prediction')
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

stocks_df= pd.read_csv("data/apple_stock_and_interest_rates.csv")
stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
stocks_df.set_index('Date', inplace=True)
fig = px.line(stocks_df, x = stocks_df.index, y = stocks_df['Close'],title = ticker)
st.plotly_chart(fig)

pricing_data, fundamental_data, news = st.tabs(['Pricing Data','analysis','Top 10 News'])
with pricing_data:
    st.header('Price Movements')
    data2 = stocks_df
    data2['% Change'] = stocks_df['Adj Close']/stocks_df['Adj Close'].shift(1)-1
    data2.dropna(inplace=True)
    st.write(data2)
    annual_return = data2['% Change'].mean()*252*100
    st.write('Annual return is',annual_return,'%')

with analysis:
    st.subheader('Closing Price vs Time chart')
    fig=plt.figure(figsize=(12,6))
    plt.plot(stocks_df.Close)
    st.pyplot(fig)
    
    st.subheader('Closing Price vs Time chart with 100MA')
    ma100 = stocks_df.Close.rolling(100).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ma100)
    plt.plot(stocks_df.Close)
    st.pyplot(fig)
    
    st.subheader('Closing Price vs Time chart with 200MA')
    ma100 = stocks_df.Close.rolling(100).mean()
    ma200 = stocks_df.Close.rolling(200).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(stocks_df.Close)
    st.pyplot(fig)
    st.write('Analysis')

with news:
    st.write('News')

#Visualizations


#data_training = pd.Dataframe(stocks_df['Close'][0:int(len(df)*0.70)])
#data_testing = pd.Dataframe(stocks_df['Close'][int(len(df)*0.70):int(len(df))])

#data_training.shape
#data_testing.shape

#Create a new df with only the 'Close' column
data = stocks_df.filter(['Close'])
#Convert df to np array
dataset = data.values
training_data_len = math.ceil(len(dataset)* .8)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

test_data = scaled_data[training_data_len - 60:,:]
x_test = []
y_test = dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

#Covert data into a np array
x_test = np.array(x_test)
#Reshape the data
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
x_test.shape

model = load_model('model.h5')

predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)

## Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD($)',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'], loc='lower right')
st.pyplot(fig2)





