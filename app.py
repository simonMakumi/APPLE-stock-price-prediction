import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

st.markdown("""
    <h1 style='color: red;'>Apple Stock Prediction</h1>
    """, unsafe_allow_html=True)

# Load stock data
stocks_df = pd.read_csv("data/apple_stock_and_interest_rates.csv")
stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
stocks_df.set_index('Date', inplace=True)

# Sidebar inputs
ticker = st.sidebar.text_input('Ticker', value='AAPL')
start_date = st.sidebar.date_input('Start Date', min_value=stocks_df.index.min(), max_value=stocks_df.index.max(), value=stocks_df.index.min())
end_date = st.sidebar.date_input('End Date', min_value=stocks_df.index.min(), max_value=stocks_df.index.max(), value=stocks_df.index.max())

# Filter data based on user input
stocks_df = stocks_df.loc[start_date:end_date]

# Display pricing data
fig = px.line(stocks_df, x=stocks_df.index, y='Close', title='Stock Price Over Time')
st.plotly_chart(fig)

# Tabs for different sections
pricing_data, analysis, predictions = st.tabs(['Pricing Data', 'Analysis', 'Predictions'])

# Pricing Data
with pricing_data:
    st.markdown("<h1 style='color: red;'>Price Movements</h1>", unsafe_allow_html=True)
    data2 = stocks_df.copy()
    data2['% Change'] = stocks_df['Adj Close'] / stocks_df['Adj Close'].shift(1) - 1
    data2.dropna(inplace=True)
    st.write(data2)
    annual_return = data2['% Change'].mean() * 252 * 100
    st.write('Annual return is', round(annual_return, 2), '%')

# Analysis
with analysis:
    st.markdown("<h1 style='color: red;'>Closing Prices Over Time</h1>", unsafe_allow_html=True)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(stocks_df['Close'])
    plt.title('Closing Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    st.pyplot(fig)
    
    st.markdown("<h1 style='color: red;'>Closing Price vs Time chart with 100MA</h1>", unsafe_allow_html=True)
    ma100 = stocks_df['Close'].rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(stocks_df['Close'], label='Close Price')
    plt.plot(ma100, label='100-Day Moving Average', color='orange')
    plt.title('Closing Price vs Time with 100-Day MA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)
    
    st.markdown("<h1 style='color: red;'>Closing Price vs Time chart with 200MA</h1>", unsafe_allow_html=True)
    ma200 = stocks_df['Close'].rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(stocks_df['Close'], label='Close Price')
    plt.plot(ma100, label='100-Day Moving Average', color='orange')
    plt.plot(ma200, label='200-Day Moving Average', color='red')
    plt.title('Closing Price vs Time with 100-Day and 200-Day MA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

with predictions:
    st.markdown("<h1 style='color: red;'>Predictions</h1>", unsafe_allow_html=True)
    
    # Prepare data for prediction
    data = stocks_df[['Close']]
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * 0.8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Prepare test data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # Load model
    model = load_model('model.h5')
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Create DataFrame for validation
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # Plot predictions
    fig = plt.figure(figsize=(16, 8))
    plt.plot(train['Close'], label='Train')
    plt.plot(valid[['Close', 'Predictions']], label=['Val', 'Predictions'])
    plt.title('Model Predictions')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD($)')
    plt.legend(loc='lower right')
    st.pyplot(fig)
