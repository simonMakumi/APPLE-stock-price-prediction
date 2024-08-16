import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from PIL import Image

# Load your trained model
model = load_model('model.h5')

# Load and preprocess data
df = pd.read_csv('data/apple_stock_and_interest_rates.csv')
data = df.filter(['Date', 'Close'])
data['Date'] = pd.to_datetime(data['Date'])
dataset = data['Close'].values.reshape(-1, 1)
training_data_len = int(len(dataset) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the testing data set
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
test_dates = data['Date'][training_data_len:].reset_index(drop=True)

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to numpy arrays
x_test = np.array(x_test)
# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "About"])

if page == "Home":
    st.title("Apple Stock Price Prediction")
    #st.image('path_to_image/apple_logo.png', use_column_width=True)  # Example image
    st.header("Welcome to the Apple Stock Price Prediction App")
    st.write("""
    Apple Inc. is one of the world's leading technology companies, and its stock prices are closely monitored by investors worldwide. 
    This app allows you to predict future stock prices using a Long Short-Term Memory (LSTM) model, a type of recurrent neural network
    known for its effectiveness in time series forecasting.
    
    
    ### Disclaimer
    Please note that this app is for educational purposes only. The predictions provided should not be used as financial advice or for
    making investment decisions.
    """)

elif page == "Prediction":
    st.title('Apple Stock Price Prediction')

    # Select date and period for prediction
    start_date = st.date_input('Select Start Date', min_value=data['Date'].min(), max_value=data['Date'].max())
    period = st.selectbox('Select Prediction Period', ['One Day', 'One Week', 'Two Weeks', 'One Month', 'Two Months', 'One Year'])

    if st.button('Predict'):
        # Get predictions
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        if period == 'One Day':
            st.subheader(f'Prediction for {test_dates.iloc[-1].date()}')
            st.write(f"${predictions[-1][0]:.2f}")
        
        elif period == 'One Week':
            st.subheader(f'Predictions for the last week:')
            for i in range(-7, 0):
                st.write(f"{test_dates.iloc[i].date()}: ${predictions[i][0]:.2f}")
        
        else:
            st.subheader(f'Prediction for {test_dates.iloc[-1].date()}')
            st.write(f"${predictions[-1][0]:.2f}")
        

elif page == "About":
    st.title("About the Project")
    st.write(""" 
    The project focuses on predicting Apple Inc.'s stock prices using an LSTM model, combining both historical stock prices and interest
    rates.
    """)

    st.subheader("Team Members")
    st.write("**Annbellah Mbungu**: [GitHub](https://github.com/Annbellah)")
    st.write("**Simon Makumi**: [GitHub](https://github.com/simonMakumi)")
    st.write("**Kelsey Maina**: [GitHub](https://github.com/Kelsey-Maina)")
    st.write("**Esther Njagi**: [GitHub](https://github.com/emukami2)")
    
    st.subheader("Summary")
    st.write("""
    The iStock Analysts have developed this predictive model as a demonstration of time series analysis and machine learning techniques.
    While the model provides predictions based on historical data, it's important to remember that past performance does not guarantee
    future results.
    The project was a collaborative effort, showcasing the skills and teamwork of all members involved.
    """)
