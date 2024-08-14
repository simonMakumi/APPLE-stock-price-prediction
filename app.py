#app.py
# importing libraries for this model 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
stocks_df= pd.read_csv("data/apple_stock_and_interest_rates.csv")
stocks_df

# Preparing the data for Prophet
df = stock_df.reset_index()[['Date', 'Close']]
df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Initializing the Prophet model
model = Prophet()

# Adding US holidays to the model
model.add_country_holidays(country_name='US')

# Fitting the model
model.fit(df)


