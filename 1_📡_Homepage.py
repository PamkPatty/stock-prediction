# pip install streamlit prophet yfinance plotly seaborn streamlit_option_menu matplotlib
import streamlit as st
from datetime import date
import requests
from bs4 import BeautifulSoup
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title='Homepage')
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'}
st.title('Stock Forecast App: Homepage📡')


selected_stock = st.text_input("👇Input ticker dataset for prediction:", "AAPL")
#selected_stock = st.selectbox("👇Input dataset for prediction:", si.tickers_sp500())
warning_1 = st.info("Please, refer to Yahoo Finance for a ticker list of **S&P 500🎫** applicable ticker symbols.  Type the symbol **EXACTLY** as provided by Yahoo Finance.")

# Current Stock value:
NEW_LINK = 'https://finance.yahoo.com/quote/{}'.format(selected_stock)

tickerData = yf.Ticker(selected_stock) # Get ticker data
# App title
string_name = tickerData.info['longName']
st.title(string_name + "🌌")

full_page_stock = requests.get(NEW_LINK, headers=headers)
soup = BeautifulSoup(full_page_stock.content, 'html.parser')
stock_price = soup.findAll("fin-streamer", {"class": "Fw(b) Fz(36px) Mb(-4px) D(ib)", "data-test": "qsp-price"})
stock_price_change = soup.findAll("fin-streamer", {"class": "Fw(500) Pstart(8px) Fz(24px)", "data-test": "qsp-price-change"})
stock_change_percent = soup.findAll("fin-streamer", {"class": "Fw(500) Pstart(8px) Fz(24px)", "data-field": "regularMarketChangePercent"})
st.subheader(stock_price[0].text.replace(",", "") + "💲")
st.text("🙊Price changed: " + stock_price_change[0].text)
st.text("🙉Percentage: " + stock_change_percent[0].text)


# Current UAH to USD Currency value:
UAH_USD = 'https://www.google.com/search?q=%D0%BA%D1%83%D1%80%D1%81+%D0%B4%D0%BE%D0%BB%D0%BB%D0%B0%D1%80%D0%B0'
full_page_uah = requests.get(UAH_USD, headers=headers)
soup = BeautifulSoup(full_page_uah.content, 'html.parser')
uah_usd_price = soup.findAll("span", {"class": "DFlfde SwHCTb", "data-precision": "2"})
uah = uah_usd_price 
st.text("🙈USD to UAH: " + uah_usd_price[0].text.replace(",", ".") + "₴")



# model_name = st.title('Forecasting model🎓')
# picked_model = ['Prophet', 'ARIMA', 'LSTM']
# chosen_model = st.selectbox("👇Select your model:", picked_model)
# with st.expander("🧠What is the difference?"):
#     st.info("Trying different models for forecasting allows you to compare their performance, determine their suitability for your data, test their robustness, gain diverse insights, explore ensembling opportunities, and stay up-to-date with the latest developments in the field. This iterative approach improves your forecasting capabilities and increases the chances of making accurate predictions.")

START = st.date_input("📆Start date:", date(2000, 1, 1))	
TODAY = date.today().strftime("%Y-%m-%d")

# if chosen_model == "Prophet":

    # Year range:
n_years = st.slider('⏳Day range of prediction:', 1, 3650, 100)
period = n_years

if n_years >= 365:
    st.warning("The more days you select, the less accuracy will be.🤕")


data = yf.download(selected_stock, START, TODAY)
data.reset_index(inplace=True)


st.subheader('Raw ['+ selected_stock +'] Dataset🥩')
data_load_state = st.success('Dataset Uploaded succcessfuly!✅')
with st.expander("👀Check it Out"):
    st.write(data.tail())
with st.expander("🤔Any missing Values?"):
    st.write(data.isnull().sum())
    st.success('We don`t have any missing values!✅')



# Candle Plot
st.subheader('Candlestick Plot 🧨')
fig = go.Figure()
fig.add_trace(go.Candlestick(x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'], name = 'market data'))
fig.update_layout(
title='📈Stock share price evolution',
yaxis_title='Stock Price (USD per Shares)',
xaxis_title='Time(Yearly)')
st.plotly_chart(fig)

st.subheader('Distribution of Data Points since ' + str(START) + '📊')
with st.expander("🧩Show Clue"):
    st.write(data.describe())
fig = plt.figure()
sns.displot(data['Close']) 
st.pyplot(fig)

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
with st.spinner('Loading data Forecast data 🎱...'):
    st.subheader('Forecast data 🎱')
    with st.expander("👀Check it Out"):
        st.write(forecast.tail())

# FBProphet Plot
st.subheader('FBProphet Plot 🎯')
with st.expander("🔮How many day(s)?"):
    st.success(f'😱Forecast plot for {period} day(s)')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)


st.subheader("Forecast components😴")
with st.expander("🧩Clue what Graphs shows"):
    st.markdown('1️⃣ Graph shows information about the trend.')
    st.markdown('2️⃣ Graph shows information about the weekly trend.')
    st.markdown('3️⃣ Graph gives us information about the annual tenure.')
fig2 = m.plot_components(forecast)
st.write(fig2)


st.subheader('ChangePoints Plot🔬')
with st.expander("🧩Show Clue"):
    st.markdown('The **Changepoints** are the date points in which the time series present abrupt changes in the trajectory.')

fig3 = m.plot(forecast)
a = add_changepoints_to_plot(fig3.gca(), m, forecast)
st.write(fig3)


# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)

# Split data into train and test sets
train_size = int(len(df_train) * 0.8)
train_data = df_train[:train_size]
test_data = df_train[train_size:]

# Make predictions on test data
if not test_data.empty:
    forecast = m.predict(test_data)

    # Calculate Prophet's Test MAE, RMSE, and R-squared
    mae = mean_absolute_error(test_data['y'], forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test_data['y'], forecast['yhat']))
    r2 = r2_score(test_data['y'], forecast['yhat'])

    # Show Prophet's Test Metrics
    st.subheader("Prophet's Test Metrics🧪")

    st.write("👁‍🗨Prediction trust factor:")
    if r2 >= 0.85:
        st.success('📗 - Hight level of trust factor.')
    elif r2 >= 0.75:
        st.info('📘 - Good level of trust factor.')
    elif r2 >= 0.7:
        st.warning('📒 - Satisfactory level of trust factor.')
    elif r2 >= 0.5:
        st.error('📙 - Low level of trust factor.')
    elif r2 >= -1:
        st.error('📕 - Very low level of trust factor.')

    with st.expander("🧩Show Clue"):
        st.info("Model evaluation is an essential step in the machine learning and data science workflow. It involves assessing the performance and effectiveness of a trained model using various evaluation metrics and techniques.")
        st.write("R-squared tends to 1️⃣ from -1️⃣")
        st.write("MAE tends to 0️⃣ from ♾")
        st.write("RMSE tends to 0️⃣ from ♾")
    st.write("R-squared:", r2)
    st.write("MAE (Mean Absolute Error):", mae)
    st.write("RMSE (Root Mean Squared Error):", rmse)

    
else:
    st.warning("No test data available for the specified date range.")

###########################################################################################################


# !pip install yfinance

# # Some other packages that I am going to use in this notebook
# !pip install hvplot
# !pip install mplcyberpunk
# import yfinance as yf
# import pandas as pd
# import pandas as pd
# import numpy as np
# import hvplot.pandas

# import mplcyberpunk
# from matplotlib import style
# style.use('cyberpunk')

# import matplotlib.pyplot as plt
# import seaborn

# elif chosen_model == "ARIMA":
#     st.header("ARIMA")

   

# elif chosen_model == "LSTM":
#     st.header("LSTM")

#         # Define the LSTM model
#     model = Sequential()
#     model.add(LSTM(units=64, input_shape=(1, 1)))
#     model.add(Dense(units=1))

#     @st.cache_resource
#     def load_data(selected_stock):
#         data = yf.download(selected_stock, START, TODAY)
#         data.reset_index(inplace=True)
#         return data

#     with st.spinner('Loading data......'):
#         data = load_data(selected_stock)

#     # Display raw dataset
#     st.subheader('Raw ['+ selected_stock +'] Dataset🥩')
#     data_load_state = st.success('Dataset Uploaded successfully!✅')
#     with st.expander("👀Check it Out"):
#         st.write(data.tail())

#     # Candle Plot
#     st.subheader('Candlestick Plot 🧨')
#     fig = go.Figure()
#     fig.add_trace(go.Candlestick(x=data['Date'],
#                 open=data['Open'],
#                 high=data['High'],
#                 low=data['Low'],
#                 close=data['Close'], name='market data'))
#     fig.update_layout(
#         title='📈Stock share price evolution',
#         yaxis_title='Stock Price (USD per Shares)')
#     st.plotly_chart(fig)

#     # Distribution of Data Points
#     st.subheader('Distribution of Data Points since ' + str(START) + '📊')
#     with st.expander("🧩Show Clue"):
#         st.markdown('The **Distribution** of a data set is the shape of the graph when all possible values are plotted on a frequency graph (showing how often they occur). This sample is used to make conclusions about the whole data set.')
#     fig = plt.figure(figsize=(9, 7))
#     sns.displot(data['Close'])
#     st.pyplot(fig)

#     # Split data into train and test sets
#     train_size = int(len(data) * 0.8)
#     train_set = data[:train_size]['Close'].values
#     test_set = data[train_size:]['Close'].values

#     # Create input sequences for LSTM
#     def create_sequences(dataset, window_size):
#         X = []
#         y = []
#         for i in range(len(dataset) - window_size):
#             X.append(dataset[i:i+window_size])
#             y.append(dataset[i+window_size])
#         return np.array(X), np.array(y)

#     # Define window size for LSTM
#     window_size = 1

#     # Create input sequences for LSTM
#     X_train, y_train = create_sequences(train_set, window_size)
#     X_test, y_test = create_sequences(test_set, window_size)

#     # Reshape the input data to include the timestep dimension
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#     # Define the LSTM model
#     model = Sequential()
#     model.add(LSTM(units=50, activation='relu', input_shape=(window_size, 1)))
#     model.add(Dense(units=1))

#     # Compile and fit the model
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(X_train, y_train, epochs=10, batch_size=32)

#     # Make predictions
#     y_pred = model.predict(X_test)

#     # Reshape the predictions and true values to original shape
#     y_pred = np.reshape(y_pred, (y_pred.shape[0],))
#     y_test = np.reshape(y_test, (y_test.shape[0],))

#     # Calculate evaluation metrics
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     r2 = r2_score(y_test, y_pred)

#     # Plot the adjusted predicted and actual values
#     fig, ax = plt.subplots(figsize=(9, 5))
#     ax.plot(y_test, label='Actual')
#     ax.plot(y_pred, label='Predicted')
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Stock Price')
#     ax.set_title('Actual vs Adjusted Predicted Stock Prices')
#     ax.legend()
#     st.pyplot(fig)

#     # Display evaluation metrics
#     st.subheader('Evaluation Metrics 📏')
#     st.write('Mean Absolute Error (MAE):', mae)
#     st.write('Root Mean Squared Error (RMSE):', rmse)
#     st.write('R-squared Score (R2):', r2)
    



