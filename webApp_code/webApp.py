import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from altair import datum
from datetime import *
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from copy import deepcopy as dc
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


st.title("Amazon Stock Analysis and Prediction")

tab1, tab2, tab3, tab4 = st.tabs(['Stock Study and Analysis', 'Interactive Analysis', 'Future Predictions', 'Conclusion'])

stock_info = pd.read_csv('https://raw.githubusercontent.com/pranitahuja00/Amazon-Stock-Prediction/main/data/AMZN.csv')
stock_daily = pd.read_csv("https://raw.githubusercontent.com/pranitahuja00/Amazon-Stock-Prediction/main/data/AMZN_daily.csv")
stock_weekly = pd.read_csv("https://raw.githubusercontent.com/pranitahuja00/Amazon-Stock-Prediction/main/data/AMZN_weekly.csv")
stock_monthly = pd.read_csv("https://raw.githubusercontent.com/pranitahuja00/Amazon-Stock-Prediction/main/data/AMZN_monthly.csv")
stock_info['Date']=pd.to_datetime(stock_info['Date'])
stock_info=stock_info.drop('Adj Close', axis=1)
stock_daily['Date']=pd.to_datetime(stock_daily['Date'])
stock_daily=stock_daily.drop('Adj Close', axis=1)
stock_weekly['Date']=pd.to_datetime(stock_weekly['Date'])
stock_weekly=stock_weekly.drop('Adj Close', axis=1)
stock_monthly['Date']=pd.to_datetime(stock_monthly['Date'])
stock_monthly=stock_monthly.drop('Adj Close', axis=1)

# Engineering Stock Movement Feature
previous_day_price = stock_daily["Close"].shift(+1)
previous_day_price[0] = stock_daily["Close"][0]
stock_daily['previous_day_price'] = previous_day_price
stock_daily['Movement'] = 0
stock_daily.loc[stock_daily['Close'] > stock_daily['previous_day_price'], 'Movement'] = 1
stock_daily.loc[stock_daily['Close'] < stock_daily['previous_day_price'], 'Movement'] = -1
stock_daily = stock_daily.drop('previous_day_price', axis=1)

previous_day_price = stock_weekly["Close"].shift(+1)
previous_day_price[0] = stock_weekly["Close"][0]
stock_weekly['previous_day_price'] = previous_day_price
stock_weekly['Movement'] = 0
stock_weekly.loc[stock_weekly['Close'] > stock_weekly['previous_day_price'], 'Movement'] = 1
stock_weekly.loc[stock_weekly['Close'] < stock_weekly['previous_day_price'], 'Movement'] = -1
stock_weekly = stock_weekly.drop('previous_day_price', axis=1)

previous_day_price = stock_monthly["Close"].shift(+1)
previous_day_price[0] = stock_monthly["Close"][0]
stock_monthly['previous_day_price'] = previous_day_price
stock_monthly['Movement'] = 0
stock_monthly.loc[stock_monthly['Close'] > stock_monthly['previous_day_price'], 'Movement'] = 1
stock_monthly.loc[stock_monthly['Close'] < stock_monthly['previous_day_price'], 'Movement'] = -1
stock_monthly = stock_monthly.drop('previous_day_price', axis=1)


# TAB 1
with tab1:
    st.subheader("Stock Analysis")

    st.write("Monthly stock closing price over the years (Scale: Log):")
    close_price_chart = alt.Chart(stock_monthly).mark_line().encode(
        x=alt.Y('Date'),
        y=alt.Y('Close').scale(type='log').title('Closing Price')
    ).interactive()
    st.altair_chart(close_price_chart, use_container_width=True)
    st.write("The above chart shows the overall stock trends of Amazon on a monthly basis (Each data point refers to the closing price of the last day of each month) from May 1997 (When the company went public) to Dec 2023. Amazon stock first boomed during the late 90s which is also known as the dot com bubble after which we can notice a general upward trend in the stock price with some significant upward and downward movements in between. The first major downward trend can be seen in the year 2000 (The early 2000s recession) followed by a significant fall in 2008 (The Great Recession).")

    st.write("Monthly stock trade volume over the years")
    volume_chart = alt.Chart(stock_monthly).mark_line().encode(
        x='Date',
        y=alt.Y('Volume').scale(type='log')
    ).interactive()
    st.altair_chart(volume_chart, use_container_width=True)
    st.write("Stock trade volume tells the amount of stocks traded between two parties in a specific amount of time. It is oftenly used as a measure of stock popularity in the market. In case of Amazon, stock trade volume is showing a general downward trend with significant ups and downs in between. Dot Com Bubble (Late 90s) era proved to be the time when Amazon stocks boomed in popularity along with price with significant ups and downs in between followed by an all time low in the recent months.")

    st.write("Closing stock price data from Dec 2022 - Dec 2023:")
    recent_daily_price = alt.Chart(stock_daily.loc[(stock_daily['Date'] >= datetime(2022,12,1)) & (stock_daily['Date'] <= datetime(2023,12,5))]).mark_line().encode(
        x=alt.X('Date').title('Date (Daily)'),
        y=alt.Y('Close', scale=alt.Scale(domain=[stock_daily['Close'].min(), stock_daily['Close'].max()])).title('Closing Price')
    ).interactive().properties(
    width=300,
    height=200
    )
    recent_weekly_price = alt.Chart(stock_weekly.loc[(stock_weekly['Date'] >= datetime(2022,12,1)) & (stock_weekly['Date'] <= datetime(2023,12,5))]).mark_line().encode(
        x=alt.X('Date').title('Date (Weekly)'),
        y=alt.Y('Close', scale=alt.Scale(domain=[stock_weekly['Close'].min(), stock_weekly['Close'].max()])).title('Closing Price')
    ).interactive().properties(
    width=300,
    height=200
    )

    recent_price_box = px.box(stock_daily.loc[(stock_daily['Date'] >= datetime(2022,12,1)) & (stock_daily['Date'] <= datetime(2023,12,5))], x=['Close', 'Open', 'High', 'Low'])
 
    stock_daily_22_23 = stock_daily.loc[(stock_daily['Date'] >= datetime(2022,12,1)) & (stock_daily['Date'] <= datetime(2023,12,5))]

    fig = go.Figure()
    fig.add_trace(go.Box(x=stock_daily_22_23['Close'], name='Close Price',
                marker_color = 'indianred'))
    fig.add_trace(go.Box(x=stock_daily_22_23['Open'], name = 'Open Price',
                marker_color = 'lightseagreen'))
    fig.add_trace(go.Box(x=stock_daily_22_23['High'], name='High',
                marker_color = 'Blue'))
    fig.add_trace(go.Box(x=stock_daily_22_23['Low'], name = 'Low',
                marker_color = 'Yellow'))

    st.altair_chart(alt.hconcat(recent_weekly_price,recent_daily_price), use_container_width=True)
    st.plotly_chart(fig, use_container_width=True)

    
    #st.write("Interday stock movement in the past 150 days")
    movement_chart = alt.Chart(stock_daily.tail(150)).mark_line().encode(
        x='Date',
        y='Movement'
    ).interactive()
    #st.altair_chart(movement_chart, use_container_width=True)

with tab2:
    st.subheader("Interactive Analysis")
    st.write("Stock Price and trade volume over custom time period:")
    start_date = st.slider('Select the starting date', value = datetime(1997,5,20), min_value = datetime(1997,5,20), max_value = datetime(2023,11,3))
    end_date = st.slider('Select the ending date', value = datetime(1997,6,20), min_value = datetime(1997,6,20), max_value = datetime(2023,12,3))

    custom_price_chart = alt.Chart(stock_weekly.loc[(stock_weekly['Date'] >= start_date) & (stock_weekly['Date'] <= end_date)]).mark_line().encode(
        x=alt.X('Date').title('Date (Weekly)'),
        y=alt.Y('Close', scale=alt.Scale(domain=[stock_weekly['Close'].min(), stock_weekly['Close'].max()])).title('Closing Price').scale(type='log')
    ).interactive()
    st.altair_chart(custom_price_chart, use_container_width=True)

    stock_custom_box = stock_daily.loc[(stock_daily['Date'] >= start_date) & (stock_daily['Date'] <= end_date)]

    fig_custom = go.Figure()
    fig_custom.add_trace(go.Box(x=stock_custom_box['Close'], name='Close Price',
                marker_color = 'indianred'))
    fig_custom.add_trace(go.Box(x=stock_custom_box['Open'], name = 'Open Price',
                marker_color = 'lightseagreen'))
    fig_custom.add_trace(go.Box(x=stock_custom_box['High'], name='High',
                marker_color = 'Blue'))
    fig_custom.add_trace(go.Box(x=stock_custom_box['Low'], name = 'Low',
                marker_color = 'Yellow'))
    st.plotly_chart(fig_custom, use_container_width=True)

    custom_volume_chart = alt.Chart(stock_weekly.loc[(stock_weekly['Date'] >= start_date) & (stock_weekly['Date'] <= end_date)]).mark_line().encode(
        x=alt.X('Date').title('Date (Weekly)'),
        y=alt.Y('Volume', scale=alt.Scale(domain=[stock_weekly['Volume'].min(), stock_weekly['Volume'].max()])).scale(type='log')
    ).interactive()
    st.altair_chart(custom_volume_chart, use_container_width=True)

    
    




# Loading the LSTM model and making predictions
data = stock_daily[['Date','Close']]
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
model = pickle.load(open("webApp_code/model.pkl", 'rb'))


def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    return df

lookback = 7
shifted_df = prepare_dataframe_for_lstm(data, lookback)
shifted_df_as_np = shifted_df.to_numpy()
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]
X = dc(np.flip(X, axis=1))
split_index = int(len(X) * 0.95)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]
X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

test_predictions = model(X_test).detach().cpu().numpy().flatten()

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dc(dummies[:, 0])
dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = y_test.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dc(dummies[:, 0])

test_pred_data = np.array([new_y_test, test_predictions])
test_pred_data = np.transpose(test_pred_data)
test_pred_data = pd.DataFrame(test_pred_data, columns=['Actual Close', 'Predicted Close'])





with tab3:
    st.subheader("Prediction using LSTM (Long Short-Term Memory) model")
    st.write("I have used an LSTM model to predict the stock closing prices which recursively uses closing price data of last 7 days to make predictions. The line plot below with the actual and predicted closing prices shows the accuracy of the model for now.")
    fig = plt.figure()
    plt.plot(new_y_test, label='Actual Close')
    plt.plot(test_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    st.pyplot(fig)

    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(y=test_pred_data['Actual Close'], name='Actual Close',
                                  mode='lines',
                marker_color = 'indianred'))
    line_fig.add_trace(go.Scatter(y=test_pred_data['Predicted Close'], name = 'Predicted Close',
                                    mode='lines',
                marker_color = 'lightseagreen'))
   
    st.plotly_chart(line_fig, use_container_width=True)

    st.subheader("Stock predictions using rolling average and SARIMA")
    roling_window = st.slider('Select the rolling average window', value=7, min_value=3, max_value=14)
    forecast_steps = st.slider('Select the number of days from Dec 4 2023 for which you want to make the predictions', value=1, min_value=1, max_value=14)
    col1, col2 = st.columns(2)
    data = stock_daily['Close']
    
    rolling_average = data.rolling(roling_window).mean()
    with col1:
        st.write('Rolling average')
        st.write(rolling_average)
    rolling_average = rolling_average.dropna()

    # Fit SARIMA model on the rolling average
    order = (1, 0, 1)  # Example order, you may need to choose based on model diagnostics
    seasonal_order = (1, 1, 1, 12)  # Example seasonal order
    sarima_model = SARIMAX(rolling_average, order=order, seasonal_order=seasonal_order)
    sarima_fit = sarima_model.fit(disp=False)

    # Forecast future values
    
    forecast = sarima_fit.get_forecast(steps=forecast_steps) 
    forecast_index = pd.date_range(start=rolling_average.index[-1], periods=forecast_steps + 1, freq='B')[1:]  # Adjust frequency as needed
    forecast_values = forecast.predicted_mean.values

    forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast_values})
    with col2:
        st.write("Forecast")
        st.write(forecast_df['Forecast'])



