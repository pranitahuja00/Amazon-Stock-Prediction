import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from altair import datum
from datetime import *


st.title("Amazon Stock Analysis and Prediction")

tab1, tab2, tab3, tab4 = st.tabs(['Overview', 'Stock Analysis', 'Future Predictions', 'Conclusion'])

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
    st.write("")

with tab2:
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

    recent_weekly_price_box = alt.Chart(stock_daily.loc[(stock_daily['Date'] >= datetime(2022,12,1)) & (stock_daily['Date'] <= datetime(2023,12,5))]).mark_boxplot().encode(
        x = alt.Y('Close', scale=alt.Scale(domain=[stock_weekly['Close'].min(), stock_weekly['Close'].max()])).title('Closing Price')
    ).interactive().properties(
    width=300,
    height=200
    )
 
    st.altair_chart(alt.hconcat(recent_weekly_price,recent_daily_price), use_container_width=True)
    #st.altair_chart(recent_weekly_price_box)





    
    st.write("Interday stock movement in the past 150 days")
    movement_chart = alt.Chart(stock_daily.tail(150)).mark_line().encode(
        x='Date',
        y='Movement'
    ).interactive()
    st.altair_chart(movement_chart, use_container_width=True)

