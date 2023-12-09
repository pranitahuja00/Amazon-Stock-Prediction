import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import *
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose


st.title("Amazon Stock Analysis and Prediction")

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Intro and Overview','Stock Study and Analysis', 'Interactive Analysis', 'Future Predictions', 'About Me'])

stock_daily = pd.read_csv("https://raw.githubusercontent.com/pranitahuja00/Amazon-Stock-Prediction/main/data/AMZN_daily.csv")
stock_weekly = pd.read_csv("https://raw.githubusercontent.com/pranitahuja00/Amazon-Stock-Prediction/main/data/AMZN_weekly.csv")
stock_monthly = pd.read_csv("https://raw.githubusercontent.com/pranitahuja00/Amazon-Stock-Prediction/main/data/AMZN_monthly.csv")
stock_daily['Date']=pd.to_datetime(stock_daily['Date'])
stock_daily=stock_daily.drop('Adj Close', axis=1)
stock_weekly['Date']=pd.to_datetime(stock_weekly['Date'])
stock_weekly=stock_weekly.drop('Adj Close', axis=1)
stock_monthly['Date']=pd.to_datetime(stock_monthly['Date'])
stock_monthly=stock_monthly.drop('Adj Close', axis=1)


# TAB 1
with tab2:
    st.subheader("Stock Analysis")

    st.write("Monthly stock closing price over the years (Scale: Log):")
    close_price_chart = alt.Chart(stock_monthly).mark_line().encode(
        x=alt.Y('Date'),
        y=alt.Y('Close').scale(type='log').title('Closing Price')
    ).interactive()
    st.altair_chart(close_price_chart, use_container_width=True)
    st.write("The above chart shows the overall closing stock price trends of Amazon on a monthly basis (Each data point refers to the closing price of the last day of each month) from May 1997 (When the company went public) to Dec 2023. Amazon stock first boomed during the late 90s which is also known as the dot com bubble after which we can notice a general upward trend in the stock price with some significant upward and downward movements in between. The first major downward trend can be seen in the year 2000 (The early 2000s recession) followed by a significant fall in 2008 (The Great Recession). The positive effect of the covid pandemic IT boom can also be seen in the chart as the stock price goes up in the second half of 2020 till 2021 which was followed by the post pandemic recession leading to a downward trend from the second quarter of 2022.")

    #Seasonality and Quarterly averages
    tab_seasonality, tab_quarterly = st.tabs(['Stock Closing Price Seasonality', 'Quarterly Averages'])
    with tab_seasonality:
        st.write("Checking seasonality trends from 2014 - 2023 using decomposition:")
        temp_monthly = stock_monthly.tail(120).set_index('Date')
        decomposed_series = seasonal_decompose(temp_monthly['Close'])
        fig_original_sea = go.Figure()
        fig_original_sea.add_trace(go.Scatter(x=temp_monthly.index, y=temp_monthly['Close'], mode='lines', name='Original'))
        fig_original_sea.add_trace(go.Scatter(x=temp_monthly.index, y=decomposed_series.seasonal, mode='lines', name='Seasonality'))
        fig_original_sea.update_layout(title='Amazon Stock Seasonality',
                  xaxis_title='Date',
                  yaxis_title='Seasonality')
        st.plotly_chart(fig_original_sea)
        st.write("Taking a closer look at the stock seasonality")
        fig_sea = go.Figure()
        fig_sea.add_trace(go.Scatter(x=temp_monthly.index, y=decomposed_series.seasonal, mode='lines', name='Seasonality'))
        fig_sea.update_layout(
                  xaxis_title='Date',
                  yaxis_title='Seasonality')
        st.plotly_chart(fig_sea)
        st.write("The chart shows the yearly pattern that the Amazon stock prices tend to follow. If the market conditions remain stable and are not adversely affected by some external events or recessions then one can expect to buy stocks at a very low price in around January or February and sell at yearly high prices in the middle of the year which would be around June or July leading to a good profit. Seasonality can often help in predicting such yearly trends which can help investors with their decisions.")

    with tab_quarterly:
        st.write("Stock price quarterly averages from 2014 - 2023:")
        temp_monthly['Quarter'] = temp_monthly.index.quarter
        temp_monthly['Year'] = temp_monthly.index.year
        quarter = temp_monthly.groupby(["Year", "Quarter"])["Close"].mean().unstack()
        quarterly_heatmap = px.imshow(quarter, text_auto=True, aspect="auto")
        st.plotly_chart(quarterly_heatmap)
        st.write("Recent quarterly averages show that the first and last quarters are most suitable to pick up some stocks at lower prices and the stock price rises up during the other two quarters.")

    # Stock volume
    st.write("Monthly stock trade volume over the years")
    volume_chart = alt.Chart(stock_monthly).mark_line().encode(
        x='Date',
        y=alt.Y('Volume').scale(type='log')
    ).interactive()
    st.altair_chart(volume_chart, use_container_width=True)
    st.write("Stock trade volume tells the amount of stocks traded between two parties in a specific amount of time. It is oftenly used as a measure of stock popularity in the market. In case of Amazon, stock trade volume is showing a general downward trend with significant ups and downs in between. Dot Com Bubble (Late 90s) era proved to be the time when Amazon stocks boomed in popularity along with price with significant ups and downs in between followed by an all time low in the recent months.")

    # Dec 2022 - Dec 2023 Stock analysis
    st.write("Comparing the last two years:")
    recent_weekly_price23 = alt.Chart(stock_weekly.loc[(stock_weekly['Date'] >= datetime(2022,12,1)) & (stock_weekly['Date'] <= datetime(2023,12,5))], title="Dec 2022 - Dec 2023").mark_line().encode(
        x=alt.X('Date').title('Date (Weekly)'),
        y=alt.Y('Close', scale=alt.Scale(domain=[stock_weekly['Close'].min(), stock_weekly['Close'].max()])).title('Closing Price')
    ).interactive().properties(
    width=300,
    height=200
    )
    recent_weekly_price22 = alt.Chart(stock_weekly.loc[(stock_weekly['Date'] >= datetime(2021,12,1)) & (stock_weekly['Date'] <= datetime(2022,12,5))], title="Dec 2021 - Dec 2022").mark_line().encode(
        x=alt.X('Date').title('Date (Weekly)'),
        y=alt.Y('Close', scale=alt.Scale(domain=[stock_weekly['Close'].min(), stock_weekly['Close'].max()])).title('Closing Price')
    ).interactive().properties(
    width=300,
    height=200
    )
    stock_daily_21_22 = stock_daily.loc[(stock_daily['Date'] >= datetime(2021,12,1)) & (stock_daily['Date'] <= datetime(2022,12,5))]
    fig2 = go.Figure()
    fig2.add_trace(go.Box(x=stock_daily_21_22['Close'], name='Close Price',
                marker_color = 'indianred', boxmean=True))
    fig2.add_trace(go.Box(x=stock_daily_21_22['Open'], name = 'Open Price',
                marker_color = 'lightseagreen', boxmean=True))
    fig2.add_trace(go.Box(x=stock_daily_21_22['High'], name='High',
                marker_color = 'lightblue', boxmean=True))
    fig2.add_trace(go.Box(x=stock_daily_21_22['Low'], name = 'Low',
                marker_color = 'Yellow', boxmean=True))
    fig2.update_layout(title="Dec 2021 - Dec 2022")
    stock_daily_22_23 = stock_daily.loc[(stock_daily['Date'] >= datetime(2022,12,1)) & (stock_daily['Date'] <= datetime(2023,12,5))]
    fig = go.Figure()
    fig.add_trace(go.Box(x=stock_daily_22_23['Close'], name='Close Price',
                marker_color = 'indianred', boxmean=True))
    fig.add_trace(go.Box(x=stock_daily_22_23['Open'], name = 'Open Price',
                marker_color = 'lightseagreen', boxmean=True))
    fig.add_trace(go.Box(x=stock_daily_22_23['High'], name='High',
                marker_color = 'lightblue', boxmean=True))
    fig.add_trace(go.Box(x=stock_daily_22_23['Low'], name = 'Low',
                marker_color = 'Yellow', boxmean=True))
    fig.update_layout(title="Dec 2022 - Dec 2023")
    
    st.altair_chart(alt.hconcat(recent_weekly_price22, recent_weekly_price23), use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig, use_container_width=True)

    st.write("The above charts represent the stock price trends from in the last two years. These plots indicate that Amazon stocks have went down significantly during the 2021 - 2022 year from around 178 USD to an annual low of 86.14 USD. The next year (2022 - 2023) witnessed a significant rising trend with some small variations in between. The stock price in the last year ranged from an all time low of 81.43 USD all the way to an all time high of 149.26 USD with median opening and closing prices of 123.01 USD and 121.166 USD and both their averages sitting at around 116.5 USD")

with tab3:
    st.subheader("Interactive Analysis")
    st.write("Stock Price and trade volume over custom time period:")
    st.write("Select a minimum time period of 2 years to view seasonality and quarterly averages as well")
    start_date = st.slider('Select the starting date', value = datetime(1997,5,20), min_value = datetime(1997,5,20), max_value = datetime(2023,11,3))
    end_date = st.slider('Select the ending date', value = datetime(1997,6,20), min_value = datetime(1997,6,20), max_value = datetime(2023,12,3))

    custom_button = st.button("Click to generate")
    if custom_button:
        custom_price_chart = alt.Chart(stock_weekly.loc[(stock_weekly['Date'] >= start_date) & (stock_weekly['Date'] <= end_date)]).mark_line().encode(
            x=alt.X('Date').title('Date (Weekly)'),
            y=alt.Y('Close', scale=alt.Scale(domain=[stock_weekly['Close'].min(), stock_weekly['Close'].max()])).title('Closing Price').scale(type='log')
        ).interactive().properties(
            width=200,
            height=150
        )
    
        custom_volume_chart = alt.Chart(stock_weekly.loc[(stock_weekly['Date'] >= start_date) & (stock_weekly['Date'] <= end_date)]).mark_line().encode(
            x=alt.X('Date').title('Date (Weekly)'),
            y=alt.Y('Volume', scale=alt.Scale(domain=[stock_weekly['Volume'].min(), stock_weekly['Volume'].max()])).scale(type='log')
        ).interactive().properties(
            width=300,
            height=150
        )

        st.altair_chart(alt.hconcat(custom_price_chart, custom_volume_chart), use_container_width= True)

        stock_custom_box = stock_daily.loc[(stock_daily['Date'] >= start_date) & (stock_daily['Date'] <= end_date)]

        fig_custom = go.Figure()
        fig_custom.add_trace(go.Box(x=stock_custom_box['Close'], name='Close Price',
                    marker_color = 'indianred', boxmean=True))
        fig_custom.add_trace(go.Box(x=stock_custom_box['Open'], name = 'Open Price',
                    marker_color = 'lightseagreen', boxmean=True))
        fig_custom.add_trace(go.Box(x=stock_custom_box['High'], name='High',
                    marker_color = 'lightblue', boxmean=True))
        fig_custom.add_trace(go.Box(x=stock_custom_box['Low'], name = 'Low',
                    marker_color = 'Yellow', boxmean=True))
        st.plotly_chart(fig_custom, use_container_width=True)

        #Seasonality and Quarterly averages
        temp_monthly_custom = stock_monthly.loc[(stock_monthly['Date'] >= start_date) & (stock_monthly['Date'] <= end_date)].set_index('Date')
        if(temp_monthly_custom.shape[0]>=24):    
            tab_seasonality_custom, tab_quarterly_custom = st.tabs(['Stock Closing Price Seasonality', 'Quarterly Averages'])
            with tab_seasonality_custom:
                st.write("Checking seasonality trends from ",start_date," - ",end_date,":")        
                decomposed_series_custom = seasonal_decompose(temp_monthly_custom['Close'])
                fig_original_sea_custom = go.Figure()
                fig_original_sea_custom.add_trace(go.Scatter(x=temp_monthly_custom.index, y=temp_monthly_custom['Close'], mode='lines', name='Original'))
                fig_original_sea_custom.add_trace(go.Scatter(x=temp_monthly_custom.index, y=decomposed_series_custom.seasonal, mode='lines', name='Seasonality'))
                fig_original_sea_custom.update_layout(title='Amazon Stock Seasonality',
                          xaxis_title='Date',
                          yaxis_title='Seasonality')
                st.plotly_chart(fig_original_sea_custom)
                st.write("Taking a closer look at the stock seasonality")
                fig_sea_custom = go.Figure()
                fig_sea_custom.add_trace(go.Scatter(x=temp_monthly_custom.index, y=decomposed_series_custom.seasonal, mode='lines', name='Seasonality'))
                fig_sea_custom.update_layout(
                          xaxis_title='Date',
                          yaxis_title='Seasonality')
                st.plotly_chart(fig_sea_custom)

            with tab_quarterly_custom:
                st.write("Stock price quarterly averages from ",start_date," - ",end_date,":")
                temp_monthly_custom['Quarter'] = temp_monthly_custom.index.quarter
                temp_monthly_custom['Year'] = temp_monthly_custom.index.year
                quarter_custom = temp_monthly_custom.groupby(["Year", "Quarter"])["Close"].mean().unstack()
                quarterly_heatmap_custom = px.imshow(quarter_custom, text_auto=True, aspect="auto")
                st.plotly_chart(quarterly_heatmap_custom)

    



with tab4:
    st.subheader("Stock predictions using rolling average and SARIMA")
    roling_window = st.slider('Select the rolling average window', value=7, min_value=3, max_value=14)
    forecast_steps = st.slider('Select the number of days from Dec 4 2023 for which you want to make the predictions', value=1, min_value=1, max_value=14)
    sarima_predict = st.button("Click to predict (! Will take some time !)")
    col1, col2 = st.columns(2)
    data = stock_daily['Close']
    
    rolling_average = data.rolling(roling_window).mean()

    with col1:
        st.write('Rolling average')
        st.write(rolling_average)
    
    rolling_average = rolling_average.dropna()

    if sarima_predict:
        # Fit SARIMA model on the rolling average
        order = (1, 0, 1)
        seasonal_order = (1, 1, 1, 12)
        sarima_model = SARIMAX(rolling_average, order=order, seasonal_order=seasonal_order)
        sarima_fit = sarima_model.fit(disp=False)

        # Forecast future values
        forecast = sarima_fit.get_forecast(steps=forecast_steps) 
        forecast_index = pd.date_range(start=rolling_average.index[-1], periods=forecast_steps + 1, freq='B')[1:]
        forecast_values = forecast.predicted_mean.values
        forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast_values})

        with col2:
            st.write("Forecast")
            st.write(forecast_df['Forecast'])

        #forecast_fig = go.Figure()
        #forecast_fig.add_trace(go.Line(x=))
