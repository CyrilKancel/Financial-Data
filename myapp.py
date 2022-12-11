# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:49:23 2022

@author: ckancel
"""
#reference: https://github.com/sumanthsripada/Financial-Analysis---Yahoo-Finance/blob/main/findash.py

import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import yahoo_fin.stock_info as si
from annotated_text import annotated_text
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas_datareader.data as web

#==============================================================================
# Main body
#==============================================================================

# --- Title ---

# Add dashboard title and description
st.title("My simple financial dashboard")
st.write("Data source: Yahoo Finance")

# --- Insert an image ---

image = Image.open('C:/Users/ckancel/Desktop\MBD/Financial Programing/Section 4/streamlit/img/stock_market.jpg')
st.image(image, caption='Stock market')

# --- Multiple choices box ---

# Get the list of stock tickers from S&P500
ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

# Add multiple choices box
tickers = st.multiselect("Ticker(s)", ticker_list)
st.write(tickers)

# --- Select date time ---

# Add select begin-end date
col1, col2 = st.columns(2)  # Create 2 columns
start_date = col1.date_input("Start date", datetime.today().date() - timedelta(days=30))
end_date = col2.date_input("End date", datetime.today().date())

# --- Add a button ---

get = st.button("Get data", key="get")

    #selection box


# --- Table to show data ---

# Add table to show stock data
# This function get the stock data and save it to cache to resuse
#source https://github.com/sumanthsripada/Financial-Analysis---Yahoo-Finance/blob/main/findash.py
def tab1():


    
    annotated_text(("Yahoo","Stock Summary","#3498DB"))
    col1,col2 = st.columns([2,2])
    # Add table to show stock data
    
    select_Period = ['-','1mo', '3mo','6mo','ytd','1y','2y','5y','max']
    default  = select_Period.index('1y')
    select_Period =  st.selectbox('Select Period', select_Period,index = default)
    ticker = st.sidebar.selectbox("Select a ticker", ticker_list,index = default)
    
    @st.cache
    def GetSummary(tickers):
        return si.get_quote_table(ticker,dict_result = False)
    @st.cache
    def GetStockData(tickers, start_date, end_date):
        return pd.concat([si.get_data(tick, start_date, end_date) for tick in tickers])
        
    if ticker != '-':   
          data = yf.download(ticker, period = select_Period)
          

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.07, subplot_titles=('Stock Trend', 'Volume'), 
               row_width=[0.2, 0.7])
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'],name="Stock Trend",showlegend=True,fill='tozeroy'),row= 1,col = 1)
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'],name="Volume",showlegend=True), row=2,col = 1)
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(title="Stock Summary Plot", yaxis_title="Close Price")
    fig.update_layout(width = 1000 , height = 600)
    st.plotly_chart(fig)
          
    if ticker != '-':
	    Summary = GetSummary(ticker)
	    Summary = Summary.set_index('attribute')
	    Summary["value"] = Summary["value"].astype(str)
	    col1.dataframe(Summary, height = 1000)
        
    
    
    @st.cache
    def convert_df_to_csv(df):
        return df.to_csv().encode('utf-8')
    st.download_button(label="Download Summary",data=convert_df_to_csv(Summary),file_name='StockSummary.csv',mime='text/csv',)

#==============================================================================
# Tab 3 - Statistics
#source https://github.com/sumanthsripada/Financial-Analysis---Yahoo-Finance/blob/main/findash.py
#==============================================================================       
def tab2():
     
#Dashboard Header 
    annotated_text(("Stock","Statistics","#3498DB"))
    
    
# Getting stock data
    def GetStatsEval(ticker):
        select_Period = ['-','1mo', '3mo','6mo','ytd','1y','2y','5y','max']
        default  = select_Period.index('1y')
        ticker = st.sidebar.selectbox("Select a ticker", ticker_list,index = default)
        return si.get_stats_valuation(ticker)
    def GetStats(ticker):
        select_Period = ['-','1mo', '3mo','6mo','ytd','1y','2y','5y','max']
        default  = select_Period.index('1y')
        ticker = st.sidebar.selectbox("Select a ticker", ticker_list,index = default)
        return si.get_stats(ticker)
    
    
    def convert_df_to_csv(df):
        return df.to_csv().encode('utf-8')
    
    
    if ticker != '-':
        statsval = GetStatsEval(ticker)
        statsval = statsval.rename(columns={0:'Valuation Measures',1:'USD'})
        
        #Valuation Measures
        annotated_text(("VALUATION","MEASURES","#3498DB"))
        st.dataframe(statsval,height = 1000)
    #Get Remaining stats
    if ticker != '-':
        stat = GetStats(ticker)
        stat = stat.set_index('Attribute')
        
        #stock Price History
        annotated_text(("STOCK PRICE","HISTORY","#3498DB"))
        Sph = stat.iloc[0:7,]
        st.dataframe(Sph,height = 1000)
        
        #share statistics
        annotated_text(("SHARE","STATISTICS","#3498DB"))
        Shs = stat.iloc[7:19,]
        st.dataframe(Shs,height = 1000)
        
        #Dividend & Splits
        annotated_text(("DIVIDEND","SPLITS","#3498DB"))
        Div = stat.iloc[19:29,]
        st.table(Div)
        
        #Financial Highlights
        annotated_text(("FINANCIAL","HIGHLIGHTS","#3498DB"))
        Finh = stat.iloc[29:31,]
        st.table(Finh)
        
        #Profitability
        annotated_text(("STOCK","PROFITABILITY","#3498DB"))
        Prof = stat.iloc[31:33,]
        st.dataframe(Prof,height = 1000)
        
        #Management Effectiveness
        annotated_text(("Management","Effectiveness","#3498DB"))
        Meff = stat.iloc[33:35,]
        st.dataframe(Meff,height = 1000)
        
        #Income Statement
        IncS = stat.iloc[35:43,]
        annotated_text(("INCOME","STATEMENT","#3498DB"))
        st.dataframe(IncS,height = 1000)
        
        #Balance Sheet
        annotated_text(("BALANCE","SHEET","#3498DB"))
        BalS = stat.iloc[43:49,]
        st.dataframe(BalS,height = 1000)
        
        #Cash Flow
        annotated_text(("CASH","FLOW","#3498DB"))
        Caf = stat.iloc[49:51,]
        st.dataframe(Caf,height = 1000)


def tab3():
    # Setup the Monte Carlo simulation
    stock_price = web.DataReader('AAPL', 'yahoo', start_date, end_date)
    close_price = stock_price['Close']
    np.random.seed(123)
    simulations = 1000
    time_horizone = 200
    
    # Run the simulation
    simulation_df = pd.DataFrame()
    
    for i in range(simulations):
        
        # The list to store the next stock price
        next_price = []
        
        # Create the next stock price
        last_price = close_price[-1]
        
        for j in range(time_horizone):
            # Generate the random percentage change around the mean (0) and std (daily_volatility)
            future_return = np.random.normal(0, daily_volatility)
    
            # Generate the random future price
            future_price = last_price * (1 + future_return)
    
            # Save the price and go next
            next_price.append(future_price)
            last_price = future_price
        
        # Store the result of the simulation
        next_price_df = pd.Series(next_price).rename('sim' + str(i))
        simulation_df = pd.concat([simulation_df, next_price_df], axis=1)


def run():
                      
    #radio box to select the tabs 
    select_tab = st.sidebar.selectbox("Select tab", ['Summary', 'Statistics', 'Monte Carlo Simulation'])
    
    #Display the selected tab
    if select_tab == 'Summary':
        tab1()
        
    elif select_tab == 'Statistics':
        tab2()
    
    elif select_tab == 'Monte Carlo Simulation':
        tab3()
    

run()
        
        