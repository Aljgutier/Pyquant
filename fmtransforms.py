import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timedelta
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import yfinance as yf
import time
import requests, pandas, lxml
from lxml import html
from fredapi import Fred
import pandas as pd
import quandl

"""
.. module:: fmtransforms.py
    :Python version 3.7 or greater
    :synopsis: transformations to support market financial analytics

.. moduleauthor:: Alberto Gutierrez <aljgutier@yahoo.com>
"""




def fmjoinff(dfleft,dfright,market_cols=['Close'],verbose=False,dropnas=True):
    """
    Financial Time Series Join Fill Forward

    **Parameters**:
        dfmarket (dataframe): of X, independent variable colums.
        dfjoin (dataframe): data frame to join
        joincols (String):  No. of training days for each prediction. 
        market_variable (String):  test start time
        verbose (boolean):  test end time
        dropna (boolean): adfsdaf
        *args**: variable lenth arguement list
        **kwargs**: arbitrary keword arcuents

    **Returns**:
        dataframe  columns include all columns in the    


    **To Do**:
        * Artwork - showing sliding window train and test with sliding window
        * Kwargs: adslfjsdaf  ... 
        * HowItWorks: asdfjsadfj 

    How does it work?

    print public_fn_with_googley_docstring(name=foo, state=None)

    BTW, this always returns 0.  **NEVER** use with :class: MyPublicClass.

    |

    """

    # to ML dataframe


    # Step by Step
    #   first do an outer join ...because GDP date could be a non-market date, otherwise join may fail
    #   transform gdp variables 
    #   fill forward
    #   remove any non market dates ... dates corresponding to close_price null/NaN


    dfjoined=dfleft.join(dfright,lsuffix='_left',rsuffix='_right', how='outer')


    for v in dfright.columns:  # check the new columns added form the right
        if (not is_numeric_dtype(dfjoined[v])):
            if verbose:
                print('... the variable',v,' is not numeric, convert to numeric')
            dfjoined[v] = pd.to_numeric(dfjoined[v],errors='coerce')


    # fill forward 
    dfjoined = dfjoined.ffill() 

    # Nulls
    # tail rows with null close_price ... verify that these are non market dates
    if verbose:
        print('Nulls:',dfjoined[dfjoined.isnull()].tail(5))
        
   
   # drop non market dates
    if dropnas:
        if verbose:
            print('\nDrop non market dates, NAs in Market Columns... ',market_cols)
        dfjoined = dfjoined.dropna(subset=market_cols) 


    if verbose:
        print('\nNulls =\n' ,dfjoined.isnull().sum())
        # Display the Dataframe
        display(dfjoined.tail(3))

    return dfjoined




# ADX - average directional index
# Directional Movement System developed by J. Willes Welder
# determines if the market is trending, which direction (PDI or NDI), and strength of the trend (ADX)
# Three Indicators
#    +DI how many moves up ... positive indicator ("PDI")
#    -DI how many moves down ... negative Indirector ("NDI"
#    ADX strength of the trend
# Reference
# https://www.google.com/search?q=python+calculating+adx&rlz=1C5CHFA_enUS819US819&oq=python+calculating+adx+&aqs=chrome..69i57j0j69i64.5444j0j1&sourceid=chrome&ie=UTF-8#kpvalbx=_7kieXpTbINLQ-gSFkJOwAQ34
#
#
# https://www.investopedia.com/ask/answers/112814/how-average-directional-index-adx-calculated-and-what-formula.asp
#    moveUp = Today high  - Yesterday High
#    moveDown = Yesterday low  - Today low 
#    if moveUp is greater than zero and moveDown: ... Positive Direction Move, PDM, moveUp 
#       PDM = moveUP, 
#    else:
#       PDM = 0 
#    if moveDown is greater than zero and moveUP: Negative Direction Move (NDM) moveDown, 
#       NDM = 1
#    else:
#       NDM = 0
#  Positive and negative PDI and NDI # assume a window 14 days 
#   PDI = 100 x  14EMA(PDM))/ATR 
#   NDI = 100 x  14EMA(NDM))/ATR 
#
# assume a window of 14 days
#  ADX Calculation 
#  ADX = 100 x 14 EMA abs(PDI - NDI) / (PDI + NDI) ... smoothed and normalized


# true range 
# what is it? difference between absolute highest and lowest price in a one day time period
#  J. Welles Welder
#  https://www.youtube.com/watch?v=53fnUG6Dvmw
# true range is the largest of 
#    todays high - low
#    abs(high - yday_close)
#    abs(low -yday_close)
def truerange(todayHigh,todayLow,yesterdayClose):

    x1 = todayHigh - todayLow
    x2 = np.abs(todayHigh - yesterdayClose)
    x3 = np.abs(todayLow - yesterdayClose)


    if (x1 > x2) and (x1 > x3):
        largest = x1
    elif (x2 > x1) and (x2 > x3):
        largest = x2
    else:
        largest = x3
    trueRange = np.abs(largest)
    return trueRange

# positive direction move, PDM
def PDM(moveUp,moveDown):
    PDM=0
    if (moveUp > 0) and (moveUp > moveDown):
        PDM=moveUp
    return PDM

# negative direction move, NDM
def NDM(moveUp,moveDown):
    NDM=0
    if (moveDown > 0) and (moveDown > moveUp):
        NDM=moveDown
    return NDM


def dfsma(df,v,windows):
    
    if not isinstance(windows, list):
        windows=[windows]
    
    for w in windows:
        v_sma = v + '_sma' + str(w)
        df[v_sma] = df[v].rolling(window=w).mean()
    
    return df

def dfrma(df,var1,var2, varname='rma'):
    # for example, inputs should ma50 and ma200
    # when the relative difference 50-day ma is greater than 200-day then the stock is bullish
    df[varname] =   ( df[var1]  -  df[var2] ) / df[var2]
    return df

def dfema(df,v,windows):
    
    if not isinstance(windows, list):
        windows=[windows]
    
    for w in windows:
        v_ema = v + '_ema' + str(w)
        df[v_ema] = df[v].ewm(span=w,min_periods=0,adjust=False,ignore_na=False).mean()
    return df


def dfnma(df,variables,windows):
    
    if not isinstance(windows, list):
        windows=[windows]

    if not isinstance(variables, list):
        variables=[variables]
    
    for w in windows:
        for v in variables:
            v_nma = v + '_nma' + str(w)
            df[v_nma] = (df[v] /df[v].shift(1) -1).rolling(window=w).mean()
    
    return df




# Variance Volitility Measures
# https://blog.quantinsti.com/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/
def dflogretstd(df,v,windows):
    df['logret'] = np.log(df[v] / df[v].shift(1))

    if not isinstance(windows, list):
        windows=[windows]
    
    for w in windows:
        v_lrstd = v + '_lrstd' + str(w)
        df[v_lrstd] = df['logret'].rolling(window=w).std()

    df.drop('logret',axis=1,inplace=True)
    return df


#def dfstd(df,v,windows):

#    if not isinstance(windows, list):
#        windows=[windows]
    
#    for w in windows:
#        v_std = v + '_std' + str(w)
#        df[v_std] = df[v].rolling(window=w).std()

        # * np.sqrt(w)

    # returns new columns with v_stdw
#    return df


def dfvelocity(df,v,windows=1):

    if not isinstance(windows, list):
        windows=[windows]


    v_d1 = str(v)+'_d1'
    df[v_d1] = df[v]-df[v].shift(1) # 1st difference


    for w in windows:
        v_avgvel = v + '_avgvel' + str(w)
        if w == 1:
            df[v_avgvel] = df[v_d1]
        else:
            df[v_avgvel] = df[v_d1].rolling(window=w).mean()

        # * np.sqrt(w)
    df.drop(v_d1,axis=1,inplace=True)
    # returns new columns with v_stdw
    return df   


def dfadx(df,v_todayClose,v_todayHigh,v_todayLow,window=14):
    v_yesterdayClose='yesterday_closePrice'
    v_yesterdayHigh='yesterday_HighPrice'
    v_yesterdayLow='yesterday_lowPrice'
    
    df[v_yesterdayClose]=df[v_todayClose].shift(1)
    df[v_yesterdayHigh]=df[v_todayHigh].shift(1)
    df[v_yesterdayLow]=df[v_todayLow].shift(1)
    
    #window=14
    #variable=v_todayClose
    #df['EMA14'] = dftmp[variable].ewm(span=window,min_periods=0,adjust=False,ignore_na=False).mean()

    # Variables 
    pdi = 'PDI' + str(window)
    ndi = 'NDI' + str(window)
    pdm = 'PDM' + str(window)
    ndm = 'NDM' + str(window)
    atr = 'ATR' + str(window)
    emaPDM = 'ema' + pdm
    emaNDM = 'ema' + ndm
    
    variable='truerange'

    df[variable]=df.apply(lambda row: truerange(row[v_todayHigh],
                    row[v_todayLow], row[v_yesterdayClose]), axis=1)

    df[atr] = df[variable].ewm(span=window,min_periods=0,adjust=False,ignore_na=False).mean()

    df['moveUp'] = df[v_todayHigh] - df[v_yesterdayHigh]
    df['moveDown'] = df[v_yesterdayLow] - df[v_todayLow]

    df[pdm]= df.apply( lambda row: PDM(row['moveUp'],row['moveDown']),axis=1)
    df[ndm]= df.apply( lambda row: NDM(row['moveUp'],row['moveDown']),axis=1)
    
    
    variable=pdm
    df[emaPDM] = df[variable].ewm(span=window,min_periods=0,adjust=False,ignore_na=False).mean()
    variable=ndm
    df[emaNDM] = df[variable].ewm(span=window,min_periods=0,adjust=False,ignore_na=False).mean()
    
    df[pdi] = 100 * df[emaPDM]/df[atr]
    df[ndi] = 100 * df[emaNDM]/df[atr]
    
    # ADX = 100 * EMA14(np.abs(PDI-NDI))/(1.0*(PDI+NDI))
    df['absPDImNDI']= np.abs(df[pdi]-df[ndi])
    
    variable='absPDImNDI'
    df['ema_abs_PDImNDI']=df[variable].ewm(span=window,min_periods=0,adjust=False,ignore_na=False).mean()
    
    df['ADX'] = 100 * df['ema_abs_PDImNDI']/(df[pdi]+df[ndi])
    
    drop_cols=['truerange', pdm,ndm,atr,emaPDM,emaNDM,'moveDown','moveUp','ema_abs_PDImNDI', 
    'absPDImNDI','yesterday_HighPrice','yesterday_closePrice', 'yesterday_lowPrice']
    df.drop( drop_cols, axis=1,inplace=True)
    return df


# the percent normalized return from the previous reported period (i.e., v.shfit(1))
def period_percent_change(df,v,new_variable_name = ''):
    if new_variable_name == '':
        new_variable_name = v + '_r'
    df[new_variable_name] = df[v]/df[v].shift(1) - 1
    return df

# GDP Functions
def _gdp_recession1q(gdpqoq):
    if (gdpqoq < 0):
        return 1
    else:
        return 0

def _gdp_recession2q(gdpqoq, gdpqoqprev):
    if ((gdpqoq < 0) and (gdpqoqprev < 0)):
        return 1
    else:
        return 0

def gdprecession(df, vn_gdp):
    df['gdp_prevq']=df[vn_gdp].shift(1)
    #df['gdp_qoq']=100*(df[vn_gdp]/df['gdp_prevq']-1)
    df = period_percent_change(df,vn_gdp,new_variable_name = 'gdp_qoq')
    df['gdp_prevqoq']=df['gdp_qoq'].shift(1)
    df['recession1q']=df.apply(lambda row: _gdp_recession1q(row['gdp_qoq']),axis=1)
    df['recession2q']=df.apply(lambda row: _gdp_recession2q(row['gdp_qoq'],row['gdp_prevqoq']),axis=1)
    df.drop(['gdp_prevq','gdp_prevqoq'],inplace=True,axis=1)
    return df




# Market Low Anticipation and low to high Reaction
def _market_sell_anticipation(row,df):
    # days before market low to get back in market
    
    Anticipation_Low=df.loc[row['Date_Sell']:row['Date_Mkt_Low']].index.size -1
    
    return Anticipation_Low

def _market_buy_anticipation(row,df):
    # days from market low to get back in market
    
    Reaction_LowHigh=df.loc[row['Date_Mkt_Low']:row['Date_Buy']].index.size -1
    
    return Reaction_LowHigh

def _days_out_of_market(row,df):
    # days out of market
    
    out_of_market=df.loc[row['Date_Sell']:row['Date_Buy']].index.size -1
    return out_of_market


def market_anticipation(df,p_1='p_s_1', Close='Close',mkt='mkt', t1=dt.datetime(1960,1,1),
                        verbose=False):
    # 
    # anticipataton of market after market prediction
    # sell date ... prior to market bottom
    # buy date ... after market bottom
    

    p=p_1 + '_m1'
    mkt_1 = 'mkt_1'
    a10 = 'a10'
    a01 = 'a01'
    m01 = 'm01'
    m10 = 'm10'
    dfa = df[[Close,mkt,p_1]].copy() 
    dfa[p] = dfa[p_1].shift(-1)
    dfa['mkt_1'] = dfa['mkt'].shift(-1)

    # ps = 0,1  down market, up market ... smoothed prediction
    # mkt = -1, 1  down market, up market

    #create variables for anticipation of down and up markets
    dfa[m10] = (dfa[mkt] == 1 ) & (dfa[mkt_1]==-1)
    dfa[a10] = (dfa[p] == 1 ) & (dfa[p_1]==0)
    dfa[m01] = (dfa[mkt] == -1 ) & (dfa[mkt_1]==1)
    dfa[a01] = (dfa[p] == 0 ) & (dfa[p_1]==1)

    dfa = dfa[[Close,mkt,mkt_1,p,p_1,m10,m01,a01,a10]] # order columns for convenience
    dfa=dfa.loc[t1:] # start on up market 1960 by default
    

    # keep only market changes and anticipation changes
    dfa['keep'] = (dfa[m10] | dfa[a10] | dfa[m01] | dfa[a01] )
    dfa = dfa[dfa['keep']]
    dfa.drop('keep',axis=1,inplace=True)
    

    
    if verbose == True:
        print('dfa with market anticipation variables')
        display(dfa.head())
        
    # keep days where mkt -1 -> 1, mkt 1 > -1, ps  0 -> 1 , ps 1 -> 0
    
    # Consolidate Hith to Low to High Market ... 4 rows into 1
    # rows come in groups of 4 ... 
    #  1 mkt turns from Bull to Bear
    #  2 ps (smooth) prediction turns from Bull to Bear
    #  3 mket turns from Bear to Bull
    #  4 ps (smooth) prediction turns fro Bear to BUll


    dfa = dfa.reset_index()
    dfa['Date_Mkt_High']=dfa['date'] # market high at close of business
    dfa['Date_Sell']=dfa['date'].shift(-1)   # sell signal close of bus prediction for next day ... out of market by next morn
    dfa['Date_Mkt_Low']=dfa['date'].shift(-2) # market low at close of business
    dfa['Date_Buy']=dfa['date'].shift(-3)  # buy signal forward predict for next day, buy by morning

    dfa['Mkt_High_Price'] = dfa['Close']
    dfa['Sell_Price'] = dfa['Close'].shift(-1)
    dfa['Mkt_Low_Price'] = dfa['Close'].shift(-2)
    dfa['Buy_Price'] = dfa['Close'].shift(-3)
    dfa['Gain-Loss'] = dfa['Sell_Price'] - dfa['Buy_Price']
    dfa['Buy_Price'] = dfa['Close'].shift(-3)
    dfa['Gain_Loss'] = dfa['Sell_Price'] - dfa['Buy_Price']
    dfa['Percent_Gain_Loss'] = dfa['Gain_Loss'] / dfa['Sell_Price']

    dfa = dfa[(dfa[mkt]==1 ) & (dfa[mkt_1]==-1)] # keep only rows with a market cycle

    if verbose == True:
        print('market cycles with anticipation points - market high, sell (get out of market) , market low, buy - (get back into market)')
        display(dfa)
    
    
    # sell signal - days before market low
    # buy signal - days from market low

    dfa['Anticipation_Sell'] = dfa.apply(lambda row: _market_sell_anticipation(row,df), axis=1 )
    dfa['Reaction_Buy'] = dfa.apply(lambda row: _market_buy_anticipation(row,df), axis=1 )
    dfa['Out_Of_Market'] = dfa.apply(lambda row: _days_out_of_market(row,df), axis=1 )


    cols=['Date_Mkt_High', 'Date_Sell','Date_Mkt_Low','Date_Buy','Mkt_High_Price','Sell_Price','Mkt_Low_Price','Buy_Price',
          'Gain_Loss','Percent_Gain_Loss', 'Anticipation_Sell','Reaction_Buy','Out_Of_Market' ]
   
   
    dfa=dfa[cols]
    dfa=dfa.reset_index(drop=True)
    
    return dfa

def bear_buysell_summary(df_anticipation, date_mkt_high, 
                         date_market_high_col = 'Date_Mkt_High',
                         market_high_price_col = 'Mkt_High_Price',
                         date_sell_col = "Date_Sell",
                         sell_price_col = "Sell_Price",
                         date_market_low_col ='Date_Mkt_Low',
                         market_low_price_col = 'Mkt_Low_Price',
                         date_buy_col = 'Date_Buy',
                         buy_price_col = 'Buy_Price',
                         anticipation_sell_col = 'Anticipation_Sell',
                         reaction_buy_col = 'Reaction_Buy'
                        ):
    
    df_mcycle = df_anticipation[df_anticipation[date_market_high_col]==date_mkt_high]
    
    d1=df_mcycle[date_market_high_col].values[0]
    d1=pd.to_datetime(d1)
    d1=d1.strftime("%Y-%m-%d")
    p1=df_mcycle[market_high_price_col].values[0]
    
    d2=df_mcycle[date_sell_col].values[0]
    d2=pd.to_datetime(d2)
    d2=d2.strftime("%Y-%m-%d")
    p2=df_mcycle[sell_price_col].values[0]
    
    d3=df_mcycle[date_market_low_col].values[0]
    d3=pd.to_datetime(d3)
    d3=d3.strftime("%Y-%m-%d")
    p3=df_mcycle[market_low_price_col].values[0]
    
    d4=df_mcycle[date_buy_col].values[0]
    d4=pd.to_datetime(d4)
    d4=d4.strftime("%Y-%m-%d")
    p4=df_mcycle[buy_price_col].values[0]
    
    anticipation_sell=df_mcycle['Anticipation_Sell'].values[0]
    reaction_buy=df_mcycle['Reaction_Buy'].values[0]
    

    return df_mcycle,d1,p1, d2, p2, d3, p3, d4, p4, anticipation_sell, reaction_buy

# Month over Month CPI

#def csentmom(df,v,window=3):
#    df['lastmonth']=df[v].shift(1)
#    df['csentmom']= df[v]/df[v].shift(1) - 1
#    v='csentmom'
#    # create 2-month moving average v_ma3 
#    df=dfsma(df,v,windows=window)
 #   df=dfstd(df,v,windows=window)
 #   return df


# Month over Month CPI

#def cpimom(df,v):
 #   if (not is_numeric_dtype(df[v])):
 #       print('... the variable',v,' is not numeric, convert to numeric')
 #       df[v] = pd.to_numeric(df[v],errors='coerce')
 #   df['cpimom'] = (df[v]/df[v].shift(1) -1 )
 #   return df



# Volatility Measures
 
#  plot_fits   Plot financial time series
#  Input is a dataframe
#  Index is in datetime forrmat



    # matplotlib references
        # Xticks
        #https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.pyplot.xticks.html

        # Matplotlib colors
        # https://matplotlib.org/3.1.1/tutorials/colors/colors.html

        # Matplotlib plot
        # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html

        # Stem Plot
        # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.stem.html?highlight=stem#matplotlib.pyplot.stem

        # Stem Plot python Graph Gallery
        # https://python-graph-gallery.com/180-basic-lollipop-plot/

        # Stem Plot Stackoverflow ... set color ... nice easy example
        # https://stackoverflow.com/questions/13145218/stem-plot-in-matplotlib

        # Plot Styles
        # https://problemsolvingwithpython.com/06-Plotting-with-Matplotlib/06.13-Plot-Styles/



