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
.. module:: finmktfunctions.py
    :Python version 3.7 or greater
    :synopsis: Basic functions and wrappers to support market financial analytics including
    transforms, plotting, and trading trading strategies

.. moduleauthor:: Alberto Gutierrez <aljgutier@yahoo.com>
"""


# Yahoo price history scraper
# https://medium.com/c%C3%B3digo-ecuador/how-to-scrape-yahoo-stock-price-history-with-python-b3612a64bdc6
# see fmscraper ipynb notebook for example
# This is an example in case we need a scraper in the future ... still requires enhancedment
#   hardwiree to 100 day download 
def format_date(date_datetime):
     date_timetuple = date_datetime.timetuple()
     date_mktime = time.mktime(date_timetuple)
     date_int = int(date_mktime)
     date_str = str(date_int)
     return date_str
def subdomain(symbol, start, end, filter='history'):
     subdoma="/quote/{0}/history?period1={1}&period2={2}&interval=1d&filter={3}&frequency=1d"
     subdomain = subdoma.format(symbol, start, end, filter)
     return subdomain
 
def header_function(subdomain):
     hdrs =  {"authority": "finance.yahoo.com",
              "method": "GET",
              "path": subdomain,
              "scheme": "https",
              "accept": "text/html",
              "accept-encoding": "gzip, deflate, br",
              "accept-language": "en-US,en;q=0.9",
              "cache-control": "no-cache",
              "cookie": "Cookie:identifier",
              "dnt": "1",
              "pragma": "no-cache",
              "sec-fetch-mode": "navigate",
              "sec-fetch-site": "same-origin",
              "sec-fetch-user": "?1",
              "upgrade-insecure-requests": "1",
              "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64)"}
     
     return hdrs



def scrape_page(url, header):
    page = requests.get(url, headers=header)
    element_html = html.fromstring(page.content)
    table = element_html.xpath('//table')
    table_tree = lxml.etree.tostring(table[0], method='xml')
    panda = pandas.read_html(table_tree)[0]

    panda = panda.loc[0:99]
    panda.columns=['Date','Open', 'High', 'Low','Close','Adj CLose','Volume']

    return panda

#if __name__ == '__main__':
#     symbol = 'BB'
     
#     dt_start = dt.datetime.today() - timedelta(days=365)
#     dt_end = dt.datetime.today()
    
#     start = format_date(dt_start)
#     end = format_date(dt_end)
     
#     sub = subdomain(symbol, start, end)
#     header = header_function(sub)
     
#     base_url = 'https://finance.yahoo.com'
#     url = base_url + sub
#     price_history = scrape_page(url, header)


def fredSeries(Series,start,end,df='',API_KEY_FRED='',save=False,savedir='.'):

    """

    Get the yield curve, T103YM from FRED (Fedral Reserve Economic Data) St. Louis. 


    Args:
        Seiries (string): Name of the Fred Series
        start (string): "YYYY-MM-DD"
        end (string): "YYYY-MM-DD"
        API_KEY_FRED (string): FRED API Key
        df (dataframe): append to dataframe 
        save (boolean): if True (default) then save the file with filename "T103YM_year-month-day_to_year-month-day=year.csv" 
        savedir ('string'): director for saving data. Default is current directory './'

    Returns:
        The dft103ym dataframe and by default saves the data to file

    Notes
    -----
        Uses the Mortada fredapi package to get 'T103YM' series from FRED (Federal Reserve Economic Data)
        (https://pypi.org/project/fredapi/), pip install fredapi

    |

    """

    fred = Fred(api_key=API_KEY_FRED)
    dfseries=pd.DataFrame(fred.get_series(Series,observation_start=start, observation_end=end))
    dfseries.columns=[Series]
    #display(dfseries.tail(3))


    ## append 
    if ~df.empty:
        #display(df.head(3))
        #display(dfsymbol.head(3))

        df = pd.concat([df,dfseries])
        df=df[~df.index.duplicated(keep='first')]

    # save
    if save == True:

        # remove special characters from string name

        s=df.index[0]
        e=df.index[len(df.index)-1]
        filename=Series+'_'+str(s.year)+'-'+str(s.month)+'-'+str(s.day)
        filename=savedir+'/'+filename+'_to_'+str(e.year)+'-'+str(e.month)+'-'+str(e.day)+'.csv'
        print("df to csv, filename = ",filename)
        #dfsymbol.reset_index(inplace=True)
        df.reset_index().to_csv(filename,index=False)

    return df


def peapply(row):
    if np.isnan(row['PE']):
        return row['Close']/row['Earnings']
    else:
        return row['PE']

def quandl_sppegetappend(dfsppe,dfsp500,quandl_api_key, start_date, end_date, save=False,savedir='./'):

    data=quandl.get("MULTPL/SP500_PE_RATIO_MONTH", authtoken=quandl_api_key, start_date=start_date, end_date=end_date)
 
    df_data=pd.DataFrame(data)
    df_data.columns=['PE']
    df_sppe=pd.concat([dfsppe,df_data])

    df_sppe=df_sppe[~df_sppe.index.duplicated(keep='first')]
    df_sppe.sort_index(inplace=True, ascending=True)
    
    df_sppe_daily=dfsp500.join(df_sppe,how='left')
    df_sppe_daily['Earnings']=df_sppe_daily['Close']/df_sppe['PE']
    df_sppe_daily['Earnings']=df_sppe_daily['Earnings'].ffill()
    df_sppe_daily['PE']=df_sppe_daily.apply(peapply,axis=1)


    #df_sppe_daily.loc[dt.datetime(2020,7,30):dt.datetime(2020,8,3)]
    #df_sppe_daily.sort_index(inplace=True, ascending=True)
    #filename="./data/sp500_sp500_pe_ratio_daily_1950-1-3_to_2020-8-4.csv"
#df_sppe_daily.to_csv()


    if save==True:
        s=df_sppe_daily.index[0]
        e=df_sppe_daily.index[len(df_sppe_daily.index)-1]
        filename='sp500_pe_daily_'+str(s.year)+'-'+str(s.month)+'-'+str(s.day)
        filename=savedir+'/'+filename+'_to_'+str(e.year)+'-'+str(e.month)+'-'+str(e.day)+'.csv'
        print("df to csv, filename = ",filename)
        #df_sppe_daily[['PE','Earnings']].reset_index().to_csv(filename,index=False)



    return df_sppe_daily[['PE','Earnings']]

def yahoof_getappend(symbol,start,end,df='',save=False,savedir='./'):

    """

    Get data from yahoo finance and append Append. 

    **Parameters**:
        symbol(string): symbol to get from yahoo
        start (datetime): start date
        end (datetime): end date
        df (dataframe): append the data from yahoo finance to df
        save (boolean): if True (default) then save the file with filename "symbol_year-month-day_to_year-month-day=year.csv" 
        savedir (string): directory to save data. Default is current directory '.'

    **Returns**: 
        The input dataframe with an additional variable "ticker_R" corresponding to daily returns

    **How it works**:
        The *yfgetappend* function uses the yfinance package to get data for the "symbol" and appends to df. 
        (https://aroussi.com/post/python-yahoo-finance). to install the yfanance package with
        anaconda: conda install -c ranaroussi yfinance 

    |

    """
    print(start, end)
    dfsymbol = yf.download(symbol,start,end)

    ## append 
    if ~df.empty:
        #display(df.head(3))
        #display(dfsymbol.head(3))
        dfsymbol = pd.concat([df,dfsymbol])
        dfsymbol=dfsymbol[~dfsymbol.index.duplicated(keep='first')]



    if save == True:

        # remove special characters from string name
        symbol2=''
        for c in symbol:
            if c.isalnum():
                symbol2 += c

        s=dfsymbol.index[0]
        e=dfsymbol.index[len(dfsymbol.index)-1]
        filename=symbol2+'_'+str(s.year)+'-'+str(s.month)+'-'+str(s.day)
        filename=savedir+'/'+filename+'_to_'+str(e.year)+'-'+str(e.month)+'-'+str(e.day)+'.csv'
        print("df to csv, filename = ",filename)
        #dfsymbol.reset_index(inplace=True)
        dfsymbol.reset_index().to_csv(filename,index=False)

    return dfsymbol



# Yield Curve 
#joincols=['gdp', 'gdp_qoq', 'gdp_prevqoq','recession1q' ,'recession2q']
# Join Financial Time Series, Fill Forward
def fmjoinff(dfmarket,dflei,joincols='',market_variables='close_price',verbose=False,dropnas=True):
    """
    Financial Time Series Join Fill Forward

    **Parameters**:
        dfmarket (dataframe): of X, independent variable colums.
        dflei (dataframe): of Y, dependent variable.
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

    if joincols=='':
        joincols=df.columns



    print("join columns =",joincols)
    dfmarket=dfmarket.join(dflei[joincols],lsuffix='l',rsuffix='r', how='outer')



    for v in joincols:
        if (not is_numeric_dtype(dfmarket[v])):
            if verbose:
                print('... the variable',v,' is not numeric, convert to numeric')
            dfmarket[v] = pd.to_numeric(dfmarket[v],errors='coerce')


    # fill forward since the LEI update occurs early and needs to be carried forwarde
    dfmarket.loc[:, joincols] = dfmarket[joincols].ffill() 

    # Nulls
    # tail rows with null close_price ... verify that these are non market dates
    if verbose:
        print('Market Variable Nulls:',dfmarket[market_variables][dfmarket[market_variables].isnull()].tail(5))
        
   
   # drop non market dates
    if dropnas:
        if verbose:
            print('\nDrop non market dates (dropna) ... ')
        dfmarket = dfmarket.dropna(subset=market_variables) 


    if verbose:
        print('\nMarket Variable Nulls =\n' ,dfmarket[market_variables].isnull().sum())
        print('\nAll Join Variable Nulls =\n' ,dfmarket[joincols].isnull().sum())  
        print('\nJoin variable nulls after start of join series\n',dfmarket.loc[dflei.index[0]:,joincols].isnull().sum())
        # Display the Dataframe
        display(dfmarket.tail(3))

    return dfmarket




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

        # * np.sqrt(w)

    # returns v_lrstdw
    return df


def dfstd(df,v,windows):

    if not isinstance(windows, list):
        windows=[windows]
    
    for w in windows:
        v_std = v + '_std' + str(w)
        df[v_std] = df[v].rolling(window=w).std()

        # * np.sqrt(w)

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
    
    return df


# GDP Functions
def gdp_recession1q(gdpqoq):
    if (gdpqoq < 0):
        return 1
    else:
        return 0

def gdp_recession2q(gdpqoq, gdpqoqprev):
    if ((gdpqoq < 0) and (gdpqoqprev < 0)):
        return 1
    else:
        return 0

def gdpqoq(df,vn_gdp):
    df['gdp_prevq']=df[vn_gdp].shift(1)
    df['gdp_qoq']=100*(df[vn_gdp]/df['gdp_prevq']-1)
    df['gdp_prevqoq']=df['gdp_qoq'].shift(1)
    df['recession1q']=df.apply(lambda row: gdp_recession1q(row['gdp_qoq']),axis=1)
    df['recession2q']=df.apply(lambda row: gdp_recession2q(row['gdp_qoq'],row['gdp_prevqoq']),axis=1)
    return df

# Month over Month CPI

def csentmom(df,v,window=3):
    df['lastmonth']=df[v].shift(1)
    df['csentmom']=df[v]-df['lastmonth']
    v='csentmom'
    # create 2-month moving average v_ma3 
    df=dfsma(df,v,windows=window)
    df=dfstd(df,v,windows=window)
    return df


# Month over Month CPI

def cpimom(df,v):
    if (not is_numeric_dtype(df[v])):
        print('... the variable',v,' is not numeric, convert to numeric')
        df[v] = pd.to_numeric(df[v],errors='coerce')
    df['lastmonth']=df[v].shift(1)
    df['cpimom']=100*(df[v]/df['lastmonth'] - 1)
    df['cpimom_lastmonth']=df['cpimom'].shift(1)
    df['cpimomvelocity']=df['cpimom']-df['cpimom_lastmonth']
    return df



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



