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
.. module:: fmget.py
    :Python version 3.7 or greater
    :synopsis: functions to get and manage financial and economic data from various APIs

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


def fred_getappend(Series,start,end,df='',API_KEY_FRED='',save=False,savedir='.'):

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
        df.sort_index(inplace=True, ascending=True)

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


def _peapply(row):
    if np.isnan(row['PE']):
        return row['Close']/row['Earnings']
    else:
        return row['PE']

# PE and Eaarnings ... need to get price from dfsp500
def quandl_sppe_getappend(dfsppe,dfsp500,quandl_api_key, start_date, end_date, save=False,savedir='./'):

    data=quandl.get("MULTPL/SP500_PE_RATIO_MONTH", authtoken=quandl_api_key, start_date=start_date, end_date=end_date)
 
    pe_data = data=quandl.get("MULTPL/SP500_PE_RATIO_MONTH", authtoken=quandl_api_key)
    pe_data.columns=["PE"]
    df_sppe=pe_data.join(dfsp500,how='left')
    df_sppe.drop(['High','Low','Open','Volume','Adj Close'],axis=1, inplace=True)
    df_sppe.dropna(how='any',inplace=True)
    df_sppe['Earnings'] = df_sppe['Close']/df_sppe['PE']

    if save==True:
        s=df_sppe.index[0]
        e=df_sppe.index[df_sppe.index.size-1]
        filename='sppe_'+str(s.year)+'-'+str(s.month)+'-'+str(s.day)
        filename=savedir+'/'+filename+'_to_'+str(e.year)+'-'+str(e.month)+'-'+str(e.day)+'.csv'
        df_sppe.reset_index().to_csv(filename,index=False)
        print("df to csv, filename = ",filename)

    return df_sppe

def yahoo_getappend(symbol,start,end,df='',save=False,savedir='./'):

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
