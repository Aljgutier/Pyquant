

from datetime import datetime, timedelta 
import time
import requests, pandas, lxml
from lxml import html

# Yahoo price history scraper
# https://medium.com/c%C3%B3digo-ecuador/how-to-scrape-yahoo-stock-price-history-with-python-b3612a64bdc6
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
     panda = pandas.read_html(table_tree)
     return panda
     panda = panda.loc[0:99]
     
if __name__ == '__main__':
     symbol = 'BB'
     
     dt_start = datetime.today() - timedelta(days=365)
     dt_end = datetime.today()
    
     start = format_date(dt_start)
     end = format_date(dt_end)
     
     sub = subdomain(symbol, start, end)
     header = header_function(sub)
     
     base_url = 'https://finance.yahoo.com'
     url = base_url + sub
     price_history = scrape_page(url, header)