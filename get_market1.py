#!/usr/bin/python
#program to read AAPL stock prices.
#name: R. Melton
#date: 2/10/21
# docs: https://alpaca.markets/docs/api-documentation/api-v2/market-data/bars/
#data = ["Millie", "Bobby", "Brown", "is", "Enola", "Holmes"]
import alpaca_trade_api as tradeapi
import http.client, requests, json
import yahoo_fin.stock_info as si

def test_yahoo(s):
  url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/auto-complete"

  querystring = {"q":s,"region":"US"}

  headers = {
      'x-rapidapi-key': "17cb0a1f2amsh7758918ce21cbf4p16088ajsnba053a73b333",
      'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com"
      }

  quote_table = si.get_quote_table("aapl", dict_result=True)
  print(s," MC: ",quote_table["Market Cap"],"\tPE: ",quote_table['PE Ratio (TTM)'])
  print()



def get_market_list(symList):
  api = tradeapi.REST()
  # Get daily price data for AAPL over the last 5 trading days.
  barset = api.get_barset(symList, 'day', limit=10)
  bars = barset[symList[0]]

  # See how much syms moved in that timeframe.
  for sym in symList:
    bars = barset[sym]
    week_open = bars[0].o
    week_close = bars[-1].c
    percent_change = (week_close - week_open) / week_open * 100
    print('{}\t moved {:12.2f}% over the last 5 days,   close\t${}'.format(sym,percent_change,week_close))
  
  print("")

if __name__ == '__main__':
  syms1=["AAPL", "T", "GOOGL", "AMZN", "FB", "NVDA","INTC","QCOM","VMW","VZ","TSLA"]
  syms2=["GM", "USA", "PEP", "TGT", "WMT"]

  get_market_list(syms1)
  get_market_list(syms2)
  test_yahoo('AAPL')

  #syms=["AAPL", "T", "GOOGL", "AMZN", "FB"]

  
  