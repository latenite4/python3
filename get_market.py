#!/usr/bin/python
# program to read stock market info
#name: R. Melton
#date: 2/10/21

# API endpoint: https://paper-api.alpaca.markets
# key ID: PKRB63AJ2WX5RPXNS3ZO, PKHL9L3VTQPWBGAI891M


import alpaca_trade_api as tradeapi

if __name__ == '__main__':
  api = tradeapi.REST()

  # Check if the market is open now.
  clock = api.get_clock()
  print('The market is {}'.format('open.' if clock.is_open else 'closed.'))

  # Check when the market was open on Dec. 1, 2018
  date = '2021-02-01'
  calendar = api.get_calendar(start=date, end=date)[0]
  print('The market opened at {} and closed at {} on {}.'.format(
    calendar.open,
    calendar.close,
    date
  ))