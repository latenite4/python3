#!/usr/bin/python
# program to read stock market info
#name: R. Melton
#date: 2/10/21

# API endpoint: https://paper-api.alpaca.markets
# key ID: PKRB63AJ2WX5RPXNS3ZO, PKHL9L3VTQPWBGAI891M


import alpaca_trade_api as tradeapi

#function to check if market is open
def check_open():
  api = tradeapi.REST()

  # Check if the market is open now.
  clock = api.get_clock()

  print('The market is {}'.format('open.' if clock.is_open else 'closed.'))
  return ('1' if clock.is_open else '0')

if __name__ == '__main__':
  check_open()