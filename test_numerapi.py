#!/usr/bin/python
#program to test Numer.ai API access
import numerapi, os
import pandas as pd



print('testing numer.ai API access')
#this is just name of key which is my account email address
npub = os.getenv('NAPI_PUBLIC_KEY')
#this is just the value of the secret key which they give me
npriv = os.getenv('NAPI_PRIV_KEY')
# CATcat0!

#print(f'API pub key {npub} '.format(npub))
print(f'API pub key {npub}  priv key {npriv}'.format(npub,npriv))

#get access to numer.ai signals API
napi = numerapi.SignalsAPI(npub, npriv)

# read in list of active Signals tickers which can change slightly era to era
eligible_tickers = pd.Series(napi.ticker_universe(), name='ticker') 
eligible_tickers.head()
print(f"Number of eligible tickers: {len(eligible_tickers)}")

#grab submission API endpoint
#test if API is accessable
#get access to numer.ai main tournament API; they use same pub and priv key 
#but it will be different for each numer.ai user

napi = numerapi.NumerAPI(npub, npriv)
napi.download_current_dataset(unzip=False)
print('numer.ai current dataset will be in this dir as a zip file.')

