# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:43:48 2023

@author: ericj
"""

p='https://api.the-odds-api.com/v4/sports/upcoming/odds/?regions=us&markets=spreads,totals&bookmakers=fliff&oddsFormat=american&apiKey=a0c2b10535688b63d12f0f10e881718c'

import pandas as pd
import requests
API_KEY = 'a0c2b10535688b63d12f0f10e881718c'
SPORT = 'upcoming' # use the sport_key from the /sports endpoint below, or use 'upcoming' to see the next 8 games across all sports

REGIONS = 'us' # uk | us | eu | au. Multiple can be specified if comma delimited

MARKETS = 'totals,spreads' # h2h | spreads | totals. Multiple can be specified if comma delimited

ODDS_FORMAT = 'american' # decimal | american

DATE_FORMAT = 'iso' # iso | unix

BOOKMAKERS = 'fliff'

odds_response = requests.get(
    f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds',
    params={
        'api_key': API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': DATE_FORMAT,
        'bookmakers': BOOKMAKERS,
    }
)


if odds_response.status_code != 200:
    print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

else:
    odds_json = odds_response.json()
    df = pd.DataFrame.from_dict(odds_json)
    for i in range(0,len(df)):
        try:
            t1=pd.DataFrame.from_dict(pd.DataFrame.from_dict(pd.DataFrame.from_dict(df.bookmakers[i])['markets'][0])['outcomes'][0])['price'][0]
            t2=pd.DataFrame.from_dict(pd.DataFrame.from_dict(pd.DataFrame.from_dict(df.bookmakers[i])['markets'][0])['outcomes'][0])['price'][1]
        except:
            t1=0
            t2=0
        try:
            r1=pd.DataFrame.from_dict(pd.DataFrame.from_dict(pd.DataFrame.from_dict(df.bookmakers[i])['markets'][0])['outcomes'][1])['price'][0]
            r2=pd.DataFrame.from_dict(pd.DataFrame.from_dict(pd.DataFrame.from_dict(df.bookmakers[i])['markets'][0])['outcomes'][1])['price'][1]
        except:
            r1=0
            r2=0
        
        if (t1==-100 or t1==-105 or t1==-110 or t1==100 or t1==105 or t1==110) and (t2==-100 or t2==-105 or t2==-110 or t2==100 or t2==105 or t2==110):
            print(pd.DataFrame.from_dict(pd.DataFrame.from_dict(pd.DataFrame.from_dict(df.bookmakers[i])['markets'][0])['outcomes'][0]))
        
        if (r1==-100 or r1==-105 or r1==-110 or r1==100 or r1==105 or r1==110) and (r2==-100 or r2==-105 or r2==-110 or r2==100 or r2==105 or r2==110):
            t=(pd.DataFrame.from_dict(pd.DataFrame.from_dict(pd.DataFrame.from_dict(df.bookmakers[i])['markets'][0])['outcomes'][0]).name)
            print(pd.concat([t,pd.DataFrame.from_dict(pd.DataFrame.from_dict(pd.DataFrame.from_dict(df.bookmakers[i])['markets'][0])['outcomes'][1])]))
        
    
    
    print('Number of events:', len(odds_json))
    
    
    # Check the usage quota
    print('Remaining requests', odds_response.headers['x-requests-remaining'])
    print('Used requests', odds_response.headers['x-requests-used'])