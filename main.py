import urllib
import time
import yfinance as yf
import datetime
import os
from modules import getIntraDay

new_data = True
update_data = False

stocks = []
with open('symbols.txt', 'r') as f:
    for line in f.readlines():
        stocks.append(line.strip('\n'))

existing_stocks = [os.path.splitext(filename)[0] for filename in os.listdir('data/')]

stocks = [item for item in stocks if item not in existing_stocks]

