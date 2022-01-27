import os
import yfinance as yf
import datetime
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import time
import datetime
import pandas as pd
import yfinance as yf
import csv
import json
import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
import datetime
import time
import statsmodels.tsa.stattools as ts
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import pandas as pd
from collections import Counter
import pprint
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


def getIntraDay(stock, filePath):
    """
    Gets the 60 day intraday data for the stock specified in the params
    stores in the file with filePath params combined with the folder name of the stock
    with a txt file containing the stock information
    """
    directoryPath = filePath + str(stock) + "/"
    filePath = filePath + str(stock) + "/Intraday_" + str(stock) + ".txt"
    # stockData = yf.download(stock, period='60d', interval='5m')
    stockData = yf.download(stock, period='60d', interval='5m')
    stockData.to_csv('temp.txt',header=None, sep=',', mode='w')
    file_data = open('temp.txt', 'rb').read()
    open('temp.txt', 'wb').write(file_data[:-2])
    if not os.path.exists(os.path.dirname(directoryPath)):
        os.mkdir(os.path.dirname(directoryPath))
        stockData.to_csv(filePath, sep=',',mode='a')
    else:
        if not os.path.isfile(filePath):
            stockData.to_csv(filePath, sep=',',mode='a')
        else:
            readExistingData = open(filePath,'r').read()
            splitExisting = readExistingData.split('\n')
            mostRecentLine = splitExisting[-2]
            lastUnix = mostRecentLine.split(',')[0]
            lastUnix = lastUnix[:len(lastUnix)-6]
            lastUnix = int(time.mktime(datetime.datetime.strptime(lastUnix, '%Y-%m-%d %H:%M:%S').timetuple()))
            # lastUnix = int(time.mktime(datetime.datetime.strptime(lastUnix, '%Y-%m-%d').timetuple()))
            readTempFile = open('temp.txt', 'r').read()
            splitTemp = readTempFile.split('\n')
            saveFile = open(filePath, 'a')
            for line in splitTemp:
                splitTempLine = line.split(',')
                lastUnixTemp = splitTempLine[0]
                lastUnixTemp = lastUnixTemp[:len(lastUnixTemp)-6]
                lastUnixTemp = int(time.mktime(datetime.datetime.strptime(lastUnixTemp, '%Y-%m-%d %H:%M:%S').timetuple()))
                # lastUnixTemp = int(time.mktime(datetime.datetime.strptime(lastUnixTemp, '%Y-%m-%d').timetuple()))
                if lastUnixTemp > lastUnix:
                    lineToWrite = line + '\n'
                    saveFile.write(lineToWrite)
            saveFile.close()
    os.remove('temp.txt')
    pass


def movingAverage(values, window):
    """
    Values are the numpy values for the data like the close values for the stock
    window is to calculate how many days back we have calculate the moving average - 12 day or 24 day etc.
    """
    weights = np.repeat(1.0, window)/window
    smas = np.convolve(values, weights, 'valid')
    return smas

def bytespdate2num(b):
    """
    Converts the bytes date to nums
    """
    return mdates.datestr2num(b.decode('utf-8'))

def graphData(stock, i, MA1, MA2):
    """
    Graph the stock in takes four parameters stocks -> is the symbol of the stock
    i -> index of the graph can be a number
    MA1 -> first moving average value 12 days or 24 days etc
    MA2 -> second moving average value
    """
    stockData = stock + '.txt'
    datep,openp,highp,lowp,closep,adjclosep,volume = np.loadtxt(stockData, delimiter=',', unpack=True, converters={0:bytespdate2num}, skiprows=1)
    plt.figure(i, facecolor='#07000d')

    x = 0
    y = len(datep)
    candleArr = []
    while x < y:
        appendLine = datep[x], openp[x], highp[x], lowp[x], closep[x]
        candleArr.append(appendLine)
        x+=1

    Av1 = movingAverage(closep, MA1)
    Av2 = movingAverage(closep, MA2)

    SP = len(datep[MA2-1:])

    label1 = str(MA1) + ' SMA'
    label2 = str(MA2) + ' SMA'


    ax1 = plt.subplot2grid((5,4),(0,0), rowspan=4, colspan=4, facecolor='#07000d')
    plt.subplots_adjust(left=0.10, bottom=0.25, right=0.93, top=0.95, wspace=0.20, hspace=0.00)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
    plt.gca().xaxis.label.set_color('w')
    plt.gca().yaxis.label.set_color('w')
    ax1.spines["top"].set_color('#5998ff')
    ax1.spines["bottom"].set_color('#5998ff')
    ax1.spines["left"].set_color('#5998ff')
    ax1.spines["right"].set_color('#5998ff')
    ax1.tick_params(axis='y', colors='w')
    ax1.tick_params(axis='x', colors='w')
    # candlestick_ohlc(ax1, candleArr, width=1, colorup='#9eff15', colordown='#ff1717', alpha=1)

    ax1.plot(datep[-SP:], Av1[-SP:], '#5998ff', label=label1, linewidth=1.5)
    ax1.plot(datep[-SP:], Av2[-SP:], '#e1edf9', label=label2, linewidth=1.5)

    plt.grid(True)
    plt.ylabel('Stock price')
    plt.suptitle(stock + ' Stock', color='w')
    plt.legend(loc=3, prop={'size':7}, fancybox=True)

    volumeMin = 0
    # volumeMin = volume.min()

    ax2 = plt.subplot2grid((5,4), (4,0), rowspan=1, colspan=4, sharex=ax1, facecolor='#07000d')
    plt.plot(datep, volume, '#00ffe8', linewidth=1)
    ax2.fill_between(datep, volumeMin, volume, facecolor='#00ffe8', alpha=0.5)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
    plt.gca().xaxis.label.set_color('w')
    plt.gca().yaxis.label.set_color('w')
    ax2.spines["top"].set_color('#5998ff')
    ax2.spines["bottom"].set_color('#5998ff')
    ax2.spines["left"].set_color('#5998ff')
    ax2.spines["right"].set_color('#5998ff')
    ax2.tick_params(axis='y', colors='w')
    ax2.tick_params(axis='x', colors='w')
    plt.xticks(rotation = -90)
    plt.grid(False)
    plt.xlabel('Date')
    plt.ylabel('Volume')


# eachStock = ['TTM', 'SBIN.NS']
# i = 0
# for k in eachStock:
#     graphData(k, i, 12, 26)
#     i = i + 1
# plt.show()

def getFundamentals(stockSymbolsPath, fundamentalsFilePath):
    """
    This will take the json file which include an 
    empty dict and parse and update the stock information 
    based on the symbols list from the text file
    """
    # stockSymbolsPath -> file containing stocks list -> ex-data/symbols.txt
    # fundamentalsFilePath -> file containing {} or existing dict -> ex-data/fundamentals.jsom
    # if the function stops in the middle it will resume from where it is started
    stocks = []
    with open(stockSymbolsPath, 'r') as f:
        for line in f.readlines():
            stocks.append(line.strip('\n'))
    existing_fundamentals = []
    with open(fundamentalsFilePath, 'r') as outfile:
        existing_fundamentals_dict = json.load(outfile)
    for i in existing_fundamentals_dict:
        existing_fundamentals.append(str(i))
    stocks = [item for item in stocks if item not in existing_fundamentals]
    stocks_fundamentals = {}
    i = 1
    for stock in stocks:
        print("Getting fundamentals for ", stock, "(", i, "/", len(stocks), ")")
        tk = yf.Ticker(stock).info
        stocks_fundamentals[stock] = tk
        with open(fundamentalsFilePath, "ab") as outfile:
            entry = {}
            entry[stock] = tk
            outfile.seek(-1, os.SEEK_END)
            outfile.truncate()
            outfile.write(','.encode())
            entry_string = json.dumps(entry)[1:-1]
            outfile.write(entry_string.encode())
            outfile.write('}'.encode())
        print("Status: Complete")
        i = i + 1
    print('Stock data process complete')

def getData(stocks, filePath, newData):
    # stock -> string eg-TATASTEEL
    # filepath -> string eg-data/
    
    # newData -> true or falls -> if new data make sure dir is empty for the first run
    # if existing data to be updated make sure the program doent stop in middle
    ## ---- This for stocks to get to Know and save data in your folder ---
    if newData:
        stocks_existing = [os.path.splitext(filename)[0] for filename in os.listdir(filePath)]
        print(len(stocks_existing))
        stocks = [item for item in stocks if item not in stocks_existing]
        print(len(stocks))
    for stock in stocks:
        stockData = yf.download(stock, period='max', interval='1d')
        filePath_ = str(filePath) + stock + '.txt'
        stockData.to_csv('temp.txt',header=None, sep=',', mode='w')
        file_data = open('temp.txt', 'rb').read()
        open('temp.txt', 'wb').write(file_data[:-2])
        try:
            # this try block checks for existing data for the ticker selected
            # and if exist it will update with the current results
            readExistingData = open(filePath_,'r').read()
            splitExisting = readExistingData.split('\n')
            mostRecentLine = splitExisting[-2]
            lastUnix = mostRecentLine.split(',')[0]
            lastUnix = int(time.mktime(datetime.datetime.strptime(lastUnix, '%Y-%m-%d').timetuple()))
            readTempFile = open('temp.txt', 'r').read()
            splitTemp = readTempFile.split('\n')
            saveFile = open(filePath_, 'a')
            for line in splitTemp:
                splitTempLine = line.split(',')
                lastUnixTemp = splitTempLine[0]
                lastUnixTemp = int(time.mktime(datetime.datetime.strptime(lastUnixTemp, '%Y-%m-%d').timetuple()))
                if lastUnixTemp > lastUnix:
                    lineToWrite = line + '\n'
                    saveFile.write(lineToWrite)
            saveFile.close()
        except Exception as e:
            print(filePath_)
            # try block fails if the path doesn't exist and the data will be 
            # created in a new csv file
            stockData.to_csv(filePath_, sep=',',mode='a')
            lastUnix = 0

def getHighBetaList(fundamentalFilePath, listSavePath):
    """
    This function will create a txt file which contains only stocks with
    higher beta value (1 to 2)
    it reqires a json file which contains the fundamental data for stocks
    """
    # fundamentalFilePath -> string -> ex - data/fundamentals.txt
    # listSavePath -> string -> ex - data/highBeta.txt
    with open(fundamentalFilePath) as f:
        stocks = json.load(f)
    BETA = []
    error_stocks = []
    for i in stocks:
        try:
            beta = stocks[i]['beta']
            if beta>1 and beta<2:
                BETA.append(i)
        except Exception as e:
            print("cannot get data for", i)
            error_stocks.append(i)
            pass
    with open(listSavePath, 'w') as f:
        for stock in BETA:
            f.write(str(stock) + "\n")

def getEOD(stock, filePath):
    """
    Gets the max financial daily data for the stock specified in the params
    stores in the file with filePath params combined with the folder name of the stock
    with a txt file containing the stock information
    """
    directoryPath = filePath + str(stock) + "/"
    filePath = filePath + str(stock) + "/Daily_" + str(stock) + ".txt"
    stockData = yf.download(stock, period='10y', interval='1d')
    stockData.to_csv('temp.txt',header=None, sep=',', mode='w')
    file_data = open('temp.txt', 'rb').read()
    open('temp.txt', 'wb').write(file_data[:-2])
    if not os.path.exists(os.path.dirname(directoryPath)):
        os.mkdir(os.path.dirname(directoryPath))
        stockData.to_csv(filePath, sep=',',mode='a')
    else:
        readExistingData = open(filePath,'r').read()
        splitExisting = readExistingData.split('\n')
        mostRecentLine = splitExisting[-2]
        lastUnix = mostRecentLine.split(',')[0]
        lastUnix = int(time.mktime(datetime.datetime.strptime(lastUnix, '%Y-%m-%d').timetuple()))
        readTempFile = open('temp.txt', 'r').read()
        splitTemp = readTempFile.split('\n')
        saveFile = open(filePath, 'a')
        for line in splitTemp:
            splitTempLine = line.split(',')
            lastUnixTemp = splitTempLine[0]
            lastUnixTemp = int(time.mktime(datetime.datetime.strptime(lastUnixTemp, '%Y-%m-%d').timetuple()))
            if lastUnixTemp > lastUnix:
                lineToWrite = line + '\n'
                saveFile.write(lineToWrite)
        saveFile.close()
    os.remove('temp.txt')
    pass

def checkADF(df):
    """
    This takes the stock dataframe which contains date, ohlc, adj close and volume data
    and check if we have ADF
    here makesure test_statistic are greater than critical_values
    p-value < 0.05
    """
    results = ts.adfuller(df['Adj Close'], 1)
    test_statistic = results[0]
    p_value = results[1]
    data_points = results[3]
    critical_values = results[4]
    return test_statistic, p_value, data_points, critical_values

def hurst(df, max_lags):
    """
    hurst value is used to check if the data is mean reverting, geometric brownian motion or is it trending
    if H < 0.5 time series is mean reverting
    if H = 0.5 time series is Geometric Brownian Motion
    if H > 0.5 time series is trending
    """
    lags = range(2, max_lags)
    tau = [np.std(np.subtract(df['Adj Close'].values[lag:], df['Adj Close'].values[:-lag])) for lag in lags]
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]

def plot_price_series(df, ts1, ts2):
    months = mdates.MonthLocator() # every month 
    fig, ax = plt.subplots()
    ax.plot(df.index, df[ts1], label=ts1)
    ax.plot(df.index, df[ts2], label=ts2)
    ax.xaxis.set_major_locator(months) 
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    # ax.set_xlim(datetime.datetime(2012, 1, 1), datetime.datetime(2013, 1, 1))
    ax.grid(True)
    fig.autofmt_xdate()
    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title("%s and %s Daily Prices" % (ts1, ts2))
    plt.legend()
    plt.show()

def plot_scatter_series(df, ts1, ts2):
    plt.xlabel("%s Price ($)" % ts1)
    plt.ylabel("%s Price ($)" % ts2)
    plt.title('%s and %s Price Scatterplot' % (ts1, ts2))
    plt.scatter(df[ts1], df[ts2])
    plt.show()

def plot_residuals(df):
    months = mdates.MonthLocator() # every month 
    fig, ax = plt.subplots()
    ax.plot(df.index, df["res"], label="Residuals") 
    ax.xaxis.set_major_locator(months) 
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")) 
    ax.set_xlim(datetime.datetime(2012, 1, 1), datetime.datetime(2013, 1, 1)) 
    ax.grid(True)
    fig.autofmt_xdate()
    plt.xlabel("Month/Year")
    plt.ylabel("Price ($)")
    plt.title("Residual Plot")
    plt.legend()
    plt.plot(df["res"])
    plt.show()

def preProcessDataForLabels(combineStocksFilePath, stock):
    hm_day = 7
    df = pd.read_csv(combineStocksFilePath, index_col=0)
    stocks = df.columns.values.tolist()
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    for i in range(1, hm_day+1):
        df['{}_{}d'.format(stock, i)] = (df[stock].shift(-i)-df[stock])/df[stock]
    
    df.fillna(0, inplace=True)
    return stocks, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extract_featuresets(combineStocksFilePath, stock):
    stocks, df = preProcessDataForLabels(combineStocksFilePath, stock)
    df['{}_target'.format(stock)] = list(map(buy_sell_hold, df['{}_1d'.format(stock)], df['{}_2d'.format(stock)], df['{}_3d'.format(stock)], df['{}_4d'.format(stock)], df['{}_5d'.format(stock)], df['{}_6d'.format(stock)], df['{}_7d'.format(stock)]))
    vals = df['{}_target'.format(stock)].values.tolist()
    str_vals = [str(i) for i in vals]
    # print('Data spread:',Counter(str_vals))
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df_vals = df[[stock for stock in stocks]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    X = df_vals.values
    y = df['{}_target'.format(stock)].values
    return X, y, df

def doML(combineStocksFilePath, stock):
    X, y, df = extract_featuresets(combineStocksFilePath, stock)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    # clf = neighbors.KNeighborsClassifier()
    clf = VotingClassifier([('lsvc', svm.LinearSVC()), ('knn', neighbors.KNeighborsClassifier()), ('rfor', RandomForestClassifier())])
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)
    print('Predicted spread:', Counter(predictions))
    return confidence

# stocks = []
# with open('data/highBeta.txt', 'r') as f:
#     for l in f.readlines():
#         stock = l.split('\n')[0]
#         stocks.append(stock)
 
# df = pd.read_csv('data/HighBetaStocks/SBIN.NS.txt')
# print(df.head())
# print(doML('data/compiledHighBeta.csv', 'SBIN.NS'))
