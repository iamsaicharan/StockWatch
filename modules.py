import os
import yfinance as yf
import datetime
import time


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