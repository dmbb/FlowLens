import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict

def RoundToNearest(n, m):
    if (m == 1):
        return n
    if (n > 0):
        r = n % m
        return n + m - r if r + r >= m else n - r
    else:
        if (n < 0):
            return RoundToNearest(abs(n), m) * -1
    return 0

def main(argv):
    name = str(argv[0])
    BASE_DIR = os.path.dirname(name)
    file = open(name,'r')
    binWidth = int(argv[1])
    
    trainSet = open(BASE_DIR+"/TrainSet_" + str(binWidth) + ".csv", 'w')
    testSet = open(BASE_DIR+"/TestSet_" + str(binWidth) + ".csv", 'w')    
    
    minBucket = RoundToNearest(-1500, binWidth)
    maxBucket = RoundToNearest(1500, binWidth) + 1
    for size in range(minBucket, maxBucket, binWidth):
        trainSet.write("packetLengthBin_" + str(size) + ", ")
        testSet.write("packetLengthBin_" + str(size) + ", ")
    trainSet.write("class\n")
    testSet.write("class\n")    
    
    i = 0

    TFlineToWrite = []
    CNlineToWrite = []


    lineToWrite = OrderedDict()

    l = file.readline()
    l = file.readline()
    l = file.readline()
    l.rstrip('\n')

    lineNumber = 0
    while l:
        lineSplit = l.split(" ")
        if (lineNumber % 2 == 0):
            timestamp = lineSplit[2]
        else:
            website = lineSplit[0][:-1]
            lineToWrite[website+"|"+timestamp] = {}
            lineToWrite[website+"|"+timestamp] = defaultdict(lambda:0, lineToWrite[website+"|"+timestamp])
            t = lineToWrite[website+"|"+timestamp]
            for x in lineSplit[1:]:
                try:
                    t[str(RoundToNearest(int(x), binWidth))] += 1
                except:
                    continue
            lineToWrite[website+"|"+timestamp] = t
        lineNumber += 1
        l = file.readline()
        l.rstrip('\n')

    max = 4
    max2 = max + 4
    counter = 0
    currentWebSite = ""
    for j in lineToWrite:
        if (currentWebSite != j.split("|")[0]):
            counter = 0

        currentWebSite = j.split("|")[0]

        if (counter < max):
            for s in range(minBucket, maxBucket, binWidth):
                trainSet.write(str(lineToWrite[j][str(s)]) + ", ")
            trainSet.write(currentWebSite + "\n")
            if (counter == 0):
                firstTimeStamp = datetime.strptime(j.split("|")[1], "%Y-%m-%d#%H:%M:%S")
                secondTimeStamp = firstTimeStamp + timedelta(days=8)
            counter += 1
        else:
            if (datetime.strptime(j.split("|")[1], "%Y-%m-%d#%H:%M:%S") < secondTimeStamp):
                lineToWrite[j] = {}
                continue
            if (counter < max2):
                for s in range(minBucket, maxBucket, binWidth):
                    testSet.write(str(lineToWrite[j][str(s)]) + ", ")
                testSet.write(currentWebSite + "\n")
            counter += 1

        lineToWrite[j] = {}

if __name__ == "__main__":
    main(sys.argv[1:])

