import os
import csv
import sys
from datetime import datetime, timedelta
import collections
import math
from collections import defaultdict, OrderedDict
import numpy as np

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

def extractDistributionWithoutTruncation(argv):
    BASE_DIR = os.path.dirname(argv[0])
    file = open(argv[0],'r')
    
    binWidth = int(argv[1])
    websiteToClassify = argv[2]

    if not os.path.exists(BASE_DIR + "/" + websiteToClassify):
        os.makedirs(BASE_DIR + "/" + websiteToClassify)
        
    trainSet = open(BASE_DIR + "/" + websiteToClassify + "/TrainSet_full_" + str(binWidth) + ".csv", 'w')
    testSet = open(BASE_DIR + "/" + websiteToClassify + "/TestSet_full_" + str(binWidth) + ".csv", 'w')
    

    #Set for all possible quantized buckets
    binsUsedByWebsite = set()
    minBucket = RoundToNearest(-1500, binWidth)
    maxBucket = RoundToNearest(1500, binWidth) + 1
    for size in range(minBucket, maxBucket, binWidth):
        binsUsedByWebsite.add(RoundToNearest(size, binWidth))


    websiteTrainInstances = int(argv[3])
    websiteTestInstances = int(argv[4])
    
    ################################################
    #Build csv with quantized bins
    ################################################

    # Write CSV datasets header (with bins used by the target website)
    for size in range(minBucket, maxBucket, binWidth):
        if (size in binsUsedByWebsite):
            trainSet.write("packetLengthBin_" + str(size) + ", ")
            testSet.write("packetLengthBin_" + str(size) + ", ")
    trainSet.write("class\n")
    testSet.write("class\n")


    file = open(argv[0],'r')
    l = file.readline() #Take out dataset header
    l = file.readline() #Take out dataset header
    trainCounter = 0
    testCounter = 0
    currWebsite = ""
    trainData = []
    testData =[]

    for lineNumber, l in enumerate(file.readlines()):
        lineSplit = l.rstrip('\n').split(" ")
        if (lineNumber % 2 == 1): #Gather website data
            website = lineSplit[0][:-1]
            if(website != currWebsite):
                currWebsite = website
                trainCounter = 0
                testCounter = 0
            
            #Build container for sample distribution
            website_bin_distribution = OrderedDict()
            for i in sorted(binsUsedByWebsite):
                website_bin_distribution[i] = 0

            #Add useful bins to the sample distribution
            for packet_size in lineSplit[1:-1]:
                packet_size_binned = RoundToNearest(int(packet_size), binWidth)
                if(packet_size_binned in binsUsedByWebsite):
                    website_bin_distribution[packet_size_binned] += 1


            if(trainCounter < websiteTrainInstances):
                bin_list = [] 
                for i in website_bin_distribution:
                    bin_list.append(str(website_bin_distribution[i]))
                trainData.append(",".join(bin_list) + ", " + currWebsite + "\n")
                trainCounter += 1
            elif(testCounter < websiteTestInstances):
                bin_list = [] 
                for i in website_bin_distribution:
                    bin_list.append(str(website_bin_distribution[i]))
                testData.append(",".join(bin_list) + ", " + currWebsite + "\n")
                #Account for processed sample
                testCounter += 1
    
    trainSet.write("".join(trainData))
    testSet.write("".join(testData))
    trainSet.close()
    testSet.close()


def extractDistributionWithTruncation(argv):
    BASE_DIR = os.path.dirname(argv[0])
    file = open(argv[0],'r')
    
    binWidth = int(argv[1])
    websiteToClassify = argv[2]

    if not os.path.exists(BASE_DIR + "/" + websiteToClassify):
        os.makedirs(BASE_DIR + "/" + websiteToClassify)
        
    trainSet = open(BASE_DIR + "/" + websiteToClassify + "/TrainSet_truncated_" + str(binWidth) + ".csv", 'w')
    testSet = open(BASE_DIR + "/" + websiteToClassify + "/TestSet_truncated_" + str(binWidth) + ".csv", 'w')
    

    websiteTrainInstances = int(argv[3])
    websiteTestInstances = int(argv[4])
    
    trainInstancesCounter = 0
    binsUsedByWebsite = set()
    minBucket = RoundToNearest(-1500, binWidth)
    maxBucket = RoundToNearest(1500, binWidth) + 1
    
    ################################################
    #Gather list of quantized buckets used by the target website in the training set (1st pass)
    ################################################

    l = file.readline() #Take out dataset header
    l = file.readline() #Take out dataset header
    for lineNumber, l in enumerate(file.readlines()):
        lineSplit = l.rstrip('\n').split(" ")
        if (lineNumber % 2 == 1): #Gather website data
            website = lineSplit[0][:-1]
            if (website == websiteToClassify):
                if(trainInstancesCounter < websiteTrainInstances):
                    for packet_size in lineSplit[1:-1]:
                        binsUsedByWebsite.add(RoundToNearest(int(packet_size), binWidth))
                    trainInstancesCounter += 1
                else:
                    break #We've analysed all training websiteToClassify samples


    #Get to know the amount of buckets used for measuring a given website
    print "Total number of buckets: " + str(int(math.floor(3000.0/binWidth)))
    print "Number of buckets after truncation: " + str(len(binsUsedByWebsite))
    #Write these stats to a file
    file = open("truncationInfo/" + websiteToClassify + ".csv", "a")
    file.write("%s, %s\n"%(binWidth, len(binsUsedByWebsite)))
    file.close()

    ################################################
    #Build csv with truncated bins (2nd pass)
    ################################################

    # Write CSV datasets header (with bins used by the target website)
    for size in range(minBucket, maxBucket, binWidth):
        if (size in binsUsedByWebsite):
            trainSet.write("packetLengthBin_" + str(size) + ", ")
            testSet.write("packetLengthBin_" + str(size) + ", ")
    trainSet.write("class\n")
    testSet.write("class\n")


    file = open(argv[0],'r')
    l = file.readline() #Take out dataset header
    l = file.readline() #Take out dataset header
    trainCounter = 0
    testCounter = 0
    currWebsite = ""
    trainData = []
    testData =[]

    for lineNumber, l in enumerate(file.readlines()):
        lineSplit = l.rstrip('\n').split(" ")
        if (lineNumber % 2 == 1): #Gather website data
            website = lineSplit[0][:-1]
            if(website != currWebsite):
                currWebsite = website
                trainCounter = 0
                testCounter = 0
            
            #Build container for sample distribution
            website_bin_distribution = OrderedDict()
            for i in sorted(binsUsedByWebsite):
                website_bin_distribution[i] = 0

            #Add useful bins to the sample distribution
            for packet_size in lineSplit[1:-1]:
                packet_size_binned = RoundToNearest(int(packet_size), binWidth)
                if(packet_size_binned in binsUsedByWebsite):
                    website_bin_distribution[packet_size_binned] += 1


            if(trainCounter < websiteTrainInstances):
                bin_list = [] 
                for i in website_bin_distribution:
                    bin_list.append(str(website_bin_distribution[i]))
                trainData.append(",".join(bin_list) + ", " + currWebsite + "\n")
                trainCounter += 1
            elif(testCounter < websiteTestInstances):
                bin_list = [] 
                for i in website_bin_distribution:
                    bin_list.append(str(website_bin_distribution[i]))
                testData.append(",".join(bin_list) + ", " + currWebsite + "\n")
                #Account for processed sample
                testCounter += 1
    
    trainSet.write("".join(trainData))
    testSet.write("".join(testData))
    trainSet.close()
    testSet.close()

if __name__ == "__main__":
    if (int(sys.argv[-1]) == 1):
        extractDistributionWithoutTruncation(sys.argv[1:-1])
    else:
        extractDistributionWithTruncation(sys.argv[1:-1])