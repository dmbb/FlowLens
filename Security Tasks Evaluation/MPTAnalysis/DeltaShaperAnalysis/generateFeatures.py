#!/usr/bin/env python
import collections
import dpkt
import subprocess
import socket
import os
import sys
import math
import csv
import numpy as np
from itertools import product
from scipy.stats import kurtosis, skew
import time
import glob


DEST_IP = '172.31.0.2'
SOURCE_IP = '172.31.0.19'

def MergeDatasets(data_folder):
    if(os.path.exists(data_folder + '/full_dataset.csv')):
        os.remove(data_folder + '/full_dataset.csv')

    features_files = [data_folder + "deltashaper_dataset.csv", data_folder + "RegularTraffic_dataset.csv"]

    print "Merging full dataset..."
    header_saved = False
    with open(data_folder + '/full_dataset.csv','wb') as fout:
        for filename in features_files:
            print "merging " + filename
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)
    print "Dataset merged!"


def CombinedMerging(data_folder):
    if(os.path.exists(data_folder + '/regular_320_dataset.csv')):
        os.remove(data_folder + '/regular_320_dataset.csv')

    features_files = [data_folder + "DeltaShaperTraffic_320_dataset.csv", data_folder + "RegularTraffic_dataset.csv"]

    print "Merging dataset..."
    header_saved = False
    with open(data_folder + '/regular_320_dataset.csv','wb') as fout:
        for filename in features_files:
            print "merging " + filename
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)
    print "Dataset merged!"



def MergeSamples(data_folder):
    #Generate training dataset
    deltashaper_files = glob.glob(data_folder + "/DeltaShaperTraffic_*.csv")

    header_saved = False
    with open(data_folder + 'deltashaper_dataset.csv','wb') as fout:
        for filename in deltashaper_files:
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)


def GenerateDatasets(data_folder):
    MergeSamples(data_folder)
    CombinedMerging(data_folder)
    MergeDatasets(data_folder)


def RoundToNearest(n, m):
        r = n % m
        return n + m - r if r + r >= m else n - r


def FeatureExtractionPLBenchmark(sampleFolder, binWidth, topk):
    #Bucket importance in decreasing order
    BUCKETS_TO_MEASURE = []
    
    #Measure interesting buckets
    if(topk != 1500):
        #Buckets in decreasing importance order 
        f_imp = np.load('classificationResults/PL_60_' + str(binWidth) + '_1500/FeatureImportance_XGBoost_DeltaShaperTraffic_320.npy')
        #Print top k
        #for f in f_imp:
        #    print str(f[1]) + " " + str(f[2])
        
        if(topk > len(f_imp)):
            print "Skipping, not enough features to accomodate for. TopK = " + str(topk) + " Features = " + str(len(f_imp))
            return
        for i in range(0,topk):
            b = int(f_imp[i][2].split("_")[1])
            print "Top-" + str(i) + " = " + str(b)
            BUCKETS_TO_MEASURE.append(b)

    #Measure all buckets
    elif(topk == 1500):
        print "Measuring all buckets according to quantization"
        for i in range(-1500,1500,binWidth):
            BUCKETS_TO_MEASURE.append(i/binWidth)


    quantized_buckets_to_measure = sorted(BUCKETS_TO_MEASURE)
    print "Quantized buckets to measure = " + str(quantized_buckets_to_measure)
    print "Number of buckets to measure = " + str(len(quantized_buckets_to_measure))


    traceInterval = 60 #Amount of time in packet trace to consider for feature extraction
    feature_set_folder = 'FeatureSets/PL_' + str(traceInterval) + "_" + str(binWidth) + "_" + str(topk)
    print feature_set_folder

    if not os.path.exists(feature_set_folder):
                os.makedirs(feature_set_folder)
    arff_path = feature_set_folder + '/' + os.path.basename(sampleFolder) + '_dataset.csv'
    arff = open(arff_path, 'wb')
    written_header = False


    for sample in os.listdir(sampleFolder):
        if(".DS_Store" in sample):
            continue
        f = open(sampleFolder + "/" + sample + "/" + sample + ".pcap")
        pcap = dpkt.pcap.Reader(f)

        #Analyse packets transmited
        bin_dict = {}
        bin_dict2 = {}

        for i in quantized_buckets_to_measure:
            bin_dict[i] = 0


        firstTime = 0.0
        setFirst = False
        for ts, buf in pcap:
            if(not(setFirst)):
                firstTime = ts
                setFirst = True

            if(ts < (firstTime + traceInterval)):

                eth = dpkt.ethernet.Ethernet(buf)
                ip_hdr = eth.data
                try:
                    src_ip_addr_str = socket.inet_ntoa(ip_hdr.src)
                    dst_ip_addr_str = socket.inet_ntoa(ip_hdr.dst)
                    #Target UDP communication between both cluster machines
                    if (ip_hdr.p == 17 and src_ip_addr_str == SOURCE_IP):
                        binned = RoundToNearest(len(buf),binWidth)
                        if(binned/binWidth in quantized_buckets_to_measure):
                            bin_dict[binned/binWidth]+=1
                    elif(ip_hdr.p == 17 and src_ip_addr_str != SOURCE_IP):
                        binned = -RoundToNearest(len(buf),binWidth) #Incoming is negative
                        if(binned/binWidth in quantized_buckets_to_measure):
                            bin_dict[binned/binWidth]+=1
                except:
                    pass
        f.close()

        od_dict = collections.OrderedDict(sorted(bin_dict.items(), key=lambda t: float(t[0])))
        bin_list = []
        for i in od_dict:
            bin_list.append(od_dict[i])


        label = os.path.basename(sampleFolder)
        if('Regular' in sampleFolder):
            label = 'Regular'

        #Write sample features to the csv file
        f_names = []
        f_values = []

        for i, b in enumerate(bin_list):
            f_names.append('packetLengthBin_' + str(quantized_buckets_to_measure[i]))
            f_values.append(b)


        f_names.append('Class')
        f_values.append(label)

        if(not written_header):
            arff.write(', '.join(f_names))
            arff.write('\n')
            print "Writing header"
            written_header = True

        l = []
        for v in f_values:
            l.append(str(v))
        arff.write(', '.join(l))
        arff.write('\n')
    arff.close()
    return feature_set_folder




def CompressFeatures(BIN_WIDTH, TOPK):
    sampleFolders = [
    "TrafficCaptures/480Resolution/DeltaShaperTraffic_320",
    "TrafficCaptures/480Resolution/RegularTraffic",
    ]
    

    if not os.path.exists('FeatureSets'):
                os.makedirs('FeatureSets')
    
    for topk in TOPK:
        for binWidth in BIN_WIDTH:
            print "\n#####################################"
            print "Generating Dataset based on Binned Packet Length Features"
            start = time.time()
            for sampleFolder in sampleFolders:
                print "\n#############################"
                print "Parsing " + sampleFolder
                print "#############################"
                feature_set_folder = FeatureExtractionPLBenchmark(sampleFolder, binWidth, topk)
            if(feature_set_folder is not None):
                GenerateDatasets(feature_set_folder + '/')
            end = time.time()
            print "Optimize_compress_bin_%s_topk_%s_time_%s"%(binWidth, topk, end-start)




def SplitDataset(DATASET_SPLIT, N_FLOWS, COVERT_FLOWS_PERC):
    print "Splitting datasets with DATASET_SPLIT= %s, N_FLOWS = %s, REG_FLOWS_PROP = %s"%(DATASET_SPLIT, N_FLOWS, COVERT_FLOWS_PERC)
    split_value = DATASET_SPLIT * N_FLOWS #samples
    covert_split_value = COVERT_FLOWS_PERC * split_value

    print "SPLIT_VALUE = %s"%(split_value)
    print "COVERT_SAMPLES_VALUE = %s"%(covert_split_value)


    for feature_folder in os.listdir("FeatureSets"):
        start = time.time()
        if(".DS_Store" not in feature_folder):
            print "Splitting %s"%("FeatureSets/" + feature_folder + "/RegularTraffic_dataset.csv")
            #Split RegularFlows
            RegularFile = open("FeatureSets/" + feature_folder + "/RegularTraffic_dataset.csv", 'rb')
            csv_reader = csv.reader(RegularFile, delimiter=',')

            PhaseOneRegularFile = open("FeatureSets/" + feature_folder + "/RegularTraffic_phase1_dataset.csv", 'w')
            PhaseTwoRegularFile = open("FeatureSets/" + feature_folder + "/RegularTraffic_phase2_dataset.csv", 'w')

            for n, row in enumerate(csv_reader):
                if(n == 0):
                    row_string = ",".join(row) + "\n"
                    PhaseOneRegularFile.write(row_string)
                    PhaseTwoRegularFile.write(row_string)
                elif(n < split_value):
                    row_string = ",".join(row) + "\n"
                    PhaseOneRegularFile.write(row_string)
                else:
                    row_string = ",".join(row) + "\n"
                    PhaseTwoRegularFile.write(row_string)

            RegularFile.close()
            PhaseOneRegularFile.close()
            PhaseTwoRegularFile.close()


            #Split CovertFlows
            print "Splitting %s"%("FeatureSets/" + feature_folder + "/DeltaShaperTraffic_320_dataset.csv")
            CovertFile = open("FeatureSets/" + feature_folder + "/DeltaShaperTraffic_320_dataset.csv", "rb")
            csv_reader = csv.reader(CovertFile, delimiter=',')

            PhaseOneCovertFile = open("FeatureSets/" + feature_folder + "/DeltaShaperTraffic_320_phase1_dataset.csv", "w")
            PhaseTwoCovertFile = open("FeatureSets/" + feature_folder +  "/DeltaShaperTraffic_320_phase2_dataset.csv", "w")

            for n, row in enumerate(csv_reader):
                if(n == 0):
                    row_string = ",".join(row) + "\n"
                    PhaseOneCovertFile.write(row_string)
                    PhaseTwoCovertFile.write(row_string)
                elif(n < covert_split_value):
                    row_string = ",".join(row) + "\n"
                    PhaseOneCovertFile.write(row_string)
                elif(n > split_value and n < split_value + covert_split_value):
                    row_string = ",".join(row) + "\n"
                    PhaseTwoCovertFile.write(row_string)

            CovertFile.close()
            PhaseOneCovertFile.close()
            PhaseTwoCovertFile.close()
            end = time.time()
            binWidth = feature_folder.split("_")[2]
            topk = feature_folder.split("_")[3]
            print "Optimize_split_bin_%s_topk_%s_time_%s"%(binWidth, topk, end-start)



def MergeTestData():
    for feature_folder in os.listdir("FeatureSets"):
        if(".DS_Store" not in feature_folder):
            print "Merging %s"%("FeatureSets/" + feature_folder + "/RegularTraffic_phase2_dataset.csv")
            print "Merging %s"%("FeatureSets/" + feature_folder + "/DeltaShaperTraffic_320_phase2_dataset.csv")

            #Merging Phase2
            PhaseTwoFile = open("FeatureSets/" + feature_folder + "/Phase2_dataset.csv", 'w')
            
            PhaseTwoRegularFile = open("FeatureSets/" + feature_folder + "/RegularTraffic_phase2_dataset.csv", 'rb')
            PhaseTwoCovertFile = open("FeatureSets/" + feature_folder +  "/DeltaShaperTraffic_320_phase2_dataset.csv", "rb")


            #Write data from the regular file
            csv_reader = csv.reader(PhaseTwoRegularFile, delimiter=',')
            for n, row in enumerate(csv_reader):
                row_string = ",".join(row) + "\n"
                PhaseTwoFile.write(row_string)
            
            #Write data from the covert file
            csv_reader = csv.reader(PhaseTwoCovertFile, delimiter=',')
            for n, row in enumerate(csv_reader):
                if(n == 0):
                    continue
                row_string = ",".join(row) + "\n"
                PhaseTwoFile.write(row_string)
            
            PhaseTwoFile.close()
            PhaseTwoRegularFile.close()
            PhaseTwoCovertFile.close()



def FeatureExtractionPLBenchmarkBasedOnTrainData(sampleFolder, binWidth, topk):
    #Bucket importance in decreasing order
    BUCKETS_TO_MEASURE = []
    
    #Measure interesting buckets
    if(topk != 1500):
        #Buckets in decreasing importance order 
        f_imp = np.load('classificationResults/PL_60_' + str(binWidth) + '_1500/FeatureImportance_XGBoost_DeltaShaperTraffic_320_phase1.npy')
        #Print top k
        #for f in f_imp:
        #    print str(f[1]) + " " + str(f[2])
        
        if(topk > len(f_imp)):
            print "Skipping, not enough features to accomodate for. TopK = " + str(topk) + " Features = " + str(len(f_imp))
            return
        for i in range(0,topk):
            b = int(f_imp[i][2].split("_")[1])
            print "Top-" + str(i) + " = " + str(b)
            BUCKETS_TO_MEASURE.append(b)

    #Measure all buckets
    elif(topk == 1500):
        print "Measuring all buckets according to quantization"
        for i in range(-1500,1500,binWidth):
            BUCKETS_TO_MEASURE.append(i/binWidth)


    quantized_buckets_to_measure = sorted(BUCKETS_TO_MEASURE)
    print "Quantized buckets to measure = " + str(quantized_buckets_to_measure)
    print "Number of buckets to measure = " + str(len(quantized_buckets_to_measure))


    traceInterval = 60 #Amount of time in packet trace to consider for feature extraction
    feature_set_folder = 'FeatureSets/PL_' + str(traceInterval) + "_" + str(binWidth) + "_" + str(topk)
    print feature_set_folder

    if not os.path.exists(feature_set_folder):
                os.makedirs(feature_set_folder)
    arff_path = feature_set_folder + '/' + os.path.basename(sampleFolder) + '_dataset.csv'
    arff = open(arff_path, 'wb')
    written_header = False


    for sample in os.listdir(sampleFolder):
        if(".DS_Store" in sample):
            continue
        f = open(sampleFolder + "/" + sample + "/" + sample + ".pcap")
        pcap = dpkt.pcap.Reader(f)

        #Analyse packets transmited
        bin_dict = {}
        

        for i in quantized_buckets_to_measure:
            bin_dict[i] = 0


        firstTime = 0.0
        setFirst = False
        for ts, buf in pcap:
            if(not(setFirst)):
                firstTime = ts
                setFirst = True

            if(ts < (firstTime + traceInterval)):

                eth = dpkt.ethernet.Ethernet(buf)
                ip_hdr = eth.data
                try:
                    src_ip_addr_str = socket.inet_ntoa(ip_hdr.src)
                    dst_ip_addr_str = socket.inet_ntoa(ip_hdr.dst)
                    #Target UDP communication between both cluster machines
                    if (ip_hdr.p == 17 and src_ip_addr_str == SOURCE_IP):
                        binned = RoundToNearest(len(buf),binWidth)
                        if(binned/binWidth in quantized_buckets_to_measure):
                            bin_dict[binned/binWidth]+=1
                    elif(ip_hdr.p == 17 and src_ip_addr_str != SOURCE_IP):
                        binned = -RoundToNearest(len(buf),binWidth) #Incoming is negative
                        if(binned/binWidth in quantized_buckets_to_measure):
                            bin_dict[binned/binWidth]+=1
                except:
                    pass
        f.close()

        od_dict = collections.OrderedDict(sorted(bin_dict.items(), key=lambda t: float(t[0])))
        bin_list = []
        for i in od_dict:
            bin_list.append(od_dict[i])


        label = os.path.basename(sampleFolder)
        if('Regular' in sampleFolder):
            label = 'Regular'

        #Write sample features to the csv file
        f_names = []
        f_values = []

        for i, b in enumerate(bin_list):
            f_names.append('packetLengthBin_' + str(quantized_buckets_to_measure[i]))
            f_values.append(b)


        f_names.append('Class')
        f_values.append(label)

        if(not written_header):
            arff.write(', '.join(f_names))
            arff.write('\n')
            print "Writing header"
            written_header = True

        l = []
        for v in f_values:
            l.append(str(v))
        arff.write(', '.join(l))
        arff.write('\n')
    arff.close()
    return feature_set_folder



def CompressFeaturesBasedOnTrainData(BIN_WIDTH, TOPK):
    sampleFolders = [
    "TrafficCaptures/480Resolution/DeltaShaperTraffic_320",
    "TrafficCaptures/480Resolution/RegularTraffic",
    ]


    if not os.path.exists('FeatureSets'):
                os.makedirs('FeatureSets')
    
    for topk in TOPK:
        for binWidth in BIN_WIDTH:
            start = time.time()
            print "\n#####################################"
            print "Generating Dataset based on Binned Packet Length Features"
            for sampleFolder in sampleFolders:
                print "\n#############################"
                print "Parsing " + sampleFolder
                print "#############################"
                feature_set_folder = FeatureExtractionPLBenchmarkBasedOnTrainData(sampleFolder, binWidth, topk)
            if(feature_set_folder is not None):
                GenerateDatasets(feature_set_folder + '/')
            end = time.time()
            print "Optimize_compress_bin_%s_topk_%s_time_%s"%(binWidth, topk, end-start)


def ExtractFirstNPackets(sampleFolder, number_of_packets):

    traceInterval = 60 #Amount of time in packet trace to consider for feature extraction
    feature_set_folder = 'FeatureSets/First_%d_packets'%(number_of_packets)
    print feature_set_folder

    if not os.path.exists(feature_set_folder):
                os.makedirs(feature_set_folder)
    arff_path = feature_set_folder + '/' + os.path.basename(sampleFolder) + '_dataset.csv'
    arff = open(arff_path, 'wb')
    written_header = False


    for sample in os.listdir(sampleFolder):
        if(".DS_Store" in sample):
            continue
        f = open(sampleFolder + "/" + sample + "/" + sample + ".pcap")
        pcap = dpkt.pcap.Reader(f)


        packet_array1 = []
        packet_array2 = []
        firstTime = 0.0
        setFirst = False
        for ts, buf in pcap:
            if(len(packet_array1) >= number_of_packets and len(packet_array2) >= number_of_packets):
                break

            if(not(setFirst)):
                firstTime = ts
                setFirst = True

            if(ts < (firstTime + traceInterval)):

                eth = dpkt.ethernet.Ethernet(buf)
                ip_hdr = eth.data
                try:
                    src_ip_addr_str = socket.inet_ntoa(ip_hdr.src)
                    dst_ip_addr_str = socket.inet_ntoa(ip_hdr.dst)
                    #Target UDP communication between both cluster machines
                    if (ip_hdr.p == 17 and src_ip_addr_str == SOURCE_IP):
                        if(len(packet_array1) < number_of_packets):
                            packet_array1.append(len(buf))
                    elif(ip_hdr.p == 17 and src_ip_addr_str != SOURCE_IP):
                        if(len(packet_array2) < number_of_packets):
                            packet_array2.append(len(buf))
                except:
                    pass
        f.close()

        label = os.path.basename(sampleFolder)
        if('Regular' in sampleFolder):
            label = 'Regular'

        if(len(packet_array1) >= number_of_packets and len(packet_array2) >= number_of_packets):
            #Write sample features to the csv file
            f_names = []
            f_values = []

            for i, b in enumerate(packet_array1):
                f_names.append('packetNumberOut_' + str(i))
                f_values.append(b)

            for i, b in enumerate(packet_array2):
                f_names.append('packetNumberIn_' + str(i))
                f_values.append(b)


            f_names.append('Class')
            f_values.append(label)

            if(not written_header):
                arff.write(', '.join(f_names))
                arff.write('\n')
                print "Writing header"
                written_header = True

            l = []
            for v in f_values:
                l.append(str(v))
            arff.write(', '.join(l))
            arff.write('\n')
        else:
            print "Sample %s has not enough packets"%(sampleFolder + "/" + sample + "/" + sample + ".pcap")
    arff.close()
    return feature_set_folder




def ExtractPacketSample(NUMBER_OF_PACKETS):
    sampleFolders = [
    "TrafficCaptures/480Resolution/DeltaShaperTraffic_320",
    "TrafficCaptures/480Resolution/RegularTraffic",
    ]

    if not os.path.exists('FeatureSets'):
                os.makedirs('FeatureSets')
    
    for number_of_packets in NUMBER_OF_PACKETS:
        print "\n#####################################"
        print "Extracting first %d packet sizes"%(number_of_packets)

        for sampleFolder in sampleFolders:
            print "\n#############################"
            print "Parsing " + sampleFolder
            print "#############################"
            feature_set_folder = ExtractFirstNPackets(sampleFolder, number_of_packets)
        if(feature_set_folder is not None):
            GenerateDatasets(feature_set_folder + '/')
