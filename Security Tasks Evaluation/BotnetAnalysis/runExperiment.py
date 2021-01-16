import os
import sys
import math
import subprocess as sub
import shutil
import csv
import numpy as np
import multiprocessing as MP
import time

import gc

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

from peershark.GenerateFlows import runGenerateFlows
from peershark.generateSuperFlows import runGenerateSuperFlows
from peershark.createTrainingData import runTrainingDataGenerator
from quantize import QuantizeDataset

data_location = "Data/"


def Classify(binWidth, ipt_bin_width):
	dataset_path = 'TrainingData/Datasets/Dataset_%s_%s.csv'%(binWidth, ipt_bin_width)
	with open(dataset_path, 'rb') as dataset_file:
		print "Loading Dataset: %s ..."%(dataset_path)

		attributes = []
		labels = []
		csv_reader = csv.reader(dataset_file)
		for n, row in enumerate(csv_reader):
			if(n == 0):
				continue
			else:
				attributes.append(row[:-1])
				labels.append(row[-1])
		
		#Split data in 66% train, 33% test
		train_x, test_x, train_y, test_y = train_test_split(attributes, labels, test_size=0.33, random_state=42, stratify=labels)

		#Define classifier
		classifier = RandomForestClassifier(random_state=42)

		#Train classifier
		#start_train = time.time()
		model = classifier.fit(np.asarray(train_x), np.asarray(train_y))
		#end_train = time.time()
		#print "Model trained in %ss"%(end_train-start_train)

		#for sample in test_x:
		#	start_sample = time.time()
		#	model.predict(np.asarray(sample).reshape((1,-1)))
		#	end_sample = time.time()
		#	print "Sample predicted in %ss"%(end_sample-start_sample)
		
		#Perform predictions
		print "Predicting %s samples"%(len(test_x))
		#start_batch = time.time()
		predictions = model.predict(np.asarray(test_x))
		#end_batch = time.time()
		#print "Batch predicted in %ss"%(end_batch-start_batch)

		#Generate metrics (benign)
		TN, FP, FN, TP = confusion_matrix(np.asarray(test_y), predictions, labels=["malicious","benign"]).ravel()
		FPR_BENIGN = float(FP)/(float(FP)+float(TN))
		RECALL_BENIGN = float(TP)/(float(TP) + float(FN))
		PRECISION_BENIGN = float(TP)/(float(TP) + float(FP))

		print "Model Precision (benign): " + "{0:.3f}".format(PRECISION_BENIGN)
		print "Model Recall (benign): " + "{0:.3f}".format(RECALL_BENIGN)
		print "Model FPR (benign): " + "{0:.3f}".format(FPR_BENIGN)
		

		#Generate metrics (malicious)
		TN, FP, FN, TP = confusion_matrix(np.asarray(test_y), predictions, labels=["benign","malicious"]).ravel()
		FPR_MALICIOUS = float(FP)/(float(FP)+float(TN))
		RECALL_MALICIOUS = float(TP)/(float(TP) + float(FN))
		PRECISION_MALICIOUS = float(TP)/(float(TP) + float(FP))

		print "Model Precision (malicious): " + "{0:.3f}".format(PRECISION_MALICIOUS)
		print "Model Recall (malicious): " + "{0:.3f}".format(RECALL_MALICIOUS)
		print "Model FPR (malicious): " + "{0:.3f}".format(FPR_MALICIOUS)

		results_file = open("classificationResults/results.csv","a") 
		results_file.write("%s, %s, %s, %s, %s, %s, %s, %s\n"%(binWidth, ipt_bin_width, "{0:.3f}".format(PRECISION_BENIGN), "{0:.3f}".format(RECALL_BENIGN), "{0:.3f}".format(FPR_BENIGN), "{0:.3f}".format(PRECISION_MALICIOUS), "{0:.3f}".format(RECALL_MALICIOUS), "{0:.3f}".format(FPR_MALICIOUS)))
		results_file.flush()
		results_file.close()
		print ""


def GenerateDataset(datasets, binWidth, ipt_bin_width):
	if not os.path.exists('TrainingData/Datasets'):
				os.makedirs('TrainingData/Datasets')
	
	datasets_to_merge = []
	for dataset in datasets:
		dataset = os.path.basename(dataset)
		datasets_to_merge.append('TrainingData/%s/trainingdata_%s_%s.csv'%(dataset, binWidth, ipt_bin_width))

	#Merge datasets in a single file
	with open('TrainingData/Datasets/Dataset_%s_%s.csv'%(binWidth, ipt_bin_width), "w") as out_dataset:
		out_dataset.write("NumberOfPackets,TotalBytesTransmitted,MedianIPT,ConversationDuration,class\n")
		for fname in datasets_to_merge:
			with open(fname, 'rb') as infile:
				csv_reader = csv.reader(infile)
				for row in csv_reader:
					new_row = row
					if(row[4] == "P2PTraffic"):
						new_row[4] = "benign"
					else:
						new_row[4] = "malicious"
					out_dataset.write(",".join(new_row) + "\n")


def RunPeerShark(quantized_pcap_data_dir, flow_data_dir, super_flow_data_dir, training_data_dir, bin_width, ipt_bin_width):
	#create a semaphore so as not to exceed threadlimit
	n_processes = 4

	#Set TIMEGAP 
	timegap = 2000

	print "Generating Flows with TIMEGAP = %s"%(timegap)
	runGenerateFlows(quantized_pcap_data_dir, flow_data_dir, n_processes, timegap)

	#Set FLOWGAP in seconds
	flowgap = 3600

	print "Generating SuperFlows with FLOWGAP = %s"%(flowgap)
	runGenerateSuperFlows(flow_data_dir, super_flow_data_dir, flowgap)

	print "Generating Training Data..."
	runTrainingDataGenerator(super_flow_data_dir, training_data_dir, bin_width, ipt_bin_width)


def Experiment(datasets, bin_width, ipt_bin_width):

	if not os.path.exists('FeatureSets'):
				os.makedirs('FeatureSets')

	#Quantize datasets according to bin width
	#Generate training sets for quantization
	for dataset in datasets:
		quantized_pcap_data_dir = 'FeatureSets/' + os.path.basename(dataset) + "/"
		flow_data_dir = 'FlowData/' + os.path.basename(dataset) + "/"
		superflow_data_dir = 'SuperFlowData/' + os.path.basename(dataset) + "/"
		training_data_dir = 'TrainingData/' + os.path.basename(dataset) + "/"

		if not os.path.exists('FeatureSets/' + os.path.basename(dataset)):
			os.makedirs('FeatureSets/' + os.path.basename(dataset))
		
		if not os.path.exists('FlowData/' + os.path.basename(dataset)):
			os.makedirs('FlowData/' + os.path.basename(dataset))
		
		if not os.path.exists('SuperFlowData/' + os.path.basename(dataset)):
			os.makedirs('SuperFlowData/' + os.path.basename(dataset))
		
		if not os.path.exists('TrainingData/' + os.path.basename(dataset)):
			os.makedirs('TrainingData/' + os.path.basename(dataset))


		print "Quantizing %s with BinWidth = %s and IPT_BinWidth = %s"% (dataset, binWidth, ipt_bin_width)
		n_processes = 4
		QuantizeDataset(dataset, bin_width, ipt_bin_width, n_processes)
		RunPeerShark(quantized_pcap_data_dir, flow_data_dir, superflow_data_dir, training_data_dir, bin_width, ipt_bin_width)

	print "Building Dataset..."
	GenerateDataset(datasets, binWidth, ipt_bin_width)

	print "Performing Classification..."
	Classify(binWidth, ipt_bin_width)
	
	start_collect = time.time()
	collected = gc.collect()
	end_collect = time.time()
	print "Time wasted on GC - Classification: %ss, collected %s objects"%(end_collect-start_collect, collected)

	shutil.rmtree('FeatureSets')
	shutil.rmtree('FlowData')
	shutil.rmtree('SuperFlowData')
	shutil.rmtree('TrainingData')



if __name__ == "__main__":
	
	DATASETS = [
	data_location + "Waledac",
	data_location + "Storm",
	data_location + "P2PTraffic"
	]

	###
	#The following parameters are now fed by the fullRun.sh shell script
	# Please run fullRun.sh instead of this file directly
	###

	#Quantization (packet size)
	#BIN_WIDTH = [1, 16, 32, 64, 128, 256]
	
	#Quantization (IPT in seconds)
	#TIMEGAP IS 2000s, FLOWGAP IS 3600s
	#IPT_BIN_WIDTH = [0, 1, 10, 60, 300, 900]

	if not os.path.exists("classificationResults"):
		os.makedirs("classificationResults")
	results_file = open("classificationResults/results.csv","a+") 
	results_file.write("BinWidth, IPT_BinWidth, Precision_Benign, Recall_Benign, FalsePositiveRate_Benign, Precision_Malicious, Recall_Malicious, FalsePositiveRate_Malicious\n")
	results_file.flush()
	results_file.close()
	

	binWidth = int(sys.argv[1])
	ipt_bin_width = int(sys.argv[2])

	print "Starting experiment with Bin width %s and IPT Bin Width %s"%(binWidth, ipt_bin_width)
	start_time = time.time()
	Experiment(DATASETS, binWidth, ipt_bin_width)
	end_time = time.time()
	time_elapsed_seconds = end_time - start_time
	print "Experiment finished in %sh\n"%("{0:.2f}".format(time_elapsed_seconds/60.0/60.0))

