import os
import csv
import numpy as np
from scipy import interp
import random
from random import shuffle
import math
import time
#Classifiers
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#Eval Metrics
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_score
from joblib import dump, load


def gatherHoldoutData(data_folder, cfg, load_model, sketch_size=0, sigma_param=0, number_of_packets=0, compress_ratio=0):
	SPLIT_FACTOR = 0.9
	#Load Datasets
	if(sketch_size == 0 and sigma_param == 0 and number_of_packets == 0 and compress_ratio == 0): #Regular Quant
		f = open(data_folder + cfg[0] + "_dataset.csv", 'r')
	elif(sketch_size != 0 and sigma_param == 0 and number_of_packets !=0): # Sketching
		f = open(data_folder + cfg[0] + "_" + str(sketch_size) + "_dataset.csv", 'r')
	elif(sketch_size == 0 and sigma_param != 0 and number_of_packets == 0): # Compressive Gaussian
		f = open(data_folder + cfg[0] + "_" + str(sigma_param) + "_" + str(compress_ratio) + "_dataset.csv", 'r')
	elif(sketch_size == 0 and sigma_param == 0 and number_of_packets == 0 and compress_ratio != 0): #Compressive Bernoulli
		f = open(data_folder + cfg[0] + "_" + str(compress_ratio) + "_dataset.csv", 'r')
	
	reader = csv.reader(f, delimiter=',')
	reg = list(reader)
	

	if(sketch_size == 0 and sigma_param == 0 and number_of_packets == 0 and compress_ratio == 0): #Regular Quant
		f = open(data_folder + cfg[1] + "_dataset.csv", 'r')
	elif(sketch_size != 0 and sigma_param == 0 and number_of_packets !=0): # Sketching
		f = open(data_folder + cfg[1] + "_" + str(sketch_size) + "_dataset.csv", 'r')
	elif(sketch_size == 0 and sigma_param != 0 and number_of_packets == 0): # Compressive Gaussian
		f = open(data_folder + cfg[1] + "_" + str(sigma_param) + "_" + str(compress_ratio) + "_dataset.csv", 'r')
	elif(sketch_size == 0 and sigma_param == 0 and number_of_packets == 0 and compress_ratio != 0): #Compressive Bernoulli
		f = open(data_folder + cfg[1] + "_" + str(compress_ratio) + "_dataset.csv", 'r')
	reader = csv.reader(f, delimiter=',')
	fac = list(reader)
	
	print "###########################################"
	print "Configuration " + cfg[1]
	print "###########################################"

	#Convert data to floats (and labels to integers)
	features_id = reg[0]
	reg_data = []
	for i in reg[1:]:
		int_array = []
		for pl in i[:-1]:
			int_array.append(float(pl))
		int_array.append(0)
		reg_data.append(int_array)

	fac_data = []
	for i in fac[1:]:
		int_array = []
		for pl in i[:-1]:
			int_array.append(float(pl))
		int_array.append(1)
		fac_data.append(int_array)


	#Shuffle both datasets
	shuffled_reg_data = random.sample(reg_data, len(reg_data))
	shuffled_fac_data = random.sample(fac_data, len(fac_data))

	#Build label tensors
	reg_labels = []
	for i in shuffled_reg_data:
		reg_labels.append(int(i[len(reg_data[0])-1]))

	fac_labels = []
	for i in shuffled_fac_data:
		fac_labels.append(int(i[len(reg_data[0])-1]))

	#Take label out of data tensors
	for i in range(0, len(shuffled_reg_data)):
		shuffled_reg_data[i].pop()

	for i in range(0, len(shuffled_fac_data)):
		shuffled_fac_data[i].pop()


	#Build training and testing datasets
	#Split each class data in the appropriate proportion for training
	reg_proportion_index = int(len(reg_labels)* SPLIT_FACTOR)
	reg_train_x = shuffled_reg_data[:reg_proportion_index]
	reg_train_y = reg_labels[:reg_proportion_index]
	

	fac_proportion_index = int(len(fac_labels)*SPLIT_FACTOR)
	fac_train_x = shuffled_fac_data[:fac_proportion_index]
	fac_train_y = fac_labels[:fac_proportion_index]


	#Create training sets by combining the randomly selected samples from each class
	train_x = reg_train_x + fac_train_x
	train_y = reg_train_y + fac_train_y

	if(not load_model):
		#Make the split for the testing data
		reg_test_x = shuffled_reg_data[reg_proportion_index:]
		reg_test_y = reg_labels[reg_proportion_index:]

		fac_test_x = shuffled_fac_data[fac_proportion_index:]
		fac_test_y = fac_labels[fac_proportion_index:]
	else:
		REG_CLASS_PROP = 1
		reg_class_imbalance_index = int(len(reg_labels)* REG_CLASS_PROP)
		reg_test_x = shuffled_reg_data[:reg_class_imbalance_index]
		reg_test_y = reg_labels[:reg_class_imbalance_index]

		COV_CLASS_PROP = 1
		cov_class_imbalance_index = int(len(fac_labels)* COV_CLASS_PROP)
		fac_test_x = shuffled_fac_data[:cov_class_imbalance_index]
		fac_test_y = fac_labels[:cov_class_imbalance_index]
	
	#Create testing set by combining the holdout samples
	test_x = reg_test_x + fac_test_x
	test_y = reg_test_y + fac_test_y

	print "Regular samples (Train = %s, Test = %s) "%(len(reg_train_x), len(reg_test_x))
	print "Covert samples (Train = %s, Test = %s) "%(len(fac_train_x), len(fac_test_x))
	return train_x, train_y, test_x, test_y, features_id


def runClassification_Holdout_acc(data_folder, feature_set, cfg,classifier, store_folder, load_model, sketch_size=0, sigma_param=0, number_of_packets=0, compress_ratio=0):
	np.random.seed(1)
	random.seed(1)
	train_x, train_y, test_x, test_y, features_id = gatherHoldoutData(data_folder, cfg, load_model, sketch_size, sigma_param, number_of_packets, compress_ratio)

	model = classifier[0]
	clf_name = classifier[1]
	print clf_name
	
	#Decide whether to load a previously generated model or to generate a new one
	if("Online" in cfg[0]):
		model_data = store_folder + '/' + feature_set + "/online_model_" + str(sketch_size) + ".data"
	elif("CompressiveGaussian" in cfg[0]):
		model_data = store_folder + '/' + feature_set + "/compressiveGaussian_model_" + str(sigma_param) + "_" + str(compress_ratio) + ".data"
	elif("CompressiveBernoulli" in cfg[0]):
		model_data = store_folder + '/' + feature_set + "/compressiveBernoulli_model_" + str(compress_ratio) + ".data"
	else:
		model_data = store_folder + '/' + feature_set + "/model.data"

	if("VanillaCompressive" not in cfg[0]):
		binWidth = feature_set.split("_")[2]
		truncation = feature_set.split("_")[3]

	if(load_model):
		#Load previously trained model from file
		print "Loading " + model_data
		model = load(model_data)
	else:
		#Train model
		start_train = time.time()
		model = model.fit(np.asarray(train_x), np.asarray(train_y))
		end_train = time.time()
		if("Compressive" not in cfg[0]):
			if("1500" in truncation):
				print "Train\t%s\t%s"%(binWidth, end_train-start_train)
		#Save model
		print "Saving " + model_data
		dump(model, model_data)

	
	#Perform predictions
	print "Predicting %s samples"%(len(test_x))
	start_batch = time.time()
	predictions = model.predict(np.asarray(test_x))
	end_batch = time.time()
	if("Compressive" not in cfg[0]):
		if("1500" in truncation):
			print "Batch\t%s\t%s"%(binWidth, end_batch-start_batch)

	#Generate metrics
	TN, FP, FN, TP = confusion_matrix(np.asarray(test_y), predictions).ravel()
	ACC = (float(TP)+float(TN))/(float(TP)+float(TN)+float(FP)+float(FN))
	FPR = float(FP)/(float(FP)+float(TN))
	FNR = float(FN)/(float(FN)+float(TP))
	AUC = roc_auc_score(np.asarray(test_y), predictions)

	########################### Feature Importance ##########################
	f_imp = model.feature_importances_
	bin_number = list(range(len(train_x[0])))

	#print importances
	f_imp = zip(bin_number,f_imp,features_id)
	f_imp.sort(key = lambda t: t[1], reverse=True)
	
	print "Features considered: " + str(len(f_imp))
	np.save(store_folder + '/' + feature_set + "/FeatureImportance_" + clf_name + "_" + cfg[1], np.array(f_imp))

	file = open('classificationResults/' + feature_set + "/FeatureImportance_" + clf_name + "_" + cfg[1] + ".txt","w") 
	for f in f_imp:
		file.write("Importance: %f, Feature: %s\n" % (f[1], f[2]))
	file.close() 

	
	print "Model Acc: " + "{0:.3f}".format(ACC)
	print "Model FPR: " + "{0:.3f}".format(FPR)
	print "Model FNR: " + "{0:.3f}".format(FNR)
	print "Model AUC: " + "{0:.3f}".format(AUC)
	print ""
	
	return ACC, FPR, FNR, AUC



def gatherAllData(data_folder, cfg, dataset_fraction):
	#Load Datasets
	f = open(data_folder + cfg[0] + "_dataset.csv", 'r')
	reader = csv.reader(f, delimiter=',')
	reg = list(reader)
	reg = reg[:int(dataset_fraction*len(reg))]


	f = open(data_folder + cfg[1] + "_dataset.csv", 'r')
	reader = csv.reader(f, delimiter=',')
	fac = list(reader)
	fac = fac[:int(dataset_fraction*len(fac))]

	print "###########################################"
	print "Configuration " + cfg[1]
	print "###########################################"

	#Convert data to floats (and labels to integers)
	features_id = reg[0]
	reg_data = []
	for i in reg[1:]:
		int_array = []
		for pl in i[:-1]:
			int_array.append(float(pl))
		int_array.append(0)
		reg_data.append(int_array)

	fac_data = []
	for i in fac[1:]:
		int_array = []
		for pl in i[:-1]:
			int_array.append(float(pl))
		int_array.append(1)
		fac_data.append(int_array)


	#Shuffle both datasets
	shuffled_reg_data = random.sample(reg_data, len(reg_data))
	shuffled_fac_data = random.sample(fac_data, len(fac_data))

	#Build label tensors
	reg_labels = []
	for i in shuffled_reg_data:
		reg_labels.append(int(i[len(reg_data[0])-1]))

	fac_labels = []
	for i in shuffled_fac_data:
		fac_labels.append(int(i[len(reg_data[0])-1]))

	#Take label out of data tensors
	for i in range(0, len(shuffled_reg_data)):
		shuffled_reg_data[i].pop()

	for i in range(0, len(shuffled_fac_data)):
		shuffled_fac_data[i].pop()

	#Create training sets by combining the randomly selected samples from each class
	train_x = shuffled_reg_data + shuffled_fac_data
	train_y = reg_labels + fac_labels

	#Shuffle positive/negative samples for CV purposes
	x_shuf = []
	y_shuf = []
	index_shuf = range(len(train_x))
	shuffle(index_shuf)
	for i in index_shuf:
		x_shuf.append(train_x[i])
		y_shuf.append(train_y[i])

	return x_shuf, y_shuf, features_id

def runClassification_CV_acc(data_folder, feature_set, cfg,classifier, store_folder):
	np.random.seed(1)
	random.seed(1)
	dataset_fraction = 1.0
	train_x, train_y, features_id = gatherAllData(data_folder, cfg, dataset_fraction)

	model = classifier[0]
	clf_name = classifier[1]

	#Report Cross-Validation Accuracy
	#scores = cross_val_score(model, np.asarray(train_x), np.asarray(train_y), cv=10)
	print clf_name
	#print "Avg. Accuracy: " + str(sum(scores)/float(len(scores)))

	cv = StratifiedKFold(n_splits=10)
	fnrs = []
	fprs = []
	accuracies = []
	importances = []

	#Split the data in k-folds, perform classification, and report acc
	for train, test in cv.split(train_x, train_y):

		model = model.fit(np.asarray(train_x)[train], np.asarray(train_y)[train])
		predictions = model.predict(np.asarray(train_x)[test])

		TN, FP, FN, TP = confusion_matrix(np.asarray(train_y)[test], predictions).ravel()
		acc = (float(TP)+float(TN))/(float(TP)+float(TN)+float(FP)+float(FN))
		#print "Fold ACC: " + str(acc)
		#print "Fold TN: " + str(TN)
		#print "Fold TP: " + str(TP)
		#print "Fold FN: " + str(FN)
		#print "Fold FP: " + str(FP)
		
		FPR = float(FP)/(float(FP)+float(TN))
		FNR = float(FN)/(float(FN)+float(TP))
		fnrs.append(FNR)
		fprs.append(FPR)

		accuracies.append(acc)
		f_imp = model.feature_importances_
		importances.append(f_imp)


	########################### Mean Feature Importance ##########################
	bin_number = list(range(len(train_x[0])))
	mean_importances = []
	for n in range(0,len(importances[0])):
		mean_imp = (importances[0][n] + importances[1][n] + importances[2][n] + importances[3][n] + importances[4][n] + importances[5][n] + importances[6][n] + importances[7][n] + importances[8][n] + importances[9][n])/10.0
		mean_importances.append(mean_imp)
	#print mean_importances
	f_imp = zip(bin_number,mean_importances,features_id)
	f_imp.sort(key = lambda t: t[1], reverse=True)
	print "Features considered: " + str(len(f_imp))
	np.save(store_folder + '/' + feature_set + "/FeatureImportance_" + clf_name + "_" + cfg[1], np.array(f_imp))

	file = open('classificationResults/' + feature_set + "/FeatureImportance_" + clf_name + "_" + cfg[1] + ".txt","w") 
	for f in f_imp:
		file.write("Importance: %f, Feature: %s\n" % (f[1], f[2]))
	file.close() 
	#for f in f_imp:
	#    print "Importance: %f, Feature: %s" % (f[1], f[2])


	########################### Mean Performance #################################
	mean_acc = np.mean(accuracies, axis=0)
	mean_fpr = np.mean(fprs)
	mean_fnr = np.mean(fnrs)
	
	print "Model Acc: " + "{0:.3f}".format(mean_acc)
	print "Model FPR: " + "{0:.3f}".format(mean_fpr)
	print "Model FNR: " + "{0:.3f}".format(mean_fnr)
	print ""
	
	return mean_acc, mean_fpr, mean_fnr


def GenerateFeatureImportanceBasedOnTrainData(mode, binWidths, topk_features, deltas=0, memory_factors=0):
	store_folder = 'classificationResults'
	if not os.path.exists(store_folder):
				os.makedirs(store_folder)

	load_model = False

	if("normal" in mode):

		cfgs = [
		["RegularTraffic_phase1",
		"DeltaShaperTraffic_320_phase1"]]

		classifiers = [
		[XGBClassifier(),"XGBoost"]]

		print "\n================================================="
		print "XGBoost - Packet Length Features - Quantization"
		print "================================================="
		for binWidth in binWidths:
			for topk in topk_features:
				start = time.time()
				feature_set = 'PL_60_' + str(binWidth) + '_' + str(topk)
				print feature_set

				#If invalid, skip and add placeholder for figure purposes
				if(topk != 1500 and topk > 1500/binWidth):
					print "Skipping sample, invalid configuration. TopK = " + str(topk) + " Total Features = " + str(1500/binWidth)
					continue

				data_folder = 'FeatureSets/' + feature_set + '/'
				if not os.path.exists(store_folder + '/' + feature_set):
						os.makedirs(store_folder + '/' + feature_set)
							
				for cfg in cfgs:
					for classifier in classifiers:
						print "Running classifiers for " + cfg[0] + " and " + cfg[1]
						acc, fpr, fnr, auc = runClassification_Holdout_acc(data_folder, feature_set, cfg, classifier, store_folder, load_model)
						to_save = []
						to_save.append(acc)
						to_save.append(fpr)
						to_save.append(fnr)
						to_save.append(auc)
						np.save(store_folder + '/' + feature_set + '/' + "classificationResults_phase1_NoSketch", np.array(to_save))
				
				end = time.time()
				print "Optimize_modelBuild_bin_%s_topk_%s_time_%s"%(binWidth, topk, end-start)



def ClassifyTestDataBasedOnModel(mode, binWidths, topk_features, n_flows, sketch_sizes=[], deltas=0, memory_factors=0, sigma_params=[], number_of_packets=[], compressive_ratio=[]):
	store_folder = 'classificationResults'
	if not os.path.exists(store_folder):
				os.makedirs(store_folder)

	load_model = True

	if("normal" in mode):

		cfgs = [
		["RegularTraffic_phase2",
		"DeltaShaperTraffic_320_phase2"]]

		classifiers = [
		[XGBClassifier(),"XGBoost"]]

		print "\n================================================="
		print "XGBoost - Packet Length Features - Quantization"
		print "================================================="
		for binWidth in binWidths:
			for topk in topk_features:

				feature_set = 'PL_60_' + str(binWidth) + '_' + str(topk)
				print feature_set

				#If invalid, skip and add placeholder for figure purposes
				if(topk != 1500 and topk > 1500/binWidth):
					print "Skipping sample, invalid configuration. TopK = " + str(topk) + " Total Features = " + str(1500/binWidth)
					continue

				data_folder = 'FeatureSets/' + feature_set + '/'
				if not os.path.exists(store_folder + '/' + feature_set):
						os.makedirs(store_folder + '/' + feature_set)
							
				for cfg in cfgs:
					for classifier in classifiers:
						print "Running classifiers for " + cfg[0] + " and " + cfg[1]
						acc, fpr, fnr, auc = runClassification_Holdout_acc(data_folder, feature_set, cfg, classifier, store_folder, load_model)
						to_save = []
						to_save.append(acc)
						to_save.append(fpr)
						to_save.append(fnr)
						to_save.append(auc)
						np.save(store_folder + '/' + feature_set + '/' + "classificationResults_phase2_NoSketch", np.array(to_save))



	if("online" in mode):
		cfgs = [
		["Online_regularTraffic_phase2",
		"Online_deltashaperTraffic_phase2"]]

		classifiers = [
		[XGBClassifier(),"XGBoost"]]

		print "\n================================================="
		print "XGBoost - Packet Length Features - Quantization"
		print "================================================="
		for sketch_size in sketch_sizes:
			for binWidth in binWidths:
				for topk in topk_features:

					feature_set = 'PL_60_' + str(binWidth) + '_' + str(topk)
					print feature_set

					#If invalid, skip and add placeholder for figure purposes
					if(topk != 1500 and topk > 1500/binWidth):
						print "Skipping sample, invalid configuration. TopK = " + str(topk) + " Total Features = " + str(1500/binWidth)
						continue

					data_folder = 'FeatureSets/' + feature_set + '/'
					if not os.path.exists(store_folder + '/' + feature_set):
							os.makedirs(store_folder + '/' + feature_set)
								
					for cfg in cfgs:
						for classifier in classifiers:
							print "Running classifiers for " + cfg[0] + "_" + str(sketch_size) + " and " + cfg[1] + "_" + str(sketch_size)
							acc, fpr, fnr, auc = runClassification_Holdout_acc(data_folder, feature_set, cfg, classifier, store_folder, load_model, sketch_size)
							to_save = []
							to_save.append(acc)
							to_save.append(fpr)
							to_save.append(fnr)
							to_save.append(auc)
							np.save(store_folder + '/' + feature_set + '/' + "classificationResults_phase2_OnlineSketch_" + str(sketch_size), np.array(to_save))


	if(mode == "compressive_gaussian"):
		cfgs = [
		["CompressiveGaussian_regularTraffic_phase2",
		"CompressiveGaussian_deltashaperTraffic_phase2"]]

		classifiers = [
		[XGBClassifier(),"XGBoost"]]
	
		print "\n================================================="
		print "XGBoost - Packet Length Features - CompressiveGaussian"
		print "================================================="

		for compress_ratio in compressive_ratio:
			for binWidth in binWidths:
				for topk in topk_features:
					for sigma_param in sigma_params:
						feature_set = 'PL_60_' + str(binWidth) + '_' + str(topk)
						print feature_set

						#If invalid, skip and add placeholder for figure purposes
						if(topk != 1500 and topk > 1500/binWidth):
							print "Skipping sample, invalid configuration. TopK = " + str(topk) + " Total Features = " + str(1500/binWidth)
							continue

						data_folder = 'FeatureSets/' + feature_set + '/'
						if not os.path.exists(store_folder + '/' + feature_set):
								os.makedirs(store_folder + '/' + feature_set)

						for cfg in cfgs:
							for classifier in classifiers:
								if(os.path.exists(data_folder + cfg[0] + "_" + str(sigma_param) + "_" + str(compress_ratio) + "_dataset.csv")):
									print "Running classifiers for " + cfg[0] + "_" + str(sigma_param) + "_" + str(compress_ratio) + " and " + cfg[1] + "_" + str(sigma_param) + "_" + str(compress_ratio)
									acc, fpr, fnr, auc = runClassification_Holdout_acc(data_folder, feature_set, cfg, classifier, store_folder, load_model, 0, sigma_param, 0, compress_ratio)
									to_save = []
									to_save.append(acc)
									to_save.append(fpr)
									to_save.append(fnr)
									to_save.append(auc)
									np.save(store_folder + '/' + feature_set + '/' + "classificationResults_phase2_CompressiveGaussian_" + str(sigma_param) + "_" + str(compress_ratio), np.array(to_save))
								else:
									print data_folder + cfg[0] + "_" + str(sigma_param) + "_" + str(compress_ratio) + "_dataset.csv is not in place."
									print "We couldn't fit this compression factor."

	if(mode == "compressive_bernoulli"):
		cfgs = [
		["CompressiveBernoulli_regularTraffic_phase2",
		"CompressiveBernoulli_deltashaperTraffic_phase2"]]

		classifiers = [
		[XGBClassifier(),"XGBoost"]]
	
		print "\n================================================="
		print "XGBoost - Packet Length Features - CompressiveBernoulli"
		print "================================================="
		
		for compress_ratio in compressive_ratio:
			for binWidth in binWidths:
				for topk in topk_features:
					feature_set = 'PL_60_' + str(binWidth) + '_' + str(topk)
					print feature_set

					#If invalid, skip and add placeholder for figure purposes
					if(topk != 1500 and topk > 1500/binWidth):
						print "Skipping sample, invalid configuration. TopK = " + str(topk) + " Total Features = " + str(1500/binWidth)
						continue

					data_folder = 'FeatureSets/' + feature_set + '/'
					if not os.path.exists(store_folder + '/' + feature_set):
							os.makedirs(store_folder + '/' + feature_set)

					for cfg in cfgs:
						for classifier in classifiers:
							if(os.path.exists(data_folder + cfg[0] + "_" + str(compress_ratio) + "_dataset.csv")):
								print "Running classifiers for " + cfg[0] + "_" + str(compress_ratio) + " and " + cfg[1] + "_" + str(compress_ratio)
								acc, fpr, fnr, auc = runClassification_Holdout_acc(data_folder, feature_set, cfg, classifier, store_folder, load_model, 0, 0, 0, compress_ratio)
								to_save = []
								to_save.append(acc)
								to_save.append(fpr)
								to_save.append(fnr)
								to_save.append(auc)
								np.save(store_folder + '/' + feature_set + '/' + "classificationResults_phase2_CompressiveBernoulli_" + str(compress_ratio), np.array(to_save))
							else:
								print data_folder + cfg[0] + "_" + str(compress_ratio) + "_dataset.csv is not in place."
								print "We couldn't fit this compression factor."




def BuildModelBasedOnTrainData(mode, binWidths, topk_features, sketch_sizes=[], sigma_params=[], number_of_packets=[], compressive_ratio=[]):
	store_folder = 'classificationResults'
	if not os.path.exists(store_folder):
				os.makedirs(store_folder)

	load_model = False

	if("normal" in mode):

		cfgs = [
		["RegularTraffic_phase1",
		"DeltaShaperTraffic_320_phase1"]]

		classifiers = [
		[XGBClassifier(),"XGBoost"]]

		print "\n================================================="
		print "XGBoost - Packet Length Features - Quantization"
		print "================================================="
		for binWidth in binWidths:
			for topk in topk_features:
				start = time.time()
				feature_set = 'PL_60_' + str(binWidth) + '_' + str(topk)
				print feature_set

				#If invalid, skip and add placeholder for figure purposes
				if(topk != 1500 and topk > 1500/binWidth):
					print "Skipping sample, invalid configuration. TopK = " + str(topk) + " Total Features = " + str(1500/binWidth)
					continue

				data_folder = 'FeatureSets/' + feature_set + '/'
				if not os.path.exists(store_folder + '/' + feature_set):
						os.makedirs(store_folder + '/' + feature_set)

				for cfg in cfgs:
					for classifier in classifiers:
						print "Running classifiers for " + cfg[0] + " and " + cfg[1]
						runClassification_Holdout_acc(data_folder, feature_set, cfg, classifier, store_folder, load_model)
				end = time.time()
				print "Optimize_modelBuild_bin_%s_topk_%s_time_%s"%(binWidth, topk, end-start)


	if("online" in mode):
		cfgs = [
		["Online_regularTraffic_phase1",
		"Online_deltashaperTraffic_phase1"]]

		classifiers = [
		[XGBClassifier(),"XGBoost"]]
	
		print "\n================================================="
		print "XGBoost - Packet Length Features - Quantization"
		print "================================================="
		for sketch_size in sketch_sizes:
			for binWidth in binWidths:
				for topk in topk_features:
					feature_set = 'PL_60_' + str(binWidth) + '_' + str(topk)
					print feature_set

					#If invalid, skip and add placeholder for figure purposes
					if(topk != 1500 and topk > 1500/binWidth):
						print "Skipping sample, invalid configuration. TopK = " + str(topk) + " Total Features = " + str(1500/binWidth)
						continue

					data_folder = 'FeatureSets/' + feature_set + '/'
					if not os.path.exists(store_folder + '/' + feature_set):
							os.makedirs(store_folder + '/' + feature_set)

					for cfg in cfgs:
						for classifier in classifiers:
							print "Running classifiers for " + cfg[0] + "_" + str(sketch_size) + " and " + cfg[1] + "_" + str(sketch_size)
							runClassification_Holdout_acc(data_folder, feature_set, cfg, classifier, store_folder, load_model, sketch_size)

	
	if(mode == "compressive_gaussian"):
		cfgs = [
		["CompressiveGaussian_regularTraffic_phase1",
		"CompressiveGaussian_deltashaperTraffic_phase1"]]

		classifiers = [
		[XGBClassifier(),"XGBoost"]]
	
		print "\n===================================================="
		print "XGBoost - Packet Length Features - CompressiveGaussian"
		print "======================================================"
		
		for compress_ratio in compressive_ratio:
			for binWidth in binWidths:
				for topk in topk_features:
					for sigma_param in sigma_params:
						feature_set = 'PL_60_' + str(binWidth) + '_' + str(topk)
						print feature_set

						data_folder = 'FeatureSets/' + feature_set + '/'
						if not os.path.exists(store_folder + '/' + feature_set):
								os.makedirs(store_folder + '/' + feature_set)
						
						for cfg in cfgs:
							for classifier in classifiers:
								if(os.path.exists(data_folder + cfg[0] + "_" + str(sigma_param) + "_" + str(compress_ratio) + "_dataset.csv")):
									print "Running classifiers for " + cfg[0] + "_" + str(sigma_param) + "_" + str(compress_ratio) + " and " + cfg[1] + "_" + str(sigma_param) + "_" + str(compress_ratio)
									runClassification_Holdout_acc(data_folder, feature_set, cfg, classifier, store_folder, load_model, 0, sigma_param, 0, compress_ratio)
								else:
									print data_folder + cfg[0] + "_" + str(sigma_param) + "_" + str(compress_ratio) + "_dataset.csv is not in place."
									print "We couldn't fit this compression factor."
	

	if(mode == "compressive_bernoulli"):
		cfgs = [
		["CompressiveBernoulli_regularTraffic_phase1",
		"CompressiveBernoulli_deltashaperTraffic_phase1"]]

		classifiers = [
		[XGBClassifier(),"XGBoost"]]
	
		print "\n====================================================="
		print "XGBoost - Packet Length Features - CompressiveBernoulli"
		print "======================================================="
		
		for compress_ratio in compressive_ratio:
			for binWidth in binWidths:
				for topk in topk_features:
					feature_set = 'PL_60_' + str(binWidth) + '_' + str(topk)
					print feature_set

					data_folder = 'FeatureSets/' + feature_set + '/'
					if not os.path.exists(store_folder + '/' + feature_set):
							os.makedirs(store_folder + '/' + feature_set)

					for cfg in cfgs:
						for classifier in classifiers:
							if(os.path.exists(data_folder + cfg[0] + "_" + str(compress_ratio) + "_dataset.csv")):
								print "Running classifiers for " + cfg[0] + "_" + str(compress_ratio) + " and " + cfg[1] + "_" + str(compress_ratio)
								runClassification_Holdout_acc(data_folder, feature_set, cfg, classifier, store_folder, load_model, 0, 0, 0, compress_ratio)
							else:
								print data_folder + cfg[0] + "_" + str(compress_ratio) + "_dataset.csv is not in place."
								print "We couldn't fit this compression factor."