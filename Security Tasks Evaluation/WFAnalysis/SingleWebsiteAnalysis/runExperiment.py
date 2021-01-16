import os
import math
import subprocess as sub
import time
import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.core.converters import Loader
from weka.classifiers import Classifier
from weka.classifiers import Evaluation

from generateFigures import GenerateFigures

dataset_location = "Data/openssh.data"

#export JAVA_HOME=/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home/

def ClassifyParam(website, mode, binWidths, truncation_modes=["full", "truncated"]):
	if not os.path.exists("classificationResults"):
		os.makedirs("classificationResults")

	
	if("normal" in mode):
		for truncation in truncation_modes:
			file = open("classificationResults/SingleWebsite_%s_%s.csv"%(truncation, website),"w")
			file.write("BinWidth, Accuracy, FalsePositiveRate, FalseNegativeRate\n")

			for binWidth in binWidths:

				train_set_file = "TrainSet_%s_%s.arff"%(truncation, binWidth)
				train_set = "Data/%s/arff/%s"%(website, train_set_file)
				test_set = "Data/%s/arff/%s"%(website, train_set_file.replace("TrainSet", "TestSet"))

				print "Loading Datasets..."
				print "Train: " + train_set
				train_data = converters.load_any_file(train_set)
				print "Test: " + test_set
				test_data = converters.load_any_file(test_set)
				
				#Set class attribute
				train_data.class_is_last()
				test_data.class_is_last()
				print "Dataset Loaded!"


				classifier_name = "weka.classifiers.meta.FilteredClassifier"

				classifier = Classifier(classname=classifier_name, options=[
					"-F", "weka.filters.unsupervised.attribute.StringToWordVector -R first-last -W 1000 -C -T -N 1 -stemmer weka.core.stemmers.NullStemmer -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\\'\\\\\\\"()?!\\\"\"",
					"-W", "weka.classifiers.bayes.NaiveBayesMultinomial"])

				start_train = time.time()
				classifier.build_classifier(train_data)
				end_train = time.time()
				print "Train\t%s\t%s"%(binWidth, end_train-start_train)

				for index, inst in enumerate(test_data):
					if(index == 0):
						start_sample = time.time()
						classifier.classify_instance(inst)
						end_sample = time.time()
						print "Sample\t%s\t%s"%(binWidth, end_sample-start_sample)

				print "Evaluating w/ Multinomial Naive Bayes classifier. BinWidth = %s"%(binWidth)
				evaluation = Evaluation(test_data)
				start_batch = time.time()
				evaluation.test_model(classifier, test_data)
				end_batch = time.time()
				print "Batch\t%s\t%s"%(binWidth,end_batch-start_batch)
				

				print evaluation.summary()
				print evaluation.matrix()
				#Just as an example, we're measuring the fpr and fnr of the website indexed as class 1

				tp = evaluation.num_true_positives(1)
				tn = evaluation.num_true_negatives(1)
				fp = evaluation.num_false_positives(1)
				fn = evaluation.num_false_negatives(1)

				acc = (tp+tn)/float(tp+tn+fp+fn)
				fpr = evaluation.false_positive_rate(1)
				fnr = evaluation.false_negative_rate(1)
				
				print "Accuracy: %s"%(acc)
				print "False Positive Rate: %s"%(fpr)
				print "False Negative Rate: %s"%(fnr)

				file.write("%s, %s, %s, %s\n"%(binWidth, acc, fpr, fnr))
			file.close()
	


def QuantizeAndCreateUnbalancedTrainTestDataset(truncate, website, binWidths):
	#2/3 train, 1/3 test (150 total, 100 -50)
	target_train_instances = 75
	target_test_instances = 75

	if(truncate):
		truncation = 0

		#Init bookeeping of truncated bins
		if not os.path.exists("truncationInfo"):
			os.makedirs("truncationInfo")
		file = open("truncationInfo/" + website + ".csv", "w") 
		file.write("BinWidth, TruncatedBins\n")
		file.close()
	else:
		truncation = 1

	for binWidth in binWidths:
		simArgs = "python ParsingUtilities/CSVParseWebsiteUnbalanced.py %s %s %s %s %s %s"%(dataset_location, binWidth, website, target_train_instances, target_test_instances, truncation)
		print "Quantizing dataset. binWidth = %s"%(binWidth) + ", truncation = " + str(truncate) + ", website = " + website
		sub.call(simArgs, shell = True)



def BuildQuantizedArffDatasets(website, mode):
	if not os.path.exists("Data/%s/arff"%(website)):
		os.makedirs("Data/%s/arff"%(website))

	if("normal" in mode):
		for f in os.listdir("Data/%s"%(website)):
			if(".csv" in f and not f.startswith("CountMin")):

				simArgs = "python ParsingUtilities/CSVParseToWeka.py Data/%s/%s Data/%s/arff/%s %s"%(website, f, website, f[:-3] + "arff", website)
				print "Generating dataset. File = " + f[:-3] + "arff"
				sub.call(simArgs, shell = True)


if __name__ == "__main__":
	modes = ["normal", "sketch"]

	TRUNCATION_MODES = [True, False]

	#Quantization
	BIN_WIDTH = [1, 4, 8, 16, 32, 64, 128, 256]

	WEBSITES = [
	"www.citibank.de",
	"mail.google.com",
	"www.youtube.com",
	"www.amazon.com",
	"www.imdb.com",
	"www.flickr.com"
	]

	jvm.start(max_heap_size="4096m")
	for website in WEBSITES:
		for truncate in TRUNCATION_MODES:
			# Generates the train and test dataset
			#Proportion should be set inside this function
			QuantizeAndCreateUnbalancedTrainTestDataset(truncate, website, BIN_WIDTH)

		BuildQuantizedArffDatasets(website, "normal")

		"""#Delete raw datasets
		for file in os.listdir("Data/" + website):
			if(file.endswith(".csv")):
				os.remove("Data/" + website + "/" + file)"""

		#Classify
		ClassifyParam(website, "normal", BIN_WIDTH)

		"""#Delete arff datasets
		for file in os.listdir("Data/"):
			if(file.endswith(".arff")):
				os.remove("Data/" + file)"""
	
		#Generate figures
		GenerateFigures()
	jvm.stop()
