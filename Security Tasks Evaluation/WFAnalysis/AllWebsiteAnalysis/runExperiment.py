import os
import math
import subprocess as sub
import shutil
import time
import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.core.converters import Loader
from weka.classifiers import Classifier
from weka.classifiers import Evaluation


dataset_location = "Data/openssh.data"

#export JAVA_HOME=/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home/
def ClassifyParam(mode, binWidths):
	if not os.path.exists("classificationResults"):
		os.makedirs("classificationResults")

	if("normal" in mode):
		file = open("classificationResults/AllVsAll.csv","w") 

		file.write("BinWidth, Accuracy\n")

		for binWidth in binWidths:

			train_set = "Data/arff/TrainSet_%s.arff"%(binWidth)
			test_set = "Data/arff/TestSet_%s.arff"%(binWidth)
			print "Loading Datasets..."

			train_data = converters.load_any_file(train_set)
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
			acc = evaluation.percent_correct/100.0
			print "Percent correct: " + str(acc)

			file.write("%s, %s\n"%(binWidth, acc))
		file.close()



def QuantizeAndCreateTrainTestDataset(binWidths):
	#2/3 train, 1/3 test (150 total, 100 -50)
	# Currently 50-50
	target_train_instances = 75
	target_test_instances = 75

	#Placeholder website for parsing script to work (compatibility issues)
	website = "www.flickr.com"

	for binWidth in binWidths:
		simArgs = "python ParsingUtilities/CSVParseWebsiteUnbalanced.py %s %s %s %s %s"%(dataset_location, binWidth, website, target_train_instances, target_test_instances)
		print "Quantizing dataset. binWidth = %s"%(binWidth)
		sub.call(simArgs, shell = True)
	
	print "Moving files to Data directory root"
	src_folder = "Data/www.flickr.com/"
	files = os.listdir(src_folder)
	for f in files:
			shutil.move(src_folder+f, "Data/")
	os.rmdir(src_folder)


def BuildQuantizedArffDatasets(mode, binWidths):
	if not os.path.exists("Data/arff"):
		os.makedirs("Data/arff")

	if("normal" in mode):
		train_set = "TrainSet"
		test_set = "TestSet"

		for binWidth in binWidths:
			simArgs = "python ParsingUtilities/CSVParseToWeka.py Data/%s_%s.csv Data/arff/%s_%s.arff"%(train_set, binWidth, train_set, binWidth)
			print "Generating train dataset. binWidth = %s"%(binWidth)
			sub.call(simArgs, shell = True)

			simArgs = "python ParsingUtilities/CSVParseToWeka.py Data/%s_%s.csv Data/arff/%s_%s.arff"%(test_set, binWidth, test_set, binWidth)
			print "Generating test dataset. binWidth = %s"%(binWidth)
			sub.call(simArgs, shell = True)




if __name__ == "__main__":
	
	#Quantization
	BIN_WIDTH = [1, 4, 8, 16, 32, 64, 128, 256]

	QuantizeAndCreateTrainTestDataset(BIN_WIDTH)


	BuildQuantizedArffDatasets("normal", BIN_WIDTH)
	
	#Classify
	#Start WEKA execution
	jvm.start(max_heap_size="4096m")
	
	#Classify
	ClassifyParam("normal", BIN_WIDTH)

	#stop weka execution
	jvm.stop()

