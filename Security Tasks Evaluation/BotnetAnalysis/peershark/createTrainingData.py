from P2P_CONSTANTS import *


def runTrainingDataGenerator(super_flow_data_dir, training_data_dir, bin_width, ipt_bin_width):
	#takes takes 50,000 examples and puts it in necessary format for training
	csvfiles = []
	if os.path.isdir(super_flow_data_dir):
		csvfiles += getCSVFiles(super_flow_data_dir)

	#print ".csv files to generate training data: %s"%(csvfiles)

	outfile = open(training_data_dir + 'trainingdata_' + str(bin_width) + "_" + str(ipt_bin_width) + '.csv','w')
	for filename in csvfiles:
		label = filename.split('/')[-2]
		inputfile = open(filename)
		line = inputfile.readline().strip()
		while line!='':
			fields = line.split(',')
			if float(fields[4])!=0 and float(fields[3])!=0 and float(fields[7])!=0:
				outfile.write(
					fields[2] + ',' +
					fields[3] + ',' +
					fields[4] + ',' +
					fields[7] + ',' +
					label + '\n')
			line = inputfile.readline().strip()
		inputfile.close()
	outfile.close()