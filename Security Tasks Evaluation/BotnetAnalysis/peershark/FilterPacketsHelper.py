from P2P_CONSTANTS import *
import os
#return a list of filenames of pcapfiles taken from InputFiles.txt
#if a directory is found then all *.pcap files in the directory are
#included(non-recursive)

def getPCapFileNames():
	pcapInputFile = open(PCAPFILES)
	lines = [eachline.strip() for eachline in pcapInputFile]
	pcapInputFile.close()
	
	pcapfilenames = []
	for eachline in lines:
		if eachline.endswith('.pcap'):
			if os.path.exists(eachline):
				pcapfilenames.append(eachline)
			else:
				print eachline + ' does not exist'
				exit()
		else:
			if os.path.isdir(eachline):
				for eachfile in os.listdir(eachline):
					if eachfile.endswith('.pcap'):
						pcapfilenames.append(eachline.rstrip('/') + '/' + eachfile)
			else:
				print eachline + ' is not a directory'
				exit()
	return pcapfilenames

#return a list of options to be used with tshark
def getTsharkOptions():
	optionsFile = open(TSHARKOPTIONSFILE)
	options = [line.strip() for line in optionsFile]
	optionsFile.close()
	return options

#return a tuple (x,y) where
#x = complete tshark command
#y = output csv filename
def contructTsharkCommand(filename,tsharkOptions):
	command = 'tshark -r ' + filename + ' '
	for eachstring in tsharkOptions:
		command = command + eachstring + ' '
	
	#construct output filename
	outfilename = filename.split('/')
	outfilename = PCAPDATADIR + outfilename[len(outfilename)-1] + '.csv'
	
	command += '>'+outfilename
	return (command,outfilename)

#remove missing tcp and udp payload lengths and subtract
#8 bytes from udp payload to account for udp header
#returns a list of strings to be printed
def preprocess(data):
	outputdata = []
	for eachline in data:
		fields = eachline.split(',')
		
		#sanity check for 6 fields. Has to be changed if tshark options are changed
		if len(fields) != 6:
			continue

		tcppayload = fields[4].strip()
		udppayload = fields[5].strip()

		#subtract udp header length	
		if udppayload != '':
			fields[5] = str(int(udppayload) - UDP_HEADERLENGTH)
			if fields[5] == '0':
				continue
		#ignore packet if both tcp and udp payload lengths are null
		elif tcppayload == '' or tcppayload == '0':
			continue

		#add all valid fields to output list
		for eachfield in fields:
			if eachfield.strip() != '':
				outputdata.append(eachfield)
				outputdata.append(',')
		outputdata.pop()
		outputdata.append('\n')
	return outputdata