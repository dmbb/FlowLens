PCAPDATADIR = './pcapdata/'
PCAPFILES = 'PcapInputFiles.txt'
TSHARKOPTIONSFILE = 'TsharkOptions.txt'
TCP_PROTO = '6'
UDP_PROTO = '17'
UDP_HEADERLENGTH = 8

#utility functions
import os
def getCSVFiles(dirname):
	csvfiles = []
	for eachfile in os.listdir(dirname):
		if eachfile.endswith('.csv'):
			csvfiles.append(dirname + eachfile)	
	return csvfiles