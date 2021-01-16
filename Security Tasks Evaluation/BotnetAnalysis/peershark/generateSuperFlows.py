from P2P_CONSTANTS import *
import socket
import Flow
import SuperFlow
import sys


def runGenerateSuperFlows(flow_data_dir, super_flow_data_dir, flowgap):
	#TIMEGAP IN SECONDS
	csvfiles = getCSVFiles(flow_data_dir)
	#print csvfiles

	flowdata = []
	for filename in csvfiles:
		inputfile = open(filename)
		data = [line.strip() for line in inputfile]
		inputfile.close()

		for eachline in data:
			fields = eachline.split(',')
			flowdata.append(SuperFlow.SuperFlow(fields))
	print '\tNo. of flows to be processed: ' + str(len(flowdata))

	
	flowdata = Flow.combineFlows(flowdata, flowgap)
	print '\tSuperflows (Flows with flowgap = ' + str(flowgap) + ' sec) : ' + str(len(flowdata))

	outfile = open(super_flow_data_dir + str(flowgap) + '.csv', 'w')
	
	to_write = []
	for flow in flowdata:
		to_write.append(
			socket.inet_ntoa(flow.ip1) + ',' +
			socket.inet_ntoa(flow.ip2) + ',' +
			str(flow.getNoOfPackets()) + ',' +
			str(flow.getNoOfBytes()) + ',' +
			'%.6f'%flow.getInterArrivaltime() + ',' +
			'%.6f'%flow.getStart() + ',' +
			'%.6f'%flow.getEnd() + ',' +
			'%.6f'%flow.getDurationInSeconds())
	outfile.write("\n".join(to_write))
	outfile.close()