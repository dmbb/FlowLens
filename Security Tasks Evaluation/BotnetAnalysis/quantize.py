import os
import csv
import multiprocessing as MP
import socket
import gc
import time

def RoundToNearest(n, m):
		r = n % m
		return n + m - r if r + r >= m else n - r

STORM_IPS = [
	"66.154.80.101", 
	"66.154.80.105",
	"66.154.80.111",
	"66.154.80.125",
	"66.154.83.107",
	"66.154.83.113",
	"66.154.83.138",
	"66.154.83.80",
	"66.154.87.39",
	"66.154.87.41",
	"66.154.87.57",
	"66.154.87.58",
	"66.154.87.61"
]

WALEDAC_IPS = [
	"192.168.58.136", 
	"192.168.58.137", 
	"192.168.58.150"
	]


def runQuantization(dataset, traffic_capture, binWidth, ipt_bin_width, sem):
	sem.acquire()

	cap_file = open(dataset + "/" + traffic_capture, 'rb')
	csv_reader = csv.reader(cap_file, delimiter=',')

	quantized_csv = open('FeatureSets/' + os.path.basename(dataset) + "/" + traffic_capture[:-4] + "_" + str(binWidth) + "_" + str(ipt_bin_width) + ".csv", "w")

	malicious_ips = []
	if(os.path.basename(dataset) == "Storm"):
		malicious_ips = STORM_IPS
	elif(os.path.basename(dataset) == "Waledac"):
		malicious_ips = WALEDAC_IPS

	#print "Malicious IPs = %s"%(malicious_ips)
	#print os.path.basename(dataset)

	to_write = []
	#Write modified packets
	for row in csv_reader:
		#Filter out non-malicious flows from Storm and Waledac datasets
		if(("Storm" in os.path.basename(dataset) or "Waledac" in os.path.basename(dataset)) and (row[0] not in malicious_ips and row[1] not in malicious_ips)):
			#print "Row not in malicious: %s - %s"%(row[0], row[1])
			continue
		else:
			new_row = row
			
			#Quantize packet size
			new_row[4] = str(RoundToNearest(int(new_row[4]), binWidth))

			#Quantize Timestamp
			if(ipt_bin_width > 0):
				new_row[3] = str(RoundToNearest(int(float(new_row[3])), ipt_bin_width))
			to_write.append(",".join(new_row))
	
	quantized_csv.write("\n".join(to_write))
	
	cap_file.close()
	quantized_csv.close()

	#start_collect = time.time()
	#collected = gc.collect()
	#end_collect = time.time()
	#print "Time wasted on GC - Quantize: %ss, collected %s objects"%(end_collect-start_collect, collected)
	sem.release()


def QuantizeDataset(dataset, binWidth, ipt_bin_width, n_processes):
	sem = MP.Semaphore(n_processes)
	traffic_captures = os.listdir(dataset)

	tasklist = []

	for traffic_capture in traffic_captures:
		task = MP.Process(target = runQuantization, args = (dataset, traffic_capture, binWidth, ipt_bin_width, sem))
		tasklist.append(task)

	print "Tasklist size = %s"%(len(tasklist))

	# #execute commands in parallel
	for i in range(0, len(tasklist), n_processes):
		for k,task in enumerate(tasklist[i:i+n_processes]):
			tasklist[i+k].start()
		for k, task in enumerate(tasklist[i:i+n_processes]):
			tasklist[i+k].join()
			#print "Joined task number %s"%(i+k)
