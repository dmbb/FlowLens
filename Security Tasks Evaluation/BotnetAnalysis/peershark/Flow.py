from Packet import *

#input: list of packets, timegap - real number
#return val: list of flows
#
#merges collection of packets(objects) into collection of flows(many-to-one)
#Working: group packets with same ip-pair(direction irrelevant) and merge all packets for
#which |packet1.time - packet2.time| < threshold(timegap)
def packetsToFlows(packets,timegap):
	#sanity check for 0 packets 
	if len(packets) == 0:
		return None

	outputflows = []
	
	#perform a radix-sort to group together packets
	#with same ip-pairs(packet.key represents an ip-pair) 
	#and sort these packets according to timestamp
	packets.sort(key = lambda packet:packet.timestamp)
	packets.sort(key = lambda packet:packet.key)
	
	nextflow = Flow(None)
	for nextpacket in packets:
		#if ip-pairs dont match or time-difference of prev and current packet greater
		#than timegap, create a new flow 
		if (nextflow.key != nextpacket.key) or ((nextpacket.timestamp - nextflow.getEnd()) > timegap):
			nextflow = Flow(nextpacket)
			outputflows.append(nextflow)
		#if not then add packet to previous flow
		else:
			nextflow.addPacket(nextpacket)

	return outputflows

#same as function packetsToFlow but merges flows instead of packets
def combineFlows(flows, flowgap):
	if len(flows) == 0:
		return None

	outputflows = []

	flows.sort(key = lambda flow:flow.getStart())
	flows.sort(key = lambda flow:flow.key)
	
	nextoutflow = Flow(None)
	for nextflow in flows:
		if (nextoutflow.key != nextflow.key) or ((nextflow.getStart() - nextoutflow.getEnd()) > flowgap):
			nextoutflow = nextflow
			outputflows.append(nextoutflow)
		else:
			nextoutflow.addFlow(nextflow)

	return outputflows

def getCustomWeightedAvg(n1, w1, n2, w2):
	num = 0
	den = 0
	if w1 > 0:
		num += w1 * n1
		den += w1
	if w2 > 0:
		num += w2 * n2
		den	+= w2
	if den <= 0:
		den = 1
	return num / den	


#write list of flows into file in desired format
def writeFlowsToFile(flowlist, filename):
	outfile = open(filename, 'w')
	
	to_write = []
	for flow in flowlist:
		to_write.append(
			socket.inet_ntoa(flow.ip1) + ',' +
			socket.inet_ntoa(flow.ip2) + ',' +
			str(flow.n_packet1) + ',' +
			str(flow.n_byte1) + ',' +
			'%.6f'%flow.t_start1 + ',' +
			'%.6f'%flow.t_end1 + ',' +
			'%.6f'%flow.getInterArrivaltime1() + ',' + 
			str(flow.n_packet2) + ',' +
			str(flow.n_byte2) + ',' +
			'%.6f'%flow.t_start2 + ',' +
			'%.6f'%flow.t_end2 + ',' +
			'%.6f'%flow.getInterArrivaltime2())
	
	outfile.write("\n".join(to_write))
	outfile.close()

#class which defines the structure of flows
class Flow:
	#constructor of default flow
	def __init__(self,firstpacket):
		if firstpacket == None:
			self.ip1 = None
			self.ip2 = None
			self.key = None
			self.n_packet1 = 0
			self.n_byte1 = 0
			self.t_start1 = 0
			self.t_end1 = 0	
			self.t_interarrival1 = []
			self.n_packet2 = 0
			self.n_byte2 = 0	
			self.t_start2 = 0
			self.t_end2 = 0
			self.t_interarrival2 = []
		else:
			if firstpacket.source < firstpacket.dest:
				self.ip1 = firstpacket.source
				self.ip2 = firstpacket.dest
				self.n_packet1 = 1
				self.n_byte1 = firstpacket.size
				self.t_start1 = firstpacket.timestamp
				self.t_end1 = firstpacket.timestamp
				self.t_interarrival1 = []						
				self.n_packet2 = 0
				self.n_byte2 = 0	
				self.t_start2 = 0
				self.t_end2 = 0
				self.t_interarrival2 = []
			else:
				self.ip1 = firstpacket.dest
				self.ip2 = firstpacket.source
				self.n_packet1 = 0
				self.n_byte1 = 0
				self.t_start1 = 0
				self.t_end1 = 0
				self.t_interarrival1 = []
				self.n_packet2 = 1			
				self.n_byte2 = firstpacket.size				
				self.t_start2 = firstpacket.timestamp
				self.t_end2 = firstpacket.timestamp
				self.t_interarrival2 = []			
			self.key = firstpacket.key
	
	#add a flow to the current flow (by changing volume and duration)
	def addFlow(self,flow):
		self.t_interarrival1 += flow.t_interarrival1
		self.t_interarrival2 += flow.t_interarrival2
		self.n_packet1 += flow.n_packet1
		self.n_packet2 += flow.n_packet2
		self.n_byte1 += flow.n_byte1
		self.n_byte2 += flow.n_byte2
				
		temp = min(self.t_start1,flow.t_start1)
		if temp == 0:
			self.t_start1 = self.t_start1 + flow.t_start1
		else:
			self.t_start1 = temp
		
		temp = min(self.t_start2,flow.t_start2)
		if temp == 0:
			self.t_start2 = self.t_start2 + flow.t_start2
		else:
			self.t_start2 = temp
		
		if(self.t_end1 < flow.t_end1):
			self.t_end1 = flow.t_end1
		if(self.t_end2 < flow.t_end2):
			self.t_end2 = flow.t_end2
	
	#add a packet to the current flow (by changing volume and duration)
	def addPacket(self,packet):
		if packet.source == self.ip1 and packet.dest == self.ip2:			
			
			#initialize flow if not initialized
			if self.n_packet1 == 0:
				self.t_start1 = packet.timestamp
				self.t_end1 = packet.timestamp
				self.n_packet1 += 1
				self.n_byte1 += packet.size
				return

			if self.t_end1 < packet.timestamp:
				self.t_interarrival1.append(packet.timestamp-self.t_end1)
				self.t_end1 = packet.timestamp
			elif self.t_start1 > packet.timestamp:
				self.t_interarrival1.append(self.t_start1-packet.timestamp)
				self.t_start1 = packet.timestamp
			self.n_packet1 += 1
			self.n_byte1 += packet.size			
		
		elif packet.source == self.ip2 and packet.dest == self.ip1:
			
			#initialize flow if not initialized
			if self.n_packet2 == 0:
				self.t_start2 = packet.timestamp
				self.t_end2 = packet.timestamp
				self.n_packet2 += 1
				self.n_byte2 += packet.size
				return
			
			if self.t_end2 < packet.timestamp:
				self.t_interarrival2.append(packet.timestamp-self.t_end2)
				self.t_end2 = packet.timestamp
			elif self.t_start2 > packet.timestamp:
				self.t_interarrival2.append(self.t_start2-packet.timestamp)
				self.t_start2 = packet.timestamp
			self.n_packet2 += 1
			self.n_byte2 += packet.size

		else:
			raise Exception('packet does not belong to flow')
	
	def getDurationInSeconds(self):
		return self.getEnd() - self.getStart()

	def getInterArrivaltime(self):
		combined = (self.t_interarrival1+self.t_interarrival2).sort()
		if len(combined) > 0:
			return combined[len(combined)/2]
		return 0	
	
	def getInterArrivaltime1(self):
		self.t_interarrival1.sort()
		if len(self.t_interarrival1) > 0:
			return self.t_interarrival1[len(self.t_interarrival1)/2]
		return 0

	def getInterArrivaltime2(self):
		self.t_interarrival2.sort()
		if len(self.t_interarrival2) > 0:
			return self.t_interarrival2[len(self.t_interarrival2)/2]
		return 0	
	
	def getNoOfBytes(self):
		return self.n_byte1 + self.n_byte2

	def getNoOfPackets(self):
		return self.n_packet1 + self.n_packet2

	def getStart(self):
		temp =  min(self.t_start1, self.t_start2)
		if temp == 0:
			return self.t_start1 + self.t_start2
		else:
			return temp

	def getEnd(self):
		return max(self.t_end1, self.t_end2)
