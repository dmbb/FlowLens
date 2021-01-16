import os
import sys
import csv

def RoundToNearest(n, m):
	if (m == 1):
		return n    
	if (n > 0):
		r = n % m
		return n + m - r if r + r >= m else n - r
	else:
		if (n < 0):
			return RoundToNearest(abs(n), m) * -1
	return 0

def main(argv):
	OutputFile = open(argv[1], 'w')
	InputFile = open(argv[0], 'rb')
	website = argv[2]

	bin_dict = {}
	
	OutputFile.write("@relation\'WF\'\n\n")
	OutputFile.write("@attribute Text string\n")
	OutputFile.write("@attribute class {Nope,%s}\n\n"%(website))
	OutputFile.write("@data\n\n")

	
	csv_reader = csv.reader(InputFile, delimiter=',')

	csv_header = ""
	text = []

	for n, row in enumerate(csv_reader):
		if(n == 0):
			#Init bin dict
			csv_header = row
			prefix = "packetLengthBin_"
			for i in range(len(csv_header)-1):
				parsedBucketSize = csv_header[i][(len(prefix) + 1):]
				bin_dict[i] = parsedBucketSize
			continue
		 
		currWebsite = row[-1]		
		bin_list = row[:-1]

		text.append("\'")		
		for i in range(len(bin_list)):
			for _ in range(int(bin_list[i])):
				text.append(str(bin_dict[i]) + " ")
		
		if (website not in currWebsite):
			text.append("\'," + "Nope" + "\n")
		else:
			text.append("\'," + website + "\n")
	

	OutputFile.write("".join(text))
	OutputFile.close()

if __name__ == "__main__":
	main(sys.argv[1:])