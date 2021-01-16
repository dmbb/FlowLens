import os
import sys
import csv

def main(argv):
	OutputFile = open(argv[1], 'w')
	InputFile = open(argv[0])
	
	bin_dict = {}
	
	OutputFile.write("@relation\'WF\'\n\n")
	OutputFile.write("@attribute Text string\n")
	OutputFile.write("@attribute class {")
	
	csv_reader = csv.reader(InputFile, delimiter=',')

	csv_header = ""
	website_list = set()
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
		website_list.add(currWebsite)		
		bin_list = row[:-1]

		text.append("\'")
		if("Online" in argv[1]): #Fix for online Sketching (Coskun et al.)
			for i in range(len(bin_list)):
				text.append(str(bin_list[i]) + " ")
		else: #For the others
			for i in range(len(bin_list)):
				for _ in range(int(bin_list[i])):
					text.append(str(bin_dict[i]) + " ")
		
		text.append("\'," + currWebsite + "\n")

	#Write classes on header
	OutputFile.write(",".join(website_list))
	OutputFile.write("}\n\n")
	#Write data
	OutputFile.write("@data\n\n")
	OutputFile.write("".join(text))
		
	
	OutputFile.close()


if __name__ == "__main__":
	main(sys.argv[1:])