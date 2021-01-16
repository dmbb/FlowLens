import csv
import numpy as np


def CreateBinaryVectorRepresentation(BIN_WIDTH, TOPK, SKETCH_SIZE):
    
    for sketch_size in SKETCH_SIZE:
        for binWidth in BIN_WIDTH:
            for topk in TOPK:
                
                """
                Generate random base vectors
                """
                
                if(topk != 1500):
                    real_bucket_number = topk
                else:
                    real_bucket_number = 1500/binWidth
                        
                random_base_vectors = []
                for i in range(0, sketch_size):
                    random_base_vector = (2*np.random.randint(0,2,size=(real_bucket_number))-1)
                    random_base_vectors.append(random_base_vector)

                n_bits = range(0, sketch_size)

                """
                Process Phase 1 Data
                """

                feature_set = 'PL_60_' + str(binWidth) + '_' + str(topk)
                data_folder = 'FeatureSets/' + feature_set + '/'

                #Regular Traffic
                print "Online_Sketch: Phase 1, Regular - " + feature_set + "/Online_regularTraffic_phase1_" + str(sketch_size) + "_dataset.csv"
                output = open(data_folder + "Online_regularTraffic_phase1_" + str(sketch_size) + "_dataset.csv", "w") 
                f = open(data_folder + "RegularTraffic_phase1_dataset.csv", 'r')
                reader = csv.reader(f, delimiter=',')

                #Process data row
                for n, row in enumerate(reader):
                    if(n == 0):
                        output.write(",".join(str(x) for x in n_bits) + "," + row[-1] + "\n")
                    else:
                        #Gather the packet vector array (v_f)
                        packet_count_vector = []
                        for i in row[:-1]:
                            packet_count_vector.append(int(i))

                        #Compute the integer array (c_f)
                        integer_array = []
                        for i in range(0, sketch_size):
                            c_f_i = 0
                            for j in range(0, real_bucket_number):
                                #print "Random_base_vector: " + str(random_base_vectors[i])
                                c_f_i += random_base_vectors[i][j] * packet_count_vector[j]
                            integer_array.append(c_f_i)

                        #Compute the binary array (s_f)
                        binary_array = []
                        for i in integer_array:
                            if(i <= 0):
                                binary_array.append(0)
                            else:
                                binary_array.append(1)

                        #print "Binary array: " + str(binary_array)
                        output.write(",".join(str(x) for x in binary_array) + "," + row[-1] + "\n")
                output.close()


                #DeltaShaper Traffic
                print "Online_Sketch: Phase 1, DeltaShaper - " + feature_set + "/Online_deltashaperTraffic_phase1_" + str(sketch_size) + "_dataset.csv"
                output = open(data_folder + "Online_deltashaperTraffic_phase1_" + str(sketch_size) + "_dataset.csv", "w") 
                f = open(data_folder + "DeltaShaperTraffic_320_phase1_dataset.csv", 'r')
                reader = csv.reader(f, delimiter=',')

                #Process data row
                for n, row in enumerate(reader):
                    if(n == 0):
                        output.write(",".join(str(x) for x in n_bits) + "," + row[-1] + "\n")
                    else:
                        #Gather the packet vector array (v_f)
                        packet_count_vector = []
                        for i in row[:-1]:
                            packet_count_vector.append(int(i))

                        #Compute the integer array (c_f)
                        integer_array = []
                        for i in range(0, sketch_size):
                            c_f_i = 0
                            for j in range(0, real_bucket_number):
                                #print "Random_base_vector: " + str(random_base_vectors[i])
                                c_f_i += random_base_vectors[i][j] * packet_count_vector[j]
                            integer_array.append(c_f_i)

                        #Compute the binary array (s_f)
                        binary_array = []
                        for i in integer_array:
                            if(i <= 0):
                                binary_array.append(0)
                            else:
                                binary_array.append(1)

                        #print "Binary array: " + str(binary_array)
                        output.write(",".join(str(x) for x in binary_array) + "," + row[-1] + "\n")
                output.close()

                ########################################################################################
                ########################################################################################
                ########################################################################################


                """
                Process Phase 2 Data
                """

                feature_set = 'PL_60_' + str(binWidth) + '_' + str(topk)
                data_folder = 'FeatureSets/' + feature_set + '/'

                #Regular Traffic
                print "Online_Sketch: Phase 2, Regular - " + feature_set + "/Online_regularTraffic_phase2_" + str(sketch_size) + "_dataset.csv"
                output = open(data_folder + "Online_regularTraffic_phase2_" + str(sketch_size) + "_dataset.csv", "w") 
                f = open(data_folder + "RegularTraffic_phase2_dataset.csv", 'r')
                reader = csv.reader(f, delimiter=',')

                #Process data row
                for n, row in enumerate(reader):
                    if(n == 0):
                        output.write(",".join(str(x) for x in n_bits) + "," + row[-1] + "\n")
                    else:
                        #Gather the packet vector array (v_f)
                        packet_count_vector = []
                        for i in row[:-1]:
                            packet_count_vector.append(int(i))

                        #Compute the integer array (c_f)
                        integer_array = []
                        for i in range(0, sketch_size):
                            c_f_i = 0
                            for j in range(0, real_bucket_number):
                                #print "Random_base_vector: " + str(random_base_vectors[i])
                                c_f_i += random_base_vectors[i][j] * packet_count_vector[j]
                            integer_array.append(c_f_i)

                        #Compute the binary array (s_f)
                        binary_array = []
                        for i in integer_array:
                            if(i <= 0):
                                binary_array.append(0)
                            else:
                                binary_array.append(1)

                        #print "Binary array: " + str(binary_array)
                        output.write(",".join(str(x) for x in binary_array) + "," + row[-1] + "\n")
                output.close()


                #DeltaShaper Traffic
                print "Online_Sketch: Phase 2, DeltaShaper - " + feature_set + "/Online_deltashaperTraffic_phase2_" + str(sketch_size) + "_dataset.csv"
                output = open(data_folder + "Online_deltashaperTraffic_phase2_" + str(sketch_size) + "_dataset.csv", "w") 
                f = open(data_folder + "DeltaShaperTraffic_320_phase2_dataset.csv", 'r')
                reader = csv.reader(f, delimiter=',')

                #Process data row
                for n, row in enumerate(reader):
                    if(n == 0):
                        output.write(",".join(str(x) for x in n_bits) + "," + row[-1] + "\n")
                    else:
                        #Gather the packet vector array (v_f)
                        packet_count_vector = []
                        for i in row[:-1]:
                            packet_count_vector.append(int(i))

                        #Compute the integer array (c_f)
                        integer_array = []
                        for i in range(0, sketch_size):
                            c_f_i = 0
                            for j in range(0, real_bucket_number):
                                #print "Random_base_vector: " + str(random_base_vectors[i])
                                c_f_i += random_base_vectors[i][j] * packet_count_vector[j]
                            integer_array.append(c_f_i)

                        #Compute the binary array (s_f)
                        binary_array = []
                        for i in integer_array:
                            if(i <= 0):
                                binary_array.append(0)
                            else:
                                binary_array.append(1)

                        #print "Binary array: " + str(binary_array)
                        output.write(",".join(str(x) for x in binary_array) + "," + row[-1] + "\n")
                output.close()

