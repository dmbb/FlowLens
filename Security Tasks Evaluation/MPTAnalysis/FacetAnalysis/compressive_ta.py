import csv
import numpy as np
import os
import math


def CreateCompressiveRepresentation(MODE, BIN_WIDTH, TOPK, SIGMA_PARAM, COMPRESSIVE_RATIO):

    for compressive_ratio in COMPRESSIVE_RATIO:
        for binWidth in BIN_WIDTH:
            for topk in TOPK:
                feature_set = 'PL_60_' + str(binWidth) + '_' + str(topk)
                data_folder = 'FeatureSets/' + feature_set + '/'

                #Sensing matrix parameters
                N = 0
                f = open(data_folder + "RegularTraffic_phase1_dataset.csv", 'r')
                reader = csv.reader(f, delimiter=',')
                for n, row in enumerate(reader):
                            if(n == 0):
                                N = len(row) -1 #Read number of bins from file
                f.close()

                M = N/compressive_ratio

                if(M < 1):
                    print "Cannot compress further(features = %d, ratio = %d), only 1 feature left"%(N, compressive_ratio)
                    continue

                np.random.seed(1)

                print "Compressive Ratio: %d"%(compressive_ratio)
                print "M: %d"%(M)
                print "N: %d"%(N)

                ######################################
                # GAUSSIAN SENSING MATRIX
                ######################################
                if MODE == "compressive_gaussian":
                    print "Start Compressive Gaussian Representation"
                    for sigma_param in SIGMA_PARAM:
                        
                        """
                        Generate sensing matrix
                        """

                        sensing_matrix = np.random.normal(0,1,(M,N))

                        """
                        Process Phase 1 Data
                        """

                        #Regular Traffic
                        print "Compressive Gaussian: Phase 1, Regular - " + feature_set + "/CompressiveGaussian_regularTraffic_phase1_" + str(sigma_param) + "_" + str(compressive_ratio) + "_dataset.csv"
                        output = open(data_folder + "CompressiveGaussian_regularTraffic_phase1_" + str(sigma_param) + "_" + str(compressive_ratio) + "_dataset.csv", "w") 
                        f = open(data_folder + "RegularTraffic_phase1_dataset.csv", 'r')
                        reader = csv.reader(f, delimiter=',')

                        #Process data row
                        for n, row in enumerate(reader):
                            if(n == 0):
                                output.write(",".join(str(x) for x in range(0,M)) + "," + row[-1] + "\n")
                            else:
                                #Gather the first n packets array
                                first_n_packets_vector = []
                                for i in row[:-1]:
                                    first_n_packets_vector.append(int(i))

                                compressed_vector = np.matmul(sensing_matrix, first_n_packets_vector)

                                #print "Compressed vector: " + str(compressed_vector)
                                output.write(",".join(str(x) for x in compressed_vector) + "," + row[-1] + "\n")
                        output.close()


                        #Facet Traffic
                        print "Compressive Gaussian: Phase 1, Facet - " + feature_set + "/CompressiveGaussian_facetTraffic_phase1_" + str(sigma_param) + "_" + str(compressive_ratio) + "_dataset.csv"
                        output = open(data_folder + "CompressiveGaussian_facetTraffic_phase1_" + str(sigma_param) + "_" + str(compressive_ratio) + "_dataset.csv", "w") 
                        f = open(data_folder + "FacetTraffic_50_phase1_dataset.csv", 'r')
                        reader = csv.reader(f, delimiter=',')

                        #Process data row
                        for n, row in enumerate(reader):
                            if(n == 0):
                                output.write(",".join(str(x) for x in range(0,M)) + "," + row[-1] + "\n")
                            else:
                                #Gather the first n packets array
                                first_n_packets_vector = []
                                for i in row[:-1]:
                                    first_n_packets_vector.append(int(i))

                                compressed_vector = np.matmul(sensing_matrix, first_n_packets_vector)

                                #print "Compressed vector: " + str(compressed_vector)
                                output.write(",".join(str(x) for x in compressed_vector) + "," + row[-1] + "\n")
                        output.close()

                        ########################################################################################
                        ########################################################################################
                        ########################################################################################


                        """
                        Process Phase 2 Data
                        """

                        #Regular Traffic
                        print "Compressive Gaussian: Phase 2, Regular - " + feature_set + "/CompressiveGaussian_regularTraffic_phase2_" + str(sigma_param) + "_"  + str(compressive_ratio) + "_dataset.csv"
                        output = open(data_folder + "CompressiveGaussian_regularTraffic_phase2_" + str(sigma_param) + "_" + str(compressive_ratio) + "_dataset.csv", "w") 
                        f = open(data_folder + "RegularTraffic_phase2_dataset.csv", 'r')
                        reader = csv.reader(f, delimiter=',')

                        #Process data row
                        for n, row in enumerate(reader):
                            if(n == 0):
                                output.write(",".join(str(x) for x in range(0,M)) + "," + row[-1] + "\n")
                            else:
                                #Gather the first n packets array
                                first_n_packets_vector = []
                                for i in row[:-1]:
                                    first_n_packets_vector.append(int(i))

                                compressed_vector = np.matmul(sensing_matrix, first_n_packets_vector)

                                #print "Compressed vector: " + str(compressed_vector)
                                output.write(",".join(str(x) for x in compressed_vector) + "," + row[-1] + "\n")
                        output.close()


                        #Facet Traffic
                        print "Compressive Gaussian Phase 2, Facet - " + feature_set + "/CompressiveGaussian_facetTraffic_phase2_" + str(sigma_param) + "_" + "_" + str(compressive_ratio) + "_dataset.csv"
                        output = open(data_folder + "CompressiveGaussian_facetTraffic_phase2_" + str(sigma_param) + "_" + str(compressive_ratio) + "_dataset.csv", "w") 
                        f = open(data_folder + "FacetTraffic_50_phase2_dataset.csv", 'r')
                        reader = csv.reader(f, delimiter=',')

                        #Process data row
                        for n, row in enumerate(reader):
                            if(n == 0):
                                output.write(",".join(str(x) for x in range(0,M)) + "," + row[-1] + "\n")
                            else:
                                #Gather the first n packets array
                                first_n_packets_vector = []
                                for i in row[:-1]:
                                    first_n_packets_vector.append(int(i))

                                compressed_vector = np.matmul(sensing_matrix, first_n_packets_vector)

                                #print "Compressed vector: " + str(compressed_vector)
                                output.write(",".join(str(x) for x in compressed_vector) + "," + row[-1] + "\n")
                        output.close()

                ######################################
                # BERNOULLI SENSING MATRIX
                ######################################
                elif MODE == "compressive_bernoulli":
                    print "Start Compressive Bernoulli Representation"

                    """
                    Generate sensing matrix
                    """
                    values = [-1/float(math.sqrt(N)), 1/float(math.sqrt(N))]
                    sensing_matrix = np.random.choice(values,(M,N), p=[0.5, 0.5])


                    """
                    Process Phase 1 Data
                    """

                    #Regular Traffic
                    print "Compressive Bernoulli: Phase 1, Regular - " + feature_set + "/CompressiveBernoulli_regularTraffic_phase1_" + str(compressive_ratio) + "_dataset.csv"
                    output = open(data_folder + "CompressiveBernoulli_regularTraffic_phase1_" + str(compressive_ratio) + "_dataset.csv", "w") 
                    f = open(data_folder + "RegularTraffic_phase1_dataset.csv", 'r')
                    reader = csv.reader(f, delimiter=',')

                    #Process data row
                    for n, row in enumerate(reader):
                        if(n == 0):
                            output.write(",".join(str(x) for x in range(0,M)) + "," + row[-1] + "\n")
                        else:
                            #Gather the first n packets array
                            first_n_packets_vector = []
                            for i in row[:-1]:
                                first_n_packets_vector.append(int(i))

                            compressed_vector = np.matmul(sensing_matrix, first_n_packets_vector)

                            #print "Compressed vector: " + str(compressed_vector)
                            output.write(",".join(str(x) for x in compressed_vector) + "," + row[-1] + "\n")
                    output.close()


                    #Facet Traffic
                    print "Compressive Bernoulli: Phase 1, Facet - " + feature_set + "/CompressiveBernoulli_facetTraffic_phase1_" + str(compressive_ratio) + "_dataset.csv"
                    output = open(data_folder + "CompressiveBernoulli_facetTraffic_phase1_" + str(compressive_ratio) + "_dataset.csv", "w") 
                    f = open(data_folder + "FacetTraffic_50_phase1_dataset.csv", 'r')
                    reader = csv.reader(f, delimiter=',')

                    #Process data row
                    for n, row in enumerate(reader):
                        if(n == 0):
                            output.write(",".join(str(x) for x in range(0,M)) + "," + row[-1] + "\n")
                        else:
                            #Gather the first n packets array
                            first_n_packets_vector = []
                            for i in row[:-1]:
                                first_n_packets_vector.append(int(i))

                            compressed_vector = np.matmul(sensing_matrix, first_n_packets_vector)

                            #print "Compressed vector: " + str(compressed_vector)
                            output.write(",".join(str(x) for x in compressed_vector) + "," + row[-1] + "\n")
                    output.close()

                    ########################################################################################
                    ########################################################################################
                    ########################################################################################


                    """
                    Process Phase 2 Data
                    """

                    #Regular Traffic
                    print "Compressive Bernoulli: Phase 2, Regular - " + feature_set + "/CompressiveBernoulli_regularTraffic_phase2_" + str(compressive_ratio) + "_dataset.csv"
                    output = open(data_folder + "CompressiveBernoulli_regularTraffic_phase2_" + str(compressive_ratio) + "_dataset.csv", "w") 
                    f = open(data_folder + "RegularTraffic_phase2_dataset.csv", 'r')
                    reader = csv.reader(f, delimiter=',')

                    #Process data row
                    for n, row in enumerate(reader):
                        if(n == 0):
                            output.write(",".join(str(x) for x in range(0,M)) + "," + row[-1] + "\n")
                        else:
                            #Gather the first n packets array
                            first_n_packets_vector = []
                            for i in row[:-1]:
                                first_n_packets_vector.append(int(i))

                            compressed_vector = np.matmul(sensing_matrix, first_n_packets_vector)

                            #print "Compressed vector: " + str(compressed_vector)
                            output.write(",".join(str(x) for x in compressed_vector) + "," + row[-1] + "\n")
                    output.close()


                    #Facet Traffic
                    print "Compressive Bernoulli Phase 2, Facet - " + feature_set + "/CompressiveBernoulli_facetTraffic_phase2_" + str(compressive_ratio) + "_dataset.csv"
                    output = open(data_folder + "CompressiveBernoulli_facetTraffic_phase2_" + str(compressive_ratio) + "_dataset.csv", "w") 
                    f = open(data_folder + "FacetTraffic_50_phase2_dataset.csv", 'r')
                    reader = csv.reader(f, delimiter=',')

                    #Process data row
                    for n, row in enumerate(reader):
                        if(n == 0):
                            output.write(",".join(str(x) for x in range(0,M)) + "," + row[-1] + "\n")
                        else:
                            #Gather the first n packets array
                            first_n_packets_vector = []
                            for i in row[:-1]:
                                first_n_packets_vector.append(int(i))

                            compressed_vector = np.matmul(sensing_matrix, first_n_packets_vector)

                            #print "Compressed vector: " + str(compressed_vector)
                            output.write(",".join(str(x) for x in compressed_vector) + "," + row[-1] + "\n")
                    output.close()