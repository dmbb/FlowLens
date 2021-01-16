import os
import math
import subprocess as sub

from generateFeatures import CompressFeatures, SplitDataset, CompressFeaturesBasedOnTrainData, MergeTestData, ExtractPacketSample
from classifier import GenerateFeatureImportanceBasedOnTrainData, ClassifyTestDataBasedOnModel, BuildModelBasedOnTrainData
from generateFigures import GenerateFigures, GenerateFiguresLines
from online_sketching import CreateBinaryVectorRepresentation
from compressive_ta import CreateCompressiveRepresentation

def Experiment(BIN_WIDTH, TOPK, DELTAS, MEMORY_FACTORS, DATASET_SPLIT, COVERT_FLOWS_PERC, N_FLOWS, ONLINE_SKETCH_SIZE, SIGMA_PARAM, NUMBER_OF_PACKETS, COMPRESSIVE_RATIO):
    """
    Phase 1a)
    Use full information and generate the best buckets.
        Datasets are split into half. 
        
        We use the first half to train/test a classifier with a balanced dataset in HoldOut 90/10
    """
    CompressFeatures(BIN_WIDTH, [TOPK[-1]])
    SplitDataset(DATASET_SPLIT, N_FLOWS, 1)
    GenerateFeatureImportanceBasedOnTrainData("normal", BIN_WIDTH, [TOPK[-1]])

    """
    Phase 1b)
    Quantize, truncate and classify according to the best buckets found
        The first half of each dataset is again used for train/test the classifier with a balanced dataset in HoldOut 90/10
        However, only the top-K bins are used for performing classification 
        
        The built model is saved to use in Phase 2.
    """
    CompressFeaturesBasedOnTrainData(BIN_WIDTH, TOPK[:-1])
    SplitDataset(DATASET_SPLIT, N_FLOWS, COVERT_FLOWS_PERC)
    BuildModelBasedOnTrainData("normal", BIN_WIDTH, TOPK)

    """
    Phase 2
    Classify new flows using quantized/truncated distributions using the previously built model
        The second half of each dataset is used for train/test the classifier with an unbalanced dataset
    """
    #Quantization + Truncation
    ClassifyTestDataBasedOnModel("normal", BIN_WIDTH, TOPK, N_FLOWS, ONLINE_SKETCH_SIZE)

    #Generate figures
    GenerateFiguresLines(BIN_WIDTH, TOPK, DELTAS, MEMORY_FACTORS, N_FLOWS)


    """
    Online Sketching - Coskun et al.
    """
    CreateBinaryVectorRepresentation(BIN_WIDTH, [TOPK[-1]], ONLINE_SKETCH_SIZE)
    BuildModelBasedOnTrainData("online", BIN_WIDTH, [TOPK[-1]], ONLINE_SKETCH_SIZE)
    ClassifyTestDataBasedOnModel("online", BIN_WIDTH, [TOPK[-1]], N_FLOWS, ONLINE_SKETCH_SIZE)

    
    """
    Compressive TA adjusted to packet distribution
    """
    CreateCompressiveRepresentation("compressive_gaussian", BIN_WIDTH, [TOPK[-1]], SIGMA_PARAM, COMPRESSIVE_RATIO)
    BuildModelBasedOnTrainData("compressive_gaussian", BIN_WIDTH, [TOPK[-1]], ONLINE_SKETCH_SIZE, SIGMA_PARAM, NUMBER_OF_PACKETS, COMPRESSIVE_RATIO)
    ClassifyTestDataBasedOnModel("compressive_gaussian", BIN_WIDTH, [TOPK[-1]], N_FLOWS, ONLINE_SKETCH_SIZE, DELTAS, MEMORY_FACTORS, SIGMA_PARAM, NUMBER_OF_PACKETS, COMPRESSIVE_RATIO)

    CreateCompressiveRepresentation("compressive_bernoulli", BIN_WIDTH, [TOPK[-1]], SIGMA_PARAM, COMPRESSIVE_RATIO)
    BuildModelBasedOnTrainData("compressive_bernoulli", BIN_WIDTH, [TOPK[-1]], ONLINE_SKETCH_SIZE, SIGMA_PARAM, NUMBER_OF_PACKETS, COMPRESSIVE_RATIO)
    ClassifyTestDataBasedOnModel("compressive_bernoulli", BIN_WIDTH, [TOPK[-1]], N_FLOWS, ONLINE_SKETCH_SIZE, DELTAS, MEMORY_FACTORS, SIGMA_PARAM, NUMBER_OF_PACKETS, COMPRESSIVE_RATIO)


if __name__ == "__main__":

    #Quantization
    BIN_WIDTH = [1, 4, 8, 16, 32, 64, 128, 256]

    #Truncation Top-K features
    TOPK = [5, 10, 20, 30, 40, 50, 1500]
    
    #Online Sketch
    ONLINE_SKETCH_SIZE = [64, 128, 256, 512, 1024, 2048]

    #Proportion of regular flows to input in sketch
    COVERT_FLOWS_PERC = 1
    
    #Proportion to split training phase (1) and testing phase (2)
    DATASET_SPLIT = 0.5

    #Total amount of flows per dataset
    N_FLOWS = 300

    #Standard deviation of Gaussian distribution (compressive TA)
    SIGMA_PARAM = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

    #Number of packets to compute compressive TA representation
    NUMBER_OF_PACKETS = [1000, 2000, 4000]

    #Compression Ratio for Compressive TA
    COMPRESSIVE_RATIO = [4, 8, 16, 32, 64, 128, 256]

    #Deprecated
    DELTAS = [0.95]
    MEMORY_FACTORS = [8, 4, 2, 1]

    #Run Experiment:
    Experiment(BIN_WIDTH, TOPK, DELTAS, MEMORY_FACTORS, DATASET_SPLIT, COVERT_FLOWS_PERC, N_FLOWS, ONLINE_SKETCH_SIZE, SIGMA_PARAM, NUMBER_OF_PACKETS, COMPRESSIVE_RATIO)






