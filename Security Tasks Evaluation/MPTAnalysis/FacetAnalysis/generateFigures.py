import os
import numpy as np
import math

import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


colors = ["0.8", "0.6", "0.2", "0.0"]
colors = ["salmon", "lightsteelblue", "darkseagreen", "thistle", "wheat", "khaki", "skyblue"]

"""
Attach a text label above each bar displaying its height
"""
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.005*height,   # original height was 1.005*height
                "{0:.2f}".format(float(height)), fontsize=7, ha='center', va='bottom')


def PlotQuantization(binWidths, n_flows):
    print "PlotQuantization"
    feature_sets = []
    set_acc = []
    set_fpr =[]
    set_fnr = []

    for binWidth in binWidths:
          
        feature_folder = 'PL_60_' + str(binWidth) + '_1500'
        #print feature_folder

        #Load configuration results
        data_folder = 'classificationResults/' + feature_folder + '/' + "classificationResults_phase2_NoSketch.npy"
        results = np.load(data_folder)
        set_acc.append(results[0])
        set_fpr.append(results[1])
        set_fnr.append(results[2])
        feature_sets.append(feature_folder)


    max_acc = 0
    max_fset = ""
    for i, f_set in enumerate(feature_sets):
        if set_acc[i] > max_acc:
            max_acc = set_acc[i]
            max_fset = f_set
    print "Max acc: %s, Best quantization set: %s"%(max_acc, max_fset)

    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(111)
    
    curr_fset = feature_sets
    curr_acc = set_acc
    curr_fpr = set_fpr
    curr_fnr = set_fnr
    #print "Current feature set: "+ str(curr_fset)
    
    ind = np.arange(len(curr_fset))  # the x locations for the groups
    width = 0.20

    rects0 = ax1.bar(ind - width - width/2, curr_acc, width, color=colors[0], label='Acc')
    autolabel(rects0,ax1)
    rects1 = ax1.bar(ind - width/2 , curr_fpr, width, color=colors[1], label='FPR')
    autolabel(rects1,ax1)
    rects2 = ax1.bar(ind + width - width/2, curr_fnr, width, color=colors[2], label='FNR')
    autolabel(rects2,ax1)


    ax1.yaxis.grid(color='black', linestyle='dotted')
    ax1.set_title('Scores for Quantization')
    ax1.set_xticks(ind)
    labels = ["K = " + str(int(x.split('_')[2])) + " -> " + str(1500/int(x.split('_')[2])) + " bins" + "\n(PerFlow = " + str(int(1500/int(x.split('_')[2]))*4) + " B)" + "\n(CGMem = " + str((n_flows * int(1500/int(x.split('_')[2]))*4)/1024) + " KB)" for x in feature_sets]
    ax1.set_xticklabels(labels)
    plt.xticks(fontsize=7)
    ax1.legend()

    plt.ylim(top=1)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    fig.savefig('Figures/Facet_bin_NoSketch.pdf')   # save the figure to file
    fig.savefig('Figures/Facet_bin_NoSketch.png')   # save the figure to file
    plt.close(fig)


def PlotQuantizationLines(binWidths, n_flows):
    print "PlotQuantizationLines"
    feature_sets = []
    set_acc = []

    for binWidth in binWidths:
          
        feature_folder = 'PL_60_' + str(binWidth) + '_1500'
        #print feature_folder

        #Load configuration results
        data_folder = 'classificationResults/' + feature_folder + '/' + "classificationResults_phase2_NoSketch.npy"
        results = np.load(data_folder)
        set_acc.append(results[3])
        feature_sets.append(feature_folder)



    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(111)
    
    curr_fset = feature_sets
    curr_acc = set_acc
    
    ind = np.arange(len(curr_fset))  # the x locations for the groups
    print curr_acc
    ax1.plot(curr_acc, color=colors[0], marker=".", markersize=12, lw=3, label='AUC')
    ax1.hlines(0.99, 0, len(ind)-1, lw=2, label='Baseline, AUC = 0.99')

    for i,j in zip(ind,curr_acc):
        ax1.annotate("{0:.2f}".format(j),xy=(i-0.15,j-0.08))


    ax1.yaxis.grid(color='black', linestyle='dotted')
    plt.yticks(fontsize=14)
    plt.ylim(bottom=0,top=1)
    plt.ylabel("AUC Score", fontsize=14)


    plt.xlim(-0.3, len(ind)-1+0.3)
    ax1.set_xticks(ind)
    labels = [str(int(x.split('_')[2])) for x in feature_sets]
    #labels = ["K = " + str(int(x.split('_')[2])) + " -> " + str(1500/int(x.split('_')[2])) + " bins" + "\n(PerFlow = " + str(int(1500/int(x.split('_')[2]))*4) + " B)" + "\n(CGMem = " + str((n_flows * int(1500/int(x.split('_')[2]))*4)/1024) + " KB)" for x in feature_sets]
    #labels = ["K = " + str(int(x.split('_')[2])) + "\nPF = " + str(int(1500/int(x.split('_')[2]))*4) + " B" + "\nTM = " + str((n_flows * int(1500/int(x.split('_')[2]))*4)/1024) + " KB" for x in feature_sets]
    ax1.set_xticklabels(labels)
    plt.xticks(fontsize=11)
    plt.xlabel("Quantization Factor", fontsize=14)
    ax1.legend()

    
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    fig.savefig('Figures/Facet_bin_NoSketch_Lines.pdf')   # save the figure to file
    fig.savefig('Figures/Facet_bin_NoSketch_Lines.png')   # save the figure to file
    plt.close(fig)


def PlotKQuantizationAndTruncation(binWidths, topk_features, n_flows):
    print "PlotKQuantizationAndTruncation"
    if not os.path.exists('Figures/Truncation_comparison'):
        os.makedirs('Figures/Truncation_comparison')

    for binWidth in binWidths:
        feature_sets = []
        set_acc = []
        set_fpr =[]
        set_fnr = []

        for topk in topk_features:
              
            feature_folder = 'PL_60_' + str(binWidth) + '_' + str(topk)
            #print feature_folder

            if(topk != 1500 and topk > 1500/binWidth):
                #print "Skipping sample, invalid configuration. TopK = " + str(topk) + " Total Features = " + str(1500/binWidth)
                set_acc.append(0)
                set_fpr.append(0)
                set_fnr.append(0)
                feature_sets.append(feature_folder)
                continue

            #Load configuration results
            #if(topk == 1500):
            #    data_folder = 'classificationResults/' + feature_folder + '/' + "classificationResults_phase1_NoSketch.npy"
            #else:
            data_folder = 'classificationResults/' + feature_folder + '/' + "classificationResults_phase2_NoSketch.npy"
            results = np.load(data_folder)
            set_acc.append(results[0])
            set_fpr.append(results[1])
            set_fnr.append(results[2])
            feature_sets.append(feature_folder)


        #Check best truncation value
        max_acc = 0
        max_fset = ""
        for i, f_set in enumerate(feature_sets[:-1]):
            if set_acc[i] > max_acc:
                max_acc = set_acc[i]
                max_fset = f_set
        print "K = " + str(binWidth) + ", Max acc: %s, Best Truncation: %s"%(max_acc, max_fset)


        #Plot figures
        fig = plt.figure(figsize=(10,4))
        ax1 = fig.add_subplot(111)
        
        curr_fset = feature_sets
        curr_acc = set_acc
        curr_fpr = set_fpr
        curr_fnr = set_fnr
        #print "Current feature set: "+ str(curr_fset)
        
        ind = np.arange(len(curr_fset))  # the x locations for the groups
        width = 0.20

        rects0 = ax1.bar(ind - width - width/2, curr_acc, width, color=colors[0], label='Acc')
        autolabel(rects0,ax1)
        rects1 = ax1.bar(ind - width/2 , curr_fpr, width, color=colors[1], label='FPR')
        autolabel(rects1,ax1)
        rects2 = ax1.bar(ind + width - width/2, curr_fnr, width, color=colors[2], label='FNR')
        autolabel(rects2,ax1)

        ax1.yaxis.grid(color='black', linestyle='dotted')
        ax1.set_title('Truncation Scores for K ='+str(binWidth))
        ax1.set_xticks(ind)
        print feature_sets
        labels = ["Top-k= " + str(int(x.split('_')[3])) + "\n(PerFlow = " + str(int(x.split('_')[3])*4) + " B)" + "\n(CGMem = " + str((n_flows * int(x.split('_')[3]) * 4)/1024) + " KB)" for x in feature_sets]
        labels[len(topk_features)-1] = str(int(1500/binWidth)) + " features\n(PerFlow = " + str(int(1500/binWidth)*4) + " B)" + "\n(CGMem = " + str(int((n_flows * int(1500/binWidth) * 4)/1024)) + " KB)"
        ax1.set_xticklabels(labels)
        plt.xticks(fontsize=9)
        ax1.legend()

        plt.ylim(top=1)
        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        fig.savefig('Figures/Truncation_comparison/Facet_bin' + str(binWidth) + '_topk_NoSketch.pdf')   # save the figure to file
        fig.savefig('Figures/Truncation_comparison/Facet_bin' + str(binWidth) + '_topk_NoSketch.png')   # save the figure to file
        plt.close(fig)

def PlotKQuantizationAndTruncationLines(binWidths, topk_features, n_flows):
    print "PlotKQuantizationAndTruncation"
    if not os.path.exists('Figures/Truncation_comparison'):
        os.makedirs('Figures/Truncation_comparison')

    for binWidth in binWidths:
        feature_sets = []
        set_acc = []

        for topk in topk_features:
              
            feature_folder = 'PL_60_' + str(binWidth) + '_' + str(topk)
            #print feature_folder

            if(topk != 1500 and topk > 1500/binWidth):
                #print "Skipping sample, invalid configuration. TopK = " + str(topk) + " Total Features = " + str(1500/binWidth)
                set_acc.append(0)
                feature_sets.append(feature_folder)
                continue

            #Load configuration results
            #if(topk == 1500):
            #    data_folder = 'classificationResults/' + feature_folder + '/' + "classificationResults_phase1_NoSketch.npy"
            #else:
            data_folder = 'classificationResults/' + feature_folder + '/' + "classificationResults_phase2_NoSketch.npy"
            results = np.load(data_folder)
            set_acc.append(results[3])
            feature_sets.append(feature_folder)


        #Plot figures
        fig = plt.figure(figsize=(10,4))
        ax1 = fig.add_subplot(111)
        
        curr_fset = feature_sets
        curr_acc = set_acc

        #print "Current feature set: "+ str(curr_fset)
        
        ind = np.arange(len(curr_fset))  # the x locations for the groups


        ax1.plot(curr_acc, color=colors[0], marker=".", markersize=12, lw=3, label='AUC')
        ax1.hlines(0.99, 0, len(ind)-1, lw=3, label='Baseline, AUC = 0.99')
        
        for i,j in zip(ind,curr_acc):
            ax1.annotate("{0:.2f}".format(j),xy=(i-0.1,j-0.08))

        plt.xlim(-0.3, len(ind)-1+0.3)
        ax1.yaxis.grid(color='black', linestyle='dotted')

        ax1.set_xticks(ind)
        print feature_sets
        labels = [str(int(x.split('_')[3])) for x in feature_sets]
        #labels = ["Top-n= " + str(int(x.split('_')[3])) + "\nPF = " + str(int(x.split('_')[3])*4) + " B" + "\nTM = " + str((n_flows * int(x.split('_')[3]) * 4)/1024) + " KB" for x in feature_sets]
        #labels[len(topk_features)-1] = str(int(1500/binWidth)) + " features\n(PF = " + str(int(1500/binWidth)*4) + " B)" + "\n(TMem = " + str(int((n_flows * int(1500/binWidth) * 4)/1024)) + " KB)"
        ax1.set_xticklabels(labels)
        plt.xticks(fontsize=9)
        plt.xlabel("Truncation Factor", fontsize=12)
        ax1.legend()
        
        
        plt.yticks(fontsize=12)
        plt.ylim(bottom=0,top=1)
        plt.ylabel("AUC Score", fontsize=12)

        plt.legend(loc='lower right', fontsize=12)
        plt.tight_layout()
        fig.savefig('Figures/Truncation_comparison/Facet_bin' + str(binWidth) + '_topk_NoSketch_Lines.pdf')   # save the figure to file
        fig.savefig('Figures/Truncation_comparison/Facet_bin' + str(binWidth) + '_topk_NoSketch_Lines.png')   # save the figure to file
        plt.close(fig)
    

def GenerateFigures(binWidths, topk_features, nFlows):
    if not os.path.exists('Figures'):
        os.makedirs('Figures')

    PlotQuantization(binWidths, nFlows)
    PlotKQuantizationAndTruncation(binWidths, topk_features, nFlows)
    


def GenerateFiguresLine(binWidths, topk_features, nFlows):
    if not os.path.exists('Figures'):
        os.makedirs('Figures')

    TOPK = [10, 20, 30, 40, 50]
    PlotQuantizationLines(binWidths, nFlows)
    PlotKQuantizationAndTruncationLines(binWidths, TOPK, nFlows)
    


if __name__ == "__main__":

    #Quantization
    BIN_WIDTH = [1, 4, 8, 16, 32, 64, 128, 256]

    #Truncation Top-K features
    TOPK = [5, 10, 20, 30, 40, 50, 1500]
    TOPK = [10, 20, 30, 40, 50]

    #Total amount of flows per dataset
    N_FLOWS = 1000

    PlotQuantizationLines(BIN_WIDTH, N_FLOWS)
    PlotKQuantizationAndTruncationLines(BIN_WIDTH, TOPK, N_FLOWS)