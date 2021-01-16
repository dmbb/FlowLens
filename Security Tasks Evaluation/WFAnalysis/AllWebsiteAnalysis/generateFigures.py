import os
from decimal import Decimal
import numpy as np
import csv

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


def PlotNormalAccuracy():
    print "Plotting accuracy for no-sketch"
    #Gather results for full distribution
    profile_data_full = open("classificationResults/AllVsAll.csv", 'rb')
    csv_reader_full = csv.reader(profile_data_full, delimiter=',')

    binWidth_full = []
    acc_full = []

    for n, row in enumerate(csv_reader_full):
        if(n == 0):
            continue
        binWidth_full.append(row[0])
        acc_full.append(round(Decimal(float(row[1])), 4))


    #Generate plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    print "Current feature set: "+ str(binWidth_full)
    print "ACC-Full: " + str(acc_full)
    
    ind = np.arange(len(binWidth_full))  # the x locations for the groups
    width = 0.4

    rects1 = ax1.bar(ind - width/2, acc_full, width, color=colors[0], label='Accuracy')
    autolabel(rects1,ax1)

    ax1.yaxis.grid(color='black', linestyle='dotted')
    ax1.set_title('Quantization effect on accuracy - WF Multiclass', fontsize = 10)
    
    ax1.set_xticks(ind)
    labels = ["K = " + x + "\nBins = " + str(3000/int(x)) for n, x in enumerate(binWidth_full)]
    ax1.set_xticklabels(labels)
    ax1.legend()

    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Quantization')

    plt.xticks(fontsize=7)
    plt.tight_layout()
    plt.ylim(0, 1)
    fig.savefig('Figures/AllVsAll.pdf')   # save the figure to file
    fig.savefig('Figures/AllVsAll.png')   # save the figure to file
    plt.close(fig)
    profile_data_full.close()



def GenerateFigures():
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    PlotNormalAccuracy()