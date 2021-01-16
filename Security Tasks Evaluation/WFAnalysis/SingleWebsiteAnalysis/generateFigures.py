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


def PlotSingleWebsiteStats():

    for profile in os.listdir("classificationResults/"):
        if(".DS_Store" in profile):
            continue

        profile_data = open("classificationResults/" + profile, 'rb')
        csv_reader = csv.reader(profile_data, delimiter=',')

        binWidth = []
        acc = []
        fpr = []
        fnr = []

        for n, row in enumerate(csv_reader):
            if(n == 0):
                continue
            binWidth.append(row[0])
            acc.append(float(row[1]))
            fpr.append(float(row[2]))
            fnr.append(float(row[3]))


        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        print "Current feature set: "+ str(binWidth)
        
        
        ind = np.arange(len(binWidth))  # the x locations for the groups
        width = 0.20

        rects0 = ax1.bar(ind - width, acc, width, colors[0], label='Acc')
        rects1 = ax1.bar(ind, fpr, width, colors[1], label='FPR')
        rects2 = ax1.bar(ind + width, fnr, width, colors[2], label='FNR')


        ax1.yaxis.grid(color='black', linestyle='dotted')
        ax1.set_title('Scores for Quantization')
        ax1.set_yscale("log")
        ax1.set_xticks(ind)
        labels = binWidth
        ax1.set_xticklabels(labels)
        ax1.legend()


        plt.tight_layout()
        #plt.ylim(0, 1)

        fig.savefig('WF_%s.pdf'%(profile[:-4]))   # save the figure to file
        fig.savefig('WF_%s.png'%(profile[:-4]))   # save the figure to file
        plt.close(fig)
        profile_data.close()

                    
def PlotNormalFPRComparison():
    websites = set()

    #Compute the set of websites to compare
    for profile in os.listdir("classificationResults/"):
        if(".DS_Store" in profile):
            continue
        website = profile.split("_")[2]
        website = website[:-4]
        websites.add(website)


    for website in websites:
        if not os.path.exists("Figures/%s"%(website)):
            os.makedirs("Figures/%s"%(website))

        #Gather results for full distribution
        profile_data_full = open("classificationResults/SingleWebsite_full_" + website + ".csv", 'rb')
        csv_reader_full = csv.reader(profile_data_full, delimiter=',')

        binWidth_full = []
        acc_full = []
        fpr_full = []
        fnr_full = []

        for n, row in enumerate(csv_reader_full):
            if(n == 0):
                continue
            binWidth_full.append(row[0])
            acc_full.append(round(Decimal(float(row[1])), 4))
            fpr_full.append(round(Decimal(float(row[2])), 9))
            fnr_full.append(round(Decimal(float(row[3])), 4))


        #Gather results for truncated distribution
        profile_data_truncated = open("classificationResults/SingleWebsite_truncated_" + website + ".csv", 'rb')
        csv_reader_truncated = csv.reader(profile_data_truncated, delimiter=',')

        binWidth_truncated = []
        acc_truncated = []
        fpr_truncated = []
        fnr_truncated = []

        for n, row in enumerate(csv_reader_truncated):
            if(n == 0):
                continue
            binWidth_truncated.append(row[0])
            acc_truncated.append(round(Decimal(float(row[1])), 4))
            fpr_truncated.append(round(Decimal(float(row[2])), 9))
            fnr_truncated.append(round(Decimal(float(row[3])), 4))

        #Gather number of bins used in the truncation
        truncated_info_file = open("truncationInfo/" + website + ".csv", 'r')
        truncation_info = csv.reader(truncated_info_file, delimiter=',')
        truncated_bins = []

        for n, row in enumerate(truncation_info):
            if(n == 0):
                continue
            truncated_bins.append(row[1])

        #Generate plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        print "Current feature set: "+ str(binWidth_full)
        print "FPR-Full: " + str(fpr_full)
        print "FPR-Truncated: " + str(fpr_truncated)
        
        ind = np.arange(len(binWidth_full))  # the x locations for the groups
        width = 0.40

        rects1 = ax1.bar(ind - width, fpr_full, width, color=colors[0], label='FPR-Full')
        #autolabel(rects1,ax1)
        rects2 = ax1.bar(ind, fpr_truncated, width, color=colors[1], label='FPR-Truncated')
        #autolabel(rects2,ax1)


        ax1.yaxis.grid(color='black', linestyle='dotted')
        ax1.set_title('Truncation effect on FPR - %s'%(website), fontsize = 10)
        
        ax1.set_xticks(ind)
        labels = ["K = " + x + "\nBins = " + str(truncated_bins[n]) for n, x in enumerate(binWidth_full)]
        ax1.set_xticklabels(labels)
        ax1.legend()

        plt.xticks(fontsize=7)
        plt.tight_layout()
        #plt.ylim(0, 1)
        fig.savefig('Figures/%s/WF_FPR_normal_%s.pdf'%(website, website))   # save the figure to file
        fig.savefig('Figures/%s/WF_FPR_normal_%s.png'%(website, website))   # save the figure to file
        plt.close(fig)
        profile_data_full.close()
        profile_data_truncated.close()


def PlotNormalFNRComparison():
    websites = set()

    #Compute the set of websites to compare
    for profile in os.listdir("classificationResults/"):
        if(".DS_Store" in profile):
            continue
        website = profile.split("_")[2]
        website = website[:-4]
        websites.add(website)


    for website in websites:
        if not os.path.exists("Figures/%s"%(website)):
            os.makedirs("Figures/%s"%(website))

        #Gather results for full distribution
        profile_data_full = open("classificationResults/SingleWebsite_full_" + website + ".csv", 'rb')
        csv_reader_full = csv.reader(profile_data_full, delimiter=',')

        binWidth_full = []
        acc_full = []
        fpr_full = []
        fnr_full = []

        for n, row in enumerate(csv_reader_full):
            if(n == 0):
                continue
            binWidth_full.append(row[0])
            acc_full.append(round(Decimal(float(row[1])), 4))
            fpr_full.append(round(Decimal(float(row[2])), 4))
            fnr_full.append(round(Decimal(float(row[3])), 4))


        #Gather results for truncated distribution
        profile_data_truncated = open("classificationResults/SingleWebsite_truncated_" + website + ".csv", 'rb')
        csv_reader_truncated = csv.reader(profile_data_truncated, delimiter=',')

        binWidth_truncated = []
        acc_truncated = []
        fpr_truncated = []
        fnr_truncated = []

        for n, row in enumerate(csv_reader_truncated):
            if(n == 0):
                continue
            binWidth_truncated.append(row[0])
            acc_truncated.append(round(Decimal(float(row[1])), 4))
            fpr_truncated.append(round(Decimal(float(row[2])), 4))
            fnr_truncated.append(round(Decimal(float(row[3])), 4))


        #Gather number of bins used in the truncation
        truncated_info_file = open("truncationInfo/" + website + ".csv", 'r')
        truncation_info = csv.reader(truncated_info_file, delimiter=',')
        truncated_bins = []

        for n, row in enumerate(truncation_info):
            if(n == 0):
                continue
            truncated_bins.append(row[1])


        #Generate plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        print "Current feature set: "+ str(binWidth_full)
        print "FNR-Full: " + str(fnr_full)
        print "FNR-Truncated: " + str(fnr_truncated)
        
        ind = np.arange(len(binWidth_full))  # the x locations for the groups
        width = 0.40

        rects1 = ax1.bar(ind - width, fnr_full, width, color=colors[0], label='FNR-Full')
        autolabel(rects1,ax1)
        rects2 = ax1.bar(ind, fnr_truncated, width, color=colors[1], label='FNR-Truncated')
        autolabel(rects2,ax1)


        ax1.yaxis.grid(color='black', linestyle='dotted')
        ax1.set_title('Truncation effect on FNR - %s'%(website), fontsize = 10)
        
        ax1.set_xticks(ind)
        labels = ["K = " + x + "\nBins = " + str(truncated_bins[n]) for n, x in enumerate(binWidth_full)]
        ax1.set_xticklabels(labels)
        ax1.legend()

        plt.xticks(fontsize=7)
        plt.tight_layout()
        plt.ylim(0, 1)
        fig.savefig('Figures/%s/WF_FNR_normal_%s.pdf'%(website, website))   # save the figure to file
        fig.savefig('Figures/%s/WF_FNR_normal_%s.png'%(website, website))   # save the figure to file
        plt.close(fig)
        profile_data_full.close()
        profile_data_truncated.close()



def GenerateFigures():
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    PlotNormalFNRComparison()
    PlotNormalFPRComparison()