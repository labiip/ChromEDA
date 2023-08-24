# coding=gbk
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import numpy as np
import itertools
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score 
from matplotlib import rcParams 

import matplotlib.colors
import random
from colorsys import hls_to_rgb
import csv
#config = {
#    "font.family":'Times New Roman',  
#    "font.size": 10,
##     "mathtext.fontset":'stix',
#}
config = {
           # "font.family": 'serif',
           "font.family":'Times New Roman',
           "font.size": 8,
           "mathtext.fontset": 'stix',
           "font.serif": ['SimSun'],
        }

rcParams.update(config)

clr_list = []
def generate_colormap():

    x= np.linspace(1, 12, 24, endpoint=True)
    y=x/x
   
    hue = 0
    num_colors = 24
    for i in range(0, num_colors):
        hue += int(360/num_colors)
        saturation = 20 + num_colors%8
        lightness = 40 + num_colors%8
        if (i % 2) == 0:
            saturation += 10
            lightness += 10
               
        clr_rgb = hls_to_rgb(hue / 360, lightness / 100, saturation / 100)
        clr_list.append(clr_rgb)
generate_colormap()

def dict2csv(dic, filename):
    """

    :param dic: the dict to csv
    :param filename: the name of the csv file
    :return: None
    """
    with open(filename,'w', encoding='utf-8') as f:
        b = []
        for i in range(0,24):
            b = dic[i]
            # b.append(dic[i])
            f.write(str(b)+'\n')
        f.close()
    # file = open(filename, 'w', encoding='utf-8', newline='')
    # csv_writer = csv.writer(file)
    # # csv_writer.writeheader()
    # # for i in range(len(dic[list(dic.keys())[0]])-2):   # 将字典逐行写入csv
    # for i in range(0,24):
    #     b= dic[i]
    #     # dic1 = {key: dic[key][i] for key in dic.keys()}
    #     csv_writer.writerow(b)
    # file.close()


def roc_plot(y_test=None,y_score=None):
   
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Binarize the output 
    n_classes = 24
    y_test = label_binarize(y_test, classes=list(range(n_classes)))
    # one vs rest
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


   # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    micro_auc = roc_auc_score(y_test, y_score, average='micro')

    lw = 2

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    #  Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
       mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

   # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    macro_auc = roc_auc_score(y_test, y_score, average='macro')
    return fpr,tpr,roc_auc
def plot_roc(data,file_name):
   # Plot all ROC curves
   #fig = plt.figure()
    n_classes = 24

    title = ['Resnet50','+Adver.','+Adver.+ Pse.','+Adver.+ Ang.','MCM']

    fontsize=32
    fig = plt.figure(figsize=(16,25))
    for m in range(5):
       for n in range(2):
            fpr ,tpr ,roc_auc = data[m+n*5]
            # if m+n*5==9:
                # dict2csv(roc_auc,'/home_lv/xinyu.fan/myUnsupervised/unsuper/resnet21da.txt')
            mean_auc = 0.0
            for i in range(24):
                mean_auc+=roc_auc[i]
            print(title[m],' mean AUC: ',mean_auc/24)
            plt.subplot(5,2,m*2+n+1)
            for i in range(n_classes):
                pt_, = plt.plot(fpr[i], tpr[i], lw=2, color=clr_list[i])
                plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0], fontsize=24)
                plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0], fontsize=24)
                # plt.xlabel('TP',fontsize=fontsize)
                # plt.ylabel('FP',fontsize=fontsize)
                plt.xlabel('FPR',fontsize=fontsize)
                plt.ylabel('TPR',fontsize=fontsize)
                if m==0 and n==0:
                    # plt.title('GI->GII',fontsize=fontsize)
                    plt.title('GI->GII',fontsize=fontsize)

                if m==0 and n==1:
                    # plt.title('GII->GI',fontsize=fontsize)
                    plt.title('GI->GII',fontsize=fontsize)
                
                # plt.legend(loc="lower right")
               
   
      
    # plt.text(-2.0,5.77,title[0],rotation=90,fontsize=fontsize)
    # plt.text(-2.0,4.36,title[1],rotation=90,fontsize=fontsize)
    # plt.text(-2.0,2.80,title[2],rotation=90,fontsize=fontsize)
    # plt.text(-2.0,1.38,title[3],rotation=90,fontsize=fontsize)
    # plt.text(-2.0,0.42,title[4],rotation=90,fontsize=fontsize)
    # for i in range(24):
    #     plt.text(1.15,6.49-i*0.2877,'chr - {0}'.format(i+1),color=clr_list[i],fontsize=30)
    # plt.subplots_adjust(left=0.17, bottom=0.04, right=0.85, top=0.97,wspace=0.38, hspace=0.25)
    # plt.savefig('fxy4_roc.tiff', dpi=300)
    # plt.close(fig)
    


G2X_re_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/resnet/G_2_X_true_label.npy")
G2X_re_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/resnet/G_2_X_probs_label.npy")

G2X00_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_0_rot_0_eta_0.3_aug_1true_label.npy")
G2X00_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_0_rot_0_eta_0.3_aug_1probs_label.npy")

G2X01_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_0_rot_1_eta_0.3_aug_1true_label.npy")
G2X01_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_0_rot_1_eta_0.3_aug_1probs_label.npy")

G2X10_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_1_rot_0_eta_0.3_aug_1true_label.npy")
G2X10_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_1_rot_0_eta_0.3_aug_1probs_label.npy")

G2X11_cap_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_1_rot_1_eta_0.3_aug_1true_label.npy")
G2X11_cap_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_1_rot_1_eta_0.3_aug_1probs_label.npy")

X2G_re_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/resnet/X_2_G_true_label.npy")
X2G_re_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/resnet/X_2_G_probs_label.npy")

X2G00_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_0_rot_0_eta_0.3_aug_1true_label.npy")
X2G00_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_0_rot_0_eta_0.3_aug_1probs_label.npy")

X2G01_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_0_rot_1_eta_0.3_aug_1true_label.npy")
X2G01_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_0_rot_1_eta_0.3_aug_1probs_label.npy")

X2G10_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_1_rot_0_eta_0.3_aug_1true_label.npy")
X2G10_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_1_rot_0_eta_0.3_aug_1probs_label.npy")

X2G11_cap_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_1_rot_1_eta_0.3_aug_1true_label.npy")
X2G11_cap_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_1_rot_1_eta_0.3_aug_1probs_label.npy")

data = [[],[],[],[],[],[],[],[],[],[]]

data[5] = roc_plot(y_test=G2X_re_true_label,y_score=G2X_re_probs_label)
data[0] = roc_plot(y_test=X2G_re_true_label,y_score=X2G_re_probs_label)
data[7] = roc_plot(y_test=G2X00_true_label,y_score=G2X00_probs_label)
data[2] = roc_plot(y_test=X2G00_true_label,y_score=X2G00_probs_label)#+adver
data[8] = roc_plot(y_test=G2X01_true_label,y_score=G2X01_probs_label)
data[3] = roc_plot(y_test=X2G01_true_label,y_score=X2G01_probs_label)#+ang
data[6] = roc_plot(y_test=G2X10_true_label,y_score=G2X10_probs_label)
data[1] = roc_plot(y_test=X2G10_true_label,y_score=X2G10_probs_label)#+pse
data[9] = roc_plot(y_test=G2X11_cap_true_label,y_score=G2X11_cap_probs_label)
data[4] = roc_plot(y_test=X2G11_cap_true_label,y_score=X2G11_cap_probs_label)


plot_roc(data,'G')


