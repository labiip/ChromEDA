# coding=gbk
import matplotlib
import matplotlib.pyplot as plt
# print(matplotlib.matplotlib_fname())
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
matplotlib.use('Agg')
import os
import pandas as pd
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
config = {
   "font.family":'Times New Roman',
   "font.size": 10,
#     "mathtext.fontset":'stix',
}
# config = {
#             "font.family": 'serif',
#             "font.size": 10,
#             "mathtext.fontset": 'stix',
#             "font.serif": ['SimSun'],
#          }

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

def organize_data(y,x):
    data_plot = pd.DataFrame(columns = ['x','y','chr'])
    y = np.array(y)
    
    # zoom all data to [0.05  0.95]
    data_plot.x = ((x[:,0] - min(x[:,0])) / (max(x[:, 0]) - min(x[:, 0]))) * 0.9 + 0.05
    data_plot.y = ((x[:, 1] - min(x[:, 1])) / (max(x[:, 1]) - min(x[:, 1]))) * 0.9 + 0.05
    data_plot.chr = y.astype(int)
    
    return data_plot
    
def plot_t_sne(data,file_name):

    n_classes = 24

    title = ['ResNet50','+Adver.','+Adver.+ Pse.','+Adver.+ Ang.','MCM']
    plt1= None  
    fig = plt.figure(figsize=(16,25))
    fontsize=32
    for m in range(5):
        for n in range(2):
            data_plot = data[m+n*5]
            plt.subplot(5,2,m*2+n+1)            
            for i in range(n_classes):
                index = data_plot.chr == i
                plt.scatter(data_plot.x[index], data_plot.y[index], s = 1, alpha=1, c = [clr_list[i]])
                # if n==0:
                #     pass
                #     # plt.ylabel(title[m],fontsize=fontsize)
                # if m==0 and n==0:
                #     # plt.title('G-bandI->GII',fontsize=fontsize)
                #     plt.title('GI->GII', fontsize=fontsize)
                    
                # if m==0 and n==1:
                #     # plt.title('GII->GI',fontsize=fontsize)
                #     plt.title('GII->GI', fontsize=fontsize)
                plt.xticks([])
                plt.yticks([]) 
                #plt.legend(loc="lower right") 
            
    # plt.text(-0.9,4.33,'G10',fontsize=fontsize)
    # arrow = plt.arrow(-0.7,4.33, 0.10, 0., head_width=0.2, head_length=0.4 )
    # plt.text(-0.55,4.35,'G21',fontsize=fontsize)
    
    # plt.text(0.25,4.33,'G22',fontsize=fontsize)
    # plt.arrow(0.45, 4.33, 0.10, 0.1, head_width=0.2, head_length=0.4)
    # plt.text(0.6,4.33,'G13',fontsize=fontsize)
    
    for i in range(24):
        if i == 22:
            plt.text(1.1,5.3-i*0.23,'chr . X',color=clr_list[i],fontsize=30)
        elif i == 23:
            plt.text(1.1, 5.3 - i * 0.23, 'chr . Y', color=clr_list[i], fontsize=30)
        else:
            plt.text(1.1,5.3-i*0.23,'chr . {0}'.format(i+1),color=clr_list[i],fontsize=30)

    plt.subplots_adjust(left=0.08, bottom=0.01, right=0.83, top=0.97,wspace=0.2, hspace=0.1)   
    plt.savefig('fxy6_t_sne.tiff', dpi=300)
    plt.close(fig)
    fontsize=28

#    plt.text(-0.79,4.06,'G1->G2',fontsize=fontsize)
#    plt.text(0.3,4.06,'G2->G1',fontsize=fontsize)
#    
#     plt.text(-1.19,3.388,title[0],rotation=270,fontsize=fontsize)
#     plt.text(-1.19,2.21,title[1],rotation=270,fontsize=fontsize)
#     plt.text(-1.19,1.38,title[2],rotation=270,fontsize=fontsize)
#     plt.text(-1.19,0.36,title[3],rotation=270,fontsize=fontsize)



#    plt.subplots_adjust(left=0.06, bottom=0.04, right=0.98, top=0.95,wspace=0.1, hspace=0.02)
#    #plt.tight_layout()   
#    plt.savefig(file_name+'t_sne.jpg', dpi=300)
#    plt.close(fig)



G2X_re_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/resnet/G_2_X_f_true_label.npy")
G2X_re_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/resnet/G_2_X_f_data.npy")

G2X00_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_0_rot_0_eta_0.3_aug_1f_true_label.npy")
G2X00_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_0_rot_0_eta_0.3_aug_1f_data.npy")

G2X01_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_0_rot_1_eta_0.3_aug_1f_true_label.npy")
G2X01_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_0_rot_1_eta_0.3_aug_1f_data.npy")

G2X10_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_1_rot_0_eta_0.3_aug_1f_true_label.npy")
G2X10_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_1_rot_0_eta_0.3_aug_1f_data.npy")

G2X11_cap_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_1_rot_1_eta_0.3_aug_1f_true_label.npy")
G2X11_cap_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/G_2_X/_ks_100_pse_1_rot_1_eta_0.3_aug_1f_data.npy")

X2G_re_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/resnet/X_2_G_f_true_label.npy")
X2G_re_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/resnet/X_2_G_f_data.npy")

X2G00_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_0_rot_0_eta_0.3_aug_1f_true_label.npy")
X2G00_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_0_rot_0_eta_0.3_aug_1f_data.npy")

X2G01_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_0_rot_1_eta_0.3_aug_1f_true_label.npy")
X2G01_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_0_rot_1_eta_0.3_aug_1f_data.npy")

X2G10_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_1_rot_0_eta_0.3_aug_1f_true_label.npy")
X2G10_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_1_rot_0_eta_0.3_aug_1f_data.npy")

X2G11_cap_true_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_1_rot_1_eta_0.3_aug_1f_true_label.npy")
X2G11_cap_probs_label=np.load("/home_lv/xinyu.fan/myUnsupervised/unsuper/result/X_2_G/_ks_100_pse_1_rot_1_eta_0.3_aug_1f_data.npy")

data = [[],[],[],[],[],[],[],[],[],[]]

data[5] = organize_data(G2X_re_true_label,G2X_re_probs_label)
data[0] = organize_data(X2G_re_true_label,X2G_re_probs_label)
data[7] = organize_data(G2X00_true_label,G2X00_probs_label)
data[2] = organize_data(X2G00_true_label,X2G00_probs_label)
data[8] = organize_data(G2X01_true_label,G2X01_probs_label)
data[3] = organize_data(X2G00_true_label,X2G01_probs_label)
data[6] = organize_data(G2X10_true_label,G2X10_probs_label)
data[1] = organize_data(X2G10_true_label,X2G10_probs_label)
data[9] = organize_data(G2X11_cap_true_label,G2X11_cap_probs_label)
data[4] = organize_data(X2G11_cap_true_label,X2G11_cap_probs_label)



plot_t_sne(data,'G')