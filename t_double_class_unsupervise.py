from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim 
from torch.optim import lr_scheduler
import matplotlib
from torch.nn.parallel.scatter_gather import gather
import matplotlib.pyplot as plt
from torch.autograd import Function
from sklearn import metrics
matplotlib.use('AGG')
import itertools
import math
import ipdb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as colors
from utils.utils import *
from utils.invariance import InvNet

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101','ResNet152']

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=24, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 1, places= 64)
        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))
        return nn.Sequential(*layers)


    def forward(self, inputs, lam):
        x = self.conv1(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)        
        x = self.avgpool(x)
        x0 = x.view(x.size(0), -1)         
        
        return x0

def ResNet18():
    return ResNet([2,2,2,2])

def ResNet34():
    return ResNet([3,4,6,3])

def ResNet50():
    return ResNet([3,4,6,3]),nn.Linear(2048, 24),nn.Linear(2048, 24),nn.Linear(2048, 4)

def ResNet101():
    return ResNet([3,4,23,3])

def ResNet152():
    return ResNet([3,8,36,3])
def default_loader(path):
    return Image.open(path)
class MyDataset (Dataset):
   
    def __init__(self, txt, transform=None, target_transform=None,rot=0):
        super(MyDataset,self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:           
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.rot = rot
        

    def __getitem__(self, index):
        fn, label = self.imgs[index]       
        img = Image.open(fn).convert('L')
        ro_u = np.random.uniform(0, 1)
        if self.rot:
            if ro_u <0.25:
                img = self.transform(img)
                rotation_labels = 0
            elif ro_u >=0.25 and ro_u <0.5:
                img = self.transform(img.rotate(90))
                rotation_labels = 1
            elif ro_u >=0.5 and ro_u <0.75:
                img = self.transform(img.rotate(180))
                rotation_labels = 2
            elif ro_u >=0.75:
                img = self.transform(img.rotate(270))
                rotation_labels = 3   
        else:
             img = self.transform(img)
             rotation_labels = 0
        return img, label, index, rotation_labels

    def __len__(self):
        return len(self.imgs)

def make_weights_for_balanced_classes(labels, nclasses):
    count = [0]*nclasses
    for item in labels:
        count[int(item)] += 1
    weight_per_class = np.zeros((nclasses,))
  
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    
    return weight_per_class

from sklearn import preprocessing
def np_softmax(x):
    #x_max = x.max()
    #x = x - x_max
    x = preprocessing.scale(x)
    x_exp = np.exp(x)
    softmax = x_exp / x_exp.sum() 
    return softmax
def edu(inputs,or_feature):
    tmp = torch.zeros(inputs.shape[0],or_feature.shape[0]).cuda()
    for i in range(inputs.shape[0]):
        tmp[i] = (torch.abs(inputs[i]- or_feature)).sum(1)
    return tmp 
def gen_pse(inputs, or_feature,or_label,or_label_weight,k=100):
    #inputs = inputs.mm(or_feature.t())
    #inputs = (torch.abs(inputs.unsqueeze(1)- or_feature)).sum(2)
    inputs = edu(inputs,or_feature)
    _, index_sorted = torch.sort(inputs, dim=1, descending=False)
    selct_label = or_label[index_sorted[:,:k]]
    B = inputs.shape[0]
    pse_label = np.zeros((B,24))
    j  = 0
    for x in selct_label:
        count = np.zeros((24,))
        for i in x:
            count[int(i)]+=1.0
        pse_label[j] = np_softmax(count*or_label_weight)
        #pse_label[j] = np.argmax(count*or_label_weight)
        j+=1
    #print(pse_label)
    return torch.from_numpy(pse_label)
       
#def plot_loss(epoch,or_test_loss_0,tar_test_loss_0,name):
#    axis = np.linspace(0, epoch, epoch+1)
#    label = 'Loss'
#    fig = plt.figure()
#    plt.title('Loss')
#    plt.plot(axis, np.array(or_test_loss_0), label="or_test_loss_0")
#    plt.plot(axis, np.array(tar_test_loss_0), label="tar_test_loss_0")
#
#    plt.legend()
#    plt.xlabel('Epochs')
#    plt.ylabel('Loss')
#    plt.grid(True)
#    plt.savefig('./model/'+name+'.jpg')
#    plt.close(fig)
    
    
def plot_loss(epoch,or_test_loss_0,or_test_loss_1,or_test_loss_2,tar_test_loss_0,tar_test_loss_1,tar_test_loss_2,name):
    axis = np.linspace(0, epoch, epoch+1)
    label = 'Loss'
    fig = plt.figure()
    plt.title('Loss')
    plt.plot(axis, np.array(or_test_loss_0), label="or_test_loss_0")
    plt.plot(axis, np.array(tar_test_loss_0), label="tar_test_loss_0")
    plt.plot(axis, np.array(or_test_loss_1), label="or_test_loss_1")
    plt.plot(axis, np.array(tar_test_loss_1), label="tar_test_loss_1")
    plt.plot(axis, np.array(or_test_loss_2), label="or_test_loss_2")
    plt.plot(axis, np.array(tar_test_loss_2), label="tar_test_loss_2")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('./model/'+name+'.jpg')
    plt.close(fig)
import time
class SymmNetsV2Solver(BaseSolver):
    def __init__(self, net, dataloaders, args,**kwargs):
        super(SymmNetsV2Solver, self).__init__(net, dataloaders,args, **kwargs)
        self.num_classes = 24

        self.feature_extractor = self.net['feature_extractor']
        self.F1 = self.net['F1']
        self.F2 = self.net['F2']
        self.Fr = self.net['Fr']
        
        self.CELoss = nn.CrossEntropyLoss()
        self.criterion_w = Weighted_CrossEntropy 
        self.or_test_loss_0 = []
        self.tar_test_loss_0 = []
        self.or_test_loss_1 = []
        self.tar_test_loss_1 = []
        self.or_test_loss_2 = []
        self.tar_test_loss_2 = []
  
        self.optimizer_g = optim.Adam(self.feature_extractor.parameters(), lr=0.0001, weight_decay=0.0005)
        self.optimizer_f = optim.Adam(list(self.F1.parameters()) + list(self.F2.parameters()), lr=0.0001, weight_decay=0.0005)
        self.optimizer_fr = optim.Adam(self.Fr.parameters(), lr=0.0001, weight_decay=0.0005)
        
        self.best_prec=0.0
        self.or_fea = torch.zeros(len(self.train_data['source']['loader'].dataset),2048).cuda()
        self.or_label = torch.zeros(len(self.train_data['source']['loader'].dataset))
        self.or_label_weight = None
        
        self.ks = args.ks
        self.train_pse = args.train_pse
        self.train_rot = args.train_rot
        self.eta = args.eta
        self.path = args.save_path + args.train_path + '_2_' + args.val_path + '/_ks_'+str(self.ks)+'_pse_'+str(self.train_pse)+'_rot_'+str(self.train_rot)+'_eta_'+str(self.eta)
        self.result_txt = open(self.path+"_res.txt", "w")
    def solve(self):
        stop = False
        while not stop:
            stop = self.complete_training()
            self.update_network()

            acc0,acc1,acc2 = self.test()
            acc01 = max(acc0,acc1)
            acc = max(acc01,acc2)
            if acc > self.best_prec:
                self.best_prec= acc
                torch.save(self.feature_extractor.state_dict(), self.path+'_E.pkl')
                torch.save(self.F1.state_dict(), self.path+'_F1.pkl')
                torch.save(self.F2.state_dict(), self.path+'_F2.pkl')
                print('best acc: ', self.best_prec)
                self.result_txt.write('\nbest acc: '+str(self.best_prec)+'\n')
            self.epoch += 1
            if self.epoch>100:
                self.result_txt.close()
                break

    def update_network(self, **kwargs):
        stop = False
        self.train_data['source']['iterator'] = iter(self.train_data['source']['loader'])
        self.train_data['target']['iterator'] = iter(self.train_data['target']['loader'])
        self.iters_per_epoch = max(len(self.train_data['target']['loader']), len(self.train_data['source']['loader']))
        iters_counter_within_epoch = 0
        
        self.feature_extractor.train()
        self.F1.train()
        self.F2.train()
        if self.train_rot:
            self.Fr.train()
        
        F1_loss_0 = 0.0
        F2_loss_0 = 0.0
        F1_loss_1 = 0.0
        F2_loss_1 = 0.0
        Fr_loss = 0.0
        F_dloss = 0.0
        pse_loss = 0.0
        if True:# 'epoch':
            lam = 2 / (1 + math.exp(-1 * 10 * self.epoch / 2000)) - 1
        while not stop:
            if False: # 'iteration':
                lam = 2 / (1 + math.exp(-1 * 10 * self.iters / (self.opt.TRAIN.MAX_EPOCH * self.iters_per_epoch))) - 1
            source_data, source_gt,or_indedx,s_rot_label = self.get_samples('source')
            target_data, _,index_target, t_rot_label  = self.get_samples('target')
            source_data = to_cuda(source_data)
            
            source_gt = to_cuda(source_gt)
            s_rot_label = to_cuda(s_rot_label)
            
            target_data = to_cuda(target_data)
            t_rot_label = to_cuda(t_rot_label)
           

            bs = len(source_gt)
            """source domain discriminative"""
            # Step A train all networks to minimize loss on source
            self.optimizer_g.zero_grad()
            self.optimizer_f.zero_grad()
            output = self.feature_extractor(torch.cat((source_data, target_data), 0),lam)
            
            output1 = F1(output)
            output2 = F2(output)
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1_s = F.softmax(output_t1)
            output_t2_s = F.softmax(output_t2)

            entropy_loss = Entropy(output_t1_s)
            entropy_loss += Entropy(output_t2_s)
            #entropy_loss =  - torch.mean(torch.log(torch.mean(output_t1_s,0)+1e-6))
            #entropy_loss -= - torch.mean(torch.log(torch.mean(output_t2_s,0)+1e-6))
            if self.epoch==1:
                self.or_label_weight= make_weights_for_balanced_classes(self.or_label, 24)
            if self.epoch>3 and self.train_pse:
                pseudo_label_t_1 = to_cuda(gen_pse(output[bs:, :].detach(), self.or_fea.detach(),self.or_label,self.or_label_weight,k=self.ks))
                #pseudo_label_t_2 = gen_pse(output[bs:, :].detach(), self.or_fea.detach(),self.or_label,self.or_label_weight,k=100).to(source_gt)
                #supervision_loss = self.criterion_w(output_t1, pseudo_label_t_1) + self.criterion_w(output_t2, pseudo_label_t_2)
                supervision_loss = torch.mean(torch.mul(-F.log_softmax(output_t1, dim=1), pseudo_label_t_1)) + torch.mean(torch.mul(-F.log_softmax(output_t2, dim=1), pseudo_label_t_1))
            else:
                supervision_loss = 0
            loss1 = self.CELoss(output_s1, source_gt)
            loss2 = self.CELoss(output_s2, source_gt)
            all_loss = loss1 + loss2 + 0.1 * entropy_loss + 0.05 * supervision_loss
            all_loss.backward()
            self.optimizer_g.step()
            self.optimizer_f.step()
            pse_loss += float(supervision_loss) 
            F1_loss_0 += float(loss1)
            F2_loss_0 += float(loss2)
            
            """target domain diversity"""
            # Step B train classifier to maximize CDD loss
            self.optimizer_g.zero_grad()
            self.optimizer_f.zero_grad()
            output = self.feature_extractor(torch.cat((source_data, target_data), 0),lam)
            
            for x,y,z in zip(output[:bs, :], source_gt,or_indedx):
                self.or_fea[z] = self.eta*x + (1-self.eta)*self.or_fea[z]
                self.or_label[z] = y
            
            output1 = self.F1(output)
            output2 = self.F2(output)
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1_s = F.softmax(output_t1)
            output_t2_s = F.softmax(output_t2)

            loss1 = self.CELoss(output_s1, source_gt)
            loss2 = self.CELoss(output_s2, source_gt)
            
#            entropy_loss =  - torch.mean(torch.log(torch.mean(output_t1_s,0)+1e-6))
#            entropy_loss -= - torch.mean(torch.log(torch.mean(output_t2_s,0)+1e-6))
            entropy_loss = Entropy(output_t1_s)
            entropy_loss += Entropy(output_t2_s)

            loss_dis = discrepancy(output_t1,output_t2)

            all_loss = loss1 + loss2 - 0.1 * loss_dis + 0.1 * entropy_loss
            all_loss.backward()
            self.optimizer_f.step()
            F1_loss_1 += float(loss1)
            F2_loss_1+= float(loss2)
            F_dloss += float(loss_dis)           
            
            # Step C train classifier to maximize CDD loss
            if self.train_rot:
                self.optimizer_g.zero_grad()
                self.optimizer_fr.zero_grad()
                output = self.feature_extractor(torch.cat((source_data, target_data), 0),lam)
                output1 = self.F1(output)
                output2 = self.F2(output)
                output_s1 = output1[:bs, :]
                output_s2 = output2[:bs, :]
                output_t1 = output1[bs:, :]
                output_t2 = output2[bs:, :]
                output_t1_s = F.softmax(output_t1)
                output_t2_s = F.softmax(output_t2) 
                entropy_loss = Entropy(output_t1_s)
                entropy_loss += Entropy(output_t2_s)          
                output_rot = self.Fr(output)
                loss_r = self.CELoss(output_rot, torch.cat((s_rot_label, t_rot_label), 0))
                loss = loss_r+0.1*entropy_loss
                loss.backward()
                self.optimizer_g.zero_grad()
                self.optimizer_fr.step()
                Fr_loss += float(loss_r)
          
            end = time.time()
            self.iters += 1
            iters_counter_within_epoch += 1
            if iters_counter_within_epoch >= self.iters_per_epoch:
                stop = True
        print('step 1 F1 loss: ',F1_loss_0/self.iters_per_epoch)
        print('step 1 F2 loss: ',F2_loss_0/self.iters_per_epoch)
        print('step 2 F1 loss: ',F1_loss_1/self.iters_per_epoch)
        print('step 2 F2 loss: ',F2_loss_1/self.iters_per_epoch)
        print('Fr loss: ',Fr_loss/self.iters_per_epoch)
        print('F_dloss  loss: ',F_dloss/self.iters_per_epoch)
        print('pse_loss  loss: ',pse_loss/self.iters_per_epoch)
        self.result_txt.write('\nstep 1 F1 loss: '+str(F1_loss_0/self.iters_per_epoch)+'\n')
        self.result_txt.write('step 1 F2 loss: '+str(F2_loss_0/self.iters_per_epoch)+'\n')
        self.result_txt.write('step 2 F1 loss: '+str(F1_loss_1/self.iters_per_epoch)+'\n')
        self.result_txt.write('step 2 F2 loss: '+str(F2_loss_1/self.iters_per_epoch)+'\n')
        self.result_txt.write('Fr loss: '+str(Fr_loss/self.iters_per_epoch)+'\n')
        self.result_txt.write('F_dloss  loss: '+str(F_dloss/self.iters_per_epoch)+'\n')
        self.result_txt.write('pse_loss  loss: '+str(pse_loss/self.iters_per_epoch)+'\n')
    def test(self):
        self.feature_extractor.eval()
        self.F1.eval()
        self.F2.eval()
        tar_test_l_0 = 0.0
        or_test_l_0 = 0.0
        tar_test_l_1 = 0.0
        or_test_l_1 = 0.0
        tar_test_l_2 = 0.0
        or_test_l_2 = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            
            correct_0 = 0.0
            correct_1 = 0.0
            correct_2 = 0.0
            n = 0
            for i, data in enumerate(dataloaders['or_test'], 0):
                n+=1
                test_data, target,_ ,_= data
                test_data, target = test_data.cuda(), target.cuda()
                test_data, target = Variable(test_data), Variable(target)
                label = target.data.cpu().numpy()
                output = self.feature_extractor(test_data,0.0)
                output0 = self.F1(output)
                output1 = self.F2(output)


                pred_0 = output0.data.max(1, keepdim=True)[1]  # get the index of the max probability
                correct_0 += pred_0.eq(target.data.view_as(pred_0)).cpu().sum()
                loss = criterion(output0, target)
                or_test_l_0+=float(loss)
      
                pred_1 = output1.data.max(1, keepdim=True)[1]  # get the index of the max probability
                correct_1 += pred_1.eq(target.data.view_as(pred_1)).cpu().sum()
                loss = criterion(output1, target)
                or_test_l_1+=float(loss)
                    
                output_add = output0 + output1
                loss = criterion(output_add, target)
                or_test_l_2+=float(loss)
                pred_2 = output_add.data.max(1)[1]
                correct_2 += pred_2.eq(target.data.view_as(pred_2)).cpu().sum()
            self.or_test_loss_0.append(or_test_l_0/n)
            self.or_test_loss_1.append(or_test_l_1/n)
            self.or_test_loss_2.append(or_test_l_2/n)
            print('\n*****or_test_0****Accuracy: {}/{} ({:.4f}%)\n'.format(correct_0, len(dataloaders['or_test'].dataset),100. * correct_0 / len(dataloaders['or_test'].dataset)))
            print('\n*****or_test_1****Accuracy: {}/{} ({:.4f}%)\n'.format(correct_1, len(dataloaders['or_test'].dataset),100. * correct_1 / len(dataloaders['or_test'].dataset)))
            print('\n*****or_test_2****Accuracy: {}/{} ({:.4f}%)\n'.format(correct_2, len(dataloaders['or_test'].dataset),100. * correct_2 / len(dataloaders['or_test'].dataset)))
            self.result_txt.write('\n*****or_test_0****Accuracy: '+str(correct_0)+'/'+ str(len(dataloaders['or_test'].dataset))+' '+str(100. * correct_0 / len(dataloaders['or_test'].dataset)) + '\n')
            self.result_txt.write('*****or_test_1****Accuracy: '+str(correct_1)+'/'+ str(len(dataloaders['or_test'].dataset))+' '+str(100. * correct_1 / len(dataloaders['or_test'].dataset)) + '\n')
            self.result_txt.write('*****or_test_2****Accuracy: '+str(correct_2)+'/'+ str(len(dataloaders['or_test'].dataset))+' '+str(100. * correct_2 / len(dataloaders['or_test'].dataset)) + '\n')
            correct_0 = 0.0
            correct_1 = 0.0
            correct_2 = 0.0
            n = 0
            for i, data in enumerate(dataloaders['tar_test'], 0):
                n+=1
                test_data, target,_,_ = data
                test_data, target = test_data.cuda(), target.cuda()
                test_data, target = Variable(test_data), Variable(target)
                label = target.data.cpu().numpy()
                output = self.feature_extractor(test_data,0.0)
                output0 = self.F1(output)
                output1 = self.F2(output)

                pred_0 = output0.data.max(1, keepdim=True)[1]  # get the index of the max probability
                correct_0 += pred_0.eq(target.data.view_as(pred_0)).cpu().sum()
                loss = criterion(output0, target)
                tar_test_l_0+=float(loss)
      
                pred_1 = output1.data.max(1, keepdim=True)[1]  # get the index of the max probability
                correct_1 += pred_1.eq(target.data.view_as(pred_1)).cpu().sum()
                loss = criterion(output1, target)
                tar_test_l_1+=float(loss)
                    
                output_add = output0 + output1
                loss = criterion(output_add, target)
                tar_test_l_2+=float(loss)
                pred_2 = output_add.data.max(1)[1]
                correct_2 += pred_2.eq(target.data.view_as(pred_2)).cpu().sum()
            self.tar_test_loss_0.append(tar_test_l_0/n)
            self.tar_test_loss_1.append(tar_test_l_1/n)
            self.tar_test_loss_2.append(tar_test_l_2/n)
            print('\n*****tar_test_0****Accuracy: {}/{} ({:.4f}%)\n'.format(correct_0, len(dataloaders['tar_test'].dataset),100. * correct_0 / len(dataloaders['tar_test'].dataset)))
            print('\n*****tar_test_1****Accuracy: {}/{} ({:.4f}%)\n'.format(correct_1, len(dataloaders['tar_test'].dataset),100. * correct_1 / len(dataloaders['tar_test'].dataset)))
            print('\n*****tar_test_2****Accuracy: {}/{} ({:.4f}%)\n'.format(correct_2, len(dataloaders['tar_test'].dataset),100. * correct_2 / len(dataloaders['tar_test'].dataset)))
            
            self.result_txt.write('\n*****tar_test_0****Accuracy: '+str(correct_0)+'/'+ str(len(dataloaders['tar_test'].dataset))+' '+str(100. * correct_0 / len(dataloaders['tar_test'].dataset)) + '\n')
            self.result_txt.write('*****tar_test_1****Accuracy: '+str(correct_1)+'/'+ str(len(dataloaders['tar_test'].dataset))+' '+str(100. * correct_1 / len(dataloaders['tar_test'].dataset)) + '\n')
            self.result_txt.write('*****tar_test_2****Accuracy: '+str(correct_2)+'/'+ str(len(dataloaders['tar_test'].dataset))+' '+str(100. * correct_2 / len(dataloaders['tar_test'].dataset)) + '\n')
            
            print('tar_test_l_0: ',tar_test_l_0/n)
            print('tar_test_l_1: ',tar_test_l_1/n)
            print('tar_test_l_2: ',tar_test_l_2/n)
            print('or_test_l_0: ',or_test_l_0/n)
            print('or_test_l_1: ',or_test_l_1/n)
            print('or_test_l_2: ',or_test_l_2/n)
            self.result_txt.write('\ntar_test_l_0: '+str(tar_test_l_0/n))
            self.result_txt.write('tar_test_l_1: '+str(tar_test_l_1/n))
            self.result_txt.write('tar_test_l_2: '+str(tar_test_l_2/n))
            self.result_txt.write('\nor_test_l_0: '+str(or_test_l_0/n))
            self.result_txt.write('or_test_l_1: '+str(or_test_l_1/n))
            self.result_txt.write('or_test_l_2: '+str(or_test_l_2/n))
            return float(100. * correct_0 / len(dataloaders['tar_test'].dataset)), float(100. * correct_1 / len(dataloaders['tar_test'].dataset)), float(100. * correct_2 / len(dataloaders['tar_test'].dataset))

import matplotlib.colors
import random
from colorsys import hls_to_rgb
cmp = []

def set_plt_style():
    # Style sheets reference
    # https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    #plt.style.use('default')
    plt.style.use('bmh')
    # plt.style.use('seaborn-dark')
    #plt.style.use('dark_background')
    # plt.style.use('classic')
    # plt.style.use('fivethirtyeight')
     #plt.style.use('ggplot')

def generate_colormap():
    # reference: 
    # 1 How to automatically generate N
    #   https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
    # 2 HSL Color Wheel
    #   https://pythonfordesigners.com/tutorials/hsl-color-wheel/
    # 3 How to convert HSL colors to RGB colors in python?
    #   http://linuxcursor.com/python-programming/how-to-convert-hsl-to-rgb-color-in-python
    # 4 Matplotlib 
    #   https://blog.csdn.net/sinat_32570141/article/details/105226330
    # 5 i want hue
    #   https://medialab.github.io/iwanthue/
    x= np.linspace(1, 12, 24, endpoint=True)
    y=x/x
    
    #fig = plt.figure()
    #ax= plt.axes()
    
    
    # assumes hue [0, 360), saturation [0, 100), lightness [0, 100)
    global clr_list
    clr_list = []
    hue = 0
    num_colors = 24
    for i in range(0, num_colors):
        hue += int(360/num_colors)  # 
        saturation = 20 + random.randint(0,9)    
        lightness = 40 + random.randint(0,9) 
        if (i % 2) == 0:    
            saturation += 10
            lightness += 10
                
        clr_rgb = hls_to_rgb(hue / 360, lightness / 100, saturation / 100)
        clr_list.append(clr_rgb)
        # random.shuffle(clr_list)
    
   
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
def plot_embedding(data, label):
    #x_min, x_max = np.min(data, 0), np.max(data, 0)
    #data = (data - x_min) / (x_max - x_min)
    data = normalization(data)


    ax = plt.figure(figsize=(3,3), dpi=300)
    #ax = plt.subplot(3,3)
    
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #global c 
    #c = ["#f16d7a","#ae716e","#cf8878","#f1ccb8","#b7d28d","#dcff93","#f1b8e4","#f1ccb8","#f1f1b8","#e1622f","#f3d64e","#c38e9e","#de772c","#edbf2b","#ff8240","#f0b631","#ff8240","#edaa2b","#e7dac9","#cb8e85","#ff9b6a","#d9b8f1","#b8f1ed","#e7dac9","#fe9778"]
    for i in range(data.shape[0]):
#        plt.text(data[i, 0], data[i, 1], str(label[i]),
#                 color=plt.cm.Set1(label[i] / 10.),
#                 fontdict={'weight': 'bold', 'size': 9})
        #plt.text(data[i, 0], data[i, 1], str(label[i]),color=c[int(label[i])],fontdict={'weight': 'bold', 'size': 3})
        
        plt.text(data[i, 0], data[i, 1], ".",color=clr_list[int(label[i])],fontdict={'weight': 'bold', 'size': 9})
        
        #plt.scatter(data[i, 0],data[i, 1],c="gold",marker=".")
    plt.xticks([])
    plt.yticks([])
    #plt.title(r'$\mathrm{Times \; New \; Roman}\/\/ $')
    #plt.title(xy)
    #return 1
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score 
from matplotlib import rcParams  
def roc_plot(y_test=None,y_score=None,title=None,file_name=None):
    config = {
        "font.family":'Times New Roman',  
        "font.size":7,
    }
    rcParams.update(config)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Binarize the output 
    n_classes = 24
    y_test = label_binarize(y_test, classes=list(range(24)))
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

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    macro_auc = roc_auc_score(y_test, y_score, average='macro')


    # Plot all ROC curves
    #fig = plt.figure()
    fig, ax = plt.subplots(figsize=(5,5))
    mean_auc = 0.0
    for i in range(n_classes):
        pt_, = plt.plot(fpr[i], tpr[i], lw=2,label='ROC curve of class {0} (area = {1:0.2f})'.format(i+1, roc_auc[i]))
        mean_auc+=roc_auc[i]
    mean_auc=round(mean_auc/24,4)    
    print('mean AUC: ',mean_auc)
    plt.tick_params(labelsize=14)
    labels_ = ax.get_xticklabels() + ax.get_yticklabels()

    [label_.set_fontname('Times New Roman') for label_ in labels_]
    font2 = {
        "family":'Times New Roman',  
        "size": 18,
    }
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',font2)
    plt.ylabel('True Positive Rate',font2)
    plt.title(title,font2)
    plt.legend(loc="lower right")    
    plt.savefig(file_name+'_'+str(mean_auc)+'_roc4.jpg')
    plt.close(fig)
if __name__=='__main__':

    parser = argparse.ArgumentParser(description='DomainNet Classification')
    parser.add_argument('--eta', type=float, default=0.3, metavar='M', help='F update ')
    parser.add_argument('--ks', type=int, default='100', metavar='B', help='The ks for pse')
    parser.add_argument('--train_rot', type=int, default='1', metavar='B', help='add train rot')
    parser.add_argument('--train_pse', type=int, default='0', metavar='B', help='add train pse')
    parser.add_argument('--train_aug', type=int, default='1', metavar='B', help='add train aug')
    parser.add_argument('--train_path', type=str, default='G', metavar='B',
                    help='directory of source datasets')
    parser.add_argument('--val_path', type=str, default='X', metavar='B',
                    help='directory of target datasets')
    parser.add_argument('--save_path', type=str, default="/home_lv/xinyu.fan/myUnsupervised/unsuper/result/", metavar='B',
                    help='directory of target datasets')
    args = parser.parse_args()
    #data_aug
    train_transforms = transforms.Compose([
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomAffine(10, translate=(0.1,0.1), scale=(0.8,1.2), shear=None, resample=False, fillcolor=0),
            transforms.Resize((224,224)),            
            transforms.ToTensor(),

            ])
    test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485], [0.229]),
            ])
    #loss_fn = MarginLoss(0.9, 0.1, 0.5)
    #data_loader 
    if args.train_path == 'G' and args.val_path == 'X':
        # or_trainset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/train0.txt",transform=train_transforms,rot=1)
        # or_testset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test0.txt", transform=test_transform)
        # tar_testset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/XJ_data/data_224/test0.txt", transform=test_transform)
        or_trainset = MyDataset(txt="/home_lv/xinyu.fan/myUnsupervised/data/GK_gdata/data_224/train0.txt",
                                transform=train_transforms, rot=1)
        or_testset = MyDataset(txt="/home_lv/xinyu.fan/myUnsupervised/data/GK_gdata/data_224/test0.txt",
                               transform=test_transform)
        tar_testset = MyDataset(txt="/home_lv/xinyu.fan/myUnsupervised/data/XJ_data/data_224/test0.txt",
                                transform=test_transform)
        # tar_trainset_0 = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/XJ_data/data_224/train0.txt",transform=train_transforms,rot=args.train_rot)
        # tar_trainset_1 = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/XJ_data/data_224/test0.txt", transform=train_transforms,rot=args.train_rot)
        # tar_trainset = ConcatDataset([tar_trainset_0]+[tar_trainset_1])
        tar_trainset = MyDataset(txt="/home_lv/xinyu.fan/myUnsupervised/data/XJ_data/data_224/test0.txt", transform=train_transforms,rot=1)
    if args.train_path == 'X' and args.val_path == 'G':
        or_trainset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/XJ_data/data_224/train0.txt", transform=train_transforms,rot=1)
        or_testset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/XJ_data/data_224/test0.txt", transform=test_transform)
        tar_testset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test0.txt", transform=test_transform)
#        tar_trainset_0 = MyDataset(txt="//home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/train0.txt", transform=train_transforms,rot=args.train_rot)
#        tar_trainset_1 = MyDataset(txt="//home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test0.txt", transform=train_transforms,rot=args.train_rot)
#        tar_trainset = ConcatDataset([tar_trainset_0]+[tar_trainset_1])
        tar_trainset = MyDataset(txt="//home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test0.txt", transform=train_transforms,rot=1)

#GK fish
    if args.train_path == 'F' and args.val_path == 'G':
        or_trainset = MyDataset(txt="/home/xianpeng.yi/muti_part/data/train0.txt", transform=train_transforms,rot=args.train_rot)  #fish->GK
        or_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data/test0.txt", transform=test_transform)
        tar_testset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test0.txt", transform=test_transform)
#        tar_trainset_0 = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/train0.txt", transform=train_transforms,rot=args.train_rot)
#        tar_trainset_1 = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test0.txt", transform=train_transforms,rot=args.train_rot)
#        tar_trainset = ConcatDataset([tar_trainset_0]+[tar_trainset_1])
        tar_trainset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test0.txt", transform=train_transforms,rot=args.train_rot)
    if args.train_path == 'G' and args.val_path == 'F':
        or_trainset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/train0.txt", transform=train_transforms,rot=args.train_rot)
        or_testset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test0.txt", transform=test_transform)
        tar_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data/test0.txt", transform=test_transform)  #GK->fish
#        tar_trainset_0 = MyDataset(txt="/home/xianpeng.yi/muti_part/data/train0.txt",transform=train_transforms,rot=args.train_rot)
#        tar_trainset_1 = MyDataset(txt="/home/xianpeng.yi/muti_part/data/test0.txt", transform=train_transforms,rot=args.train_rot)
#        tar_trainset = ConcatDataset([tar_trainset_0]+[tar_trainset_1])
        tar_trainset = MyDataset(txt="/home/xianpeng.yi/muti_part/data/test0.txt", transform=train_transforms,rot=args.train_rot)
#qband gk
    if args.train_path == 'Q' and args.val_path == 'G':
        or_trainset = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/train0.txt", transform=train_transforms,rot=args.train_rot)  #qband->GK
        or_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/test0.txt", transform=test_transform)
        tar_testset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test0.txt", transform=test_transform)
#        tar_trainset_0 = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/train0.txt", transform=train_transforms,rot=args.train_rot)
#        tar_trainset_1 = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test0.txt", transform=train_transforms,rot=args.train_rot)
#        tar_trainset = ConcatDataset([tar_trainset_0]+[tar_trainset_1])
        tar_trainset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test0.txt", transform=train_transforms,rot=args.train_rot)
    if args.train_path == 'G' and args.val_path == 'Q':
        or_trainset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/train0.txt", transform=train_transforms,rot=args.train_rot)
        or_testset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test0.txt", transform=test_transform)
        tar_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/test1.txt", transform=test_transform)  
#        tar_trainset_0 = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/train0.txt", transform=train_transforms,rot=args.train_rot)  #GK->qband
#        tar_trainset_1 = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/test0.txt", transform=train_transforms,rot=args.train_rot)
#        tar_trainset = ConcatDataset([tar_trainset_0]+[tar_trainset_1])
        tar_trainset = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/test0.txt", transform=train_transforms,rot=args.train_rot)
#fish qband
    if args.train_path == 'Q' and args.val_path == 'F':
        or_trainset = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/train0.txt", transform=train_transforms,rot=args.train_rot)  #qband->fish
        or_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/test0.txt", transform=test_transform)
        tar_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data/test0.txt", transform=test_transform)
#        tar_trainset_0 = MyDataset(txt="/home/xianpeng.yi/muti_part/data/train0.txt", transform=train_transforms,rot=args.train_rot)
#        tar_trainset_1 = MyDataset(txt="/home/xianpeng.yi/muti_part/data/test0.txt", transform=train_transforms,rot=args.train_rot)
#        tar_trainset = ConcatDataset([tar_trainset_0]+[tar_trainset_1])
        tar_trainset = MyDataset(txt="/home/xianpeng.yi/muti_part/data/test0.txt", transform=train_transforms,rot=args.train_rot)
    if args.train_path == 'F' and args.val_path == 'Q':
        or_trainset = MyDataset(txt="/home/xianpeng.yi/muti_part/data/train0.txt", transform=train_transforms,rot=args.train_rot)
        or_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data/test0.txt", transform=test_transform)
        tar_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/test0.txt", transform=test_transform)  #fish->qband
#        tar_trainset_0 = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/train0.txt", transform=train_transforms,rot=args.train_rot)  
#        tar_trainset_1 = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/test0.txt", transform=train_transforms,rot=args.train_rot) #fish->qband
#        tar_trainset = ConcatDataset([tar_trainset_0]+[tar_trainset_1])
        tar_trainset = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/test0.txt", transform=train_transforms,rot=args.train_rot)
    
    or_trainloader = torch.utils.data.DataLoader(or_trainset, batch_size= 32, shuffle=True,num_workers=2,pin_memory=True)
    or_testloader= torch.utils.data.DataLoader(or_testset, batch_size= 64, shuffle=True,num_workers=2,pin_memory=True)
    tar_testloader= torch.utils.data.DataLoader(tar_testset, batch_size= 64, shuffle=True,num_workers=2,pin_memory=True)
    tar_trainloader= torch.utils.data.DataLoader(tar_trainset, batch_size= 32, shuffle=True,num_workers=2,pin_memory=True)
    dataloaders = {}
    dataloaders['source'] = or_trainloader
    dataloaders['target'] = tar_trainloader
    dataloaders['or_test'] = or_testloader
    dataloaders['tar_test'] = tar_testloader 
    model,F1,F2,Fr = ResNet50()
    path = args.save_path + args.train_path +'_2_'+args.val_path +'/_ks_'+str(args.ks)+'_pse_'+str(args.train_pse)+'_rot_'+str(args.train_rot)+'_eta_'+str(args.eta)+ '_aug_'+str(args.train_aug)
    #model = nn.DataParallel(model)
    model = model.cuda()
    F1 = F1.cuda()
    F2 = F2.cuda()

    model.load_state_dict(torch.load(path+'_E.pkl'))
    F1.load_state_dict(torch.load(path+'_F1.pkl'))
    F2.load_state_dict(torch.load(path+'_F2.pkl'))
    model.eval()
    F1.eval()
    F2.eval()
    mf = nn.Softmax(dim=1)
    with torch.no_grad():
        f_data = None
        y_true = []
        y_pre0 = []
        y_pre1 = []
        y_pre2 = []
        probs_label0 = None
        probs_label1 = None
        probs_label2 = None
        for i, data in enumerate(tar_testloader, 0):
             test_data, target,_ , _ = data
             test_data, target = test_data.cuda(), target.cuda()
             test_data, target = Variable(test_data), Variable(target)
             output = model(test_data,0.0)
             output0 = F1(output)
             output1 = F2(output)
             #f_data+=list(output.data.cpu().numpy())
             y_true+=list(target.data.cpu().numpy())
             y_pre0.append(output0.data.max(1, keepdim=True)[1].cpu().numpy())
             y_pre1.append(output1.data.max(1, keepdim=True)[1].cpu().numpy())
             y_pre2.append((output0+output1).data.max(1, keepdim=True)[1].cpu().numpy())
             probs = mf(output)
             if i==0:
                probs_label0 = mf(output0).data.cpu().numpy()
                probs_label1 = mf(output1).data.cpu().numpy()
                probs_label2 = mf(output0+output1).data.cpu().numpy()
                f_data =  output.data.cpu().numpy()
             else:
                probs_label0 = np.concatenate((probs_label0,mf(output0).data.cpu().numpy()),axis=0)
                probs_label1 = np.concatenate((probs_label1,mf(output1).data.cpu().numpy()),axis=0)
                probs_label2 = np.concatenate((probs_label2,mf(output0+output1).data.cpu().numpy()),axis=0)
                f_data =  np.concatenate((f_data,output.data.cpu().numpy()),axis=0)
        tsne1 = TSNE(n_components=2, init='pca',n_iter=1000,min_grad_norm =1e-8, random_state=0,method="exact")
        #roc_plot(y_test=y_true,y_score=probs_label0,file_name=path+'_c0')   
        #roc_plot(y_test=y_true,y_score=probs_label1,file_name=path+'_c1')
        np.save(path+'true_label.npy',np.array(y_true))
        np.save(path+'probs_label.npy',probs_label2)   
        #roc_plot(y_test=y_true,y_score=probs_label2,file_name=path+'_c2')  
        f_data = PCA(n_components=50).fit_transform(f_data) # dim reduction 
        f_data = tsne1.fit_transform(f_data)
        from t_sne import plot_tsne
        np.save(path+'f_data.npy',f_data)
        np.save(path+'f_true_label.npy',np.array(y_true))
        #plot_tsne(f_data,y_true,path)
    
