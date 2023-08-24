from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
class MyDataset (Dataset):
   
  def __init__(self, txt, transform=None, target_transform=None):
    super(MyDataset,self).__init__()
    fh = open(txt, 'r')
    imgs = []
    for line in fh:           
      line = line.strip('\n')
      line = line.rstrip()
      words = line.split()
      imgs.append((words[0], int(words[1])))
    self.imgs = imgs
    self.trainortest = transform
    self.transform_train_rotation= transforms.Compose([
    transforms.RandomChoice([transforms.RandomRotation((59,61), resample=False, expand=False),
    transforms.RandomRotation((119,121), resample=False, expand=False),
    transforms.RandomRotation((179,181), resample=False, expand=False),]),            
    transforms.ToTensor(),
    ])
    self.transform_train_translate = transforms.Compose([
    transforms.RandomAffine(0, translate=(0.12,0.1), fillcolor=0),            
    transforms.ToTensor(),
    ])
    
    self.transform_train_scale = transforms.Compose([
    transforms.RandomAffine(0, scale=(0.8,1.2), fillcolor=0),            
    transforms.ToTensor(),
    ])
    self.transform_train_shear= transforms.Compose([
    transforms.RandomAffine(0, shear=(-60.0,60.0,-60.0,-60.0), resample=False, fillcolor=0),            
    transforms.ToTensor(),
    ])
    self.target_transform = target_transform
        
    self.transforms_train_sum = transforms.Compose([
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomAffine(60, translate=(0.1,0.1), scale=(0.8,1.2), shear=(-60.0,60.0,-60.0,-60.0), resample=False, fillcolor=0),         
      transforms.ToTensor(),
    ])
    self.transforms_train_orgin = transforms.Compose([
      #transforms.RandomHorizontalFlip(p=0.5),
      #transforms.RandomAffine(60, translate=(0.1,0.1), scale=(0.8,1.2), shear=(-30.0,30.0,-30.0,-30.0), resample=False, fillcolor=0),         
      transforms.ToTensor(),
    ])
    self.transforms_test_orgin = transforms.Compose([
      #transforms.RandomHorizontalFlip(p=0.5),
      #transforms.RandomAffine(60, translate=(0.1,0.1), scale=(0.8,1.2), shear=(-30.0,30.0,-30.0,-30.0), resample=False, fillcolor=0),         
      transforms.ToTensor(),
    ])
  def __getitem__(self, index):
    fn, label = self.imgs[index]       
    img = Image.open(fn).convert('L')
    if self.trainortest=="train":
      img1 = self.transform_train_rotation(img)
      img2 = self.transform_train_translate(img)
      img3 = self.transform_train_scale(img)
      img4 = self.transform_train_shear(img)
      img5 = self.transforms_train_sum(img)
      img0 = self.transforms_train_orgin(img)
      return img0, img1, img2, img3, img4, img5, label, index
    else:
      img6 = self.transforms_test_orgin(img)
      return img6, label,index
  def __len__(self):
    return len(self.imgs)
path="/home/xinyu.fan/muti_part/data_qband/"
#train_transforms = transforms.Compose([
#    transforms.RandomHorizontalFlip(p=0.5),
#    transforms.RandomAffine(20, translate=(0.1,0.1), scale=(0.8,1.2), shear=None, resample=False, fillcolor=0),
#            
#    transforms.ToTensor(),
#    transforms.RandomErasing(p=0.5,scale=(0.01,0.035),ratio=(0.3,1.0),value=0,inplace=False)
#
#    ])
#test_transform = transforms.Compose([
#    transforms.ToTensor(),
#    ])
def loaddata(path,form="train"):
    return MyDataset(path, transform=form)
#trainset = MyDataset("/home/xianpeng.yi/myUnsupervised/idea1/data_pre/test.txt", transform="train")
#trainloader = torch.utils.data.DataLoader(trainset, batch_size= 1, shuffle=True)
#for i in range (3):
#    for j, data in enumerate(trainloader, 0):             
#        inputs1,inputs2, inputs3, inputs4,inputs6, inputs7, labels, index= data
#        inputs1,inputs2, labels = inputs1.cpu(),inputs2.cpu(), labels.cpu()
#        inputs1 = inputs1.data.cpu().numpy()
#        inputs2 = inputs2.data.cpu().numpy()
#        label1 = labels.data.cpu().numpy()
#        print(index,label1)