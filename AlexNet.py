from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
# from deform_conv_v2 import DeformConv2d
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib
from torch.nn.parallel.scatter_gather import gather
import matplotlib.pyplot as plt

matplotlib.use('AGG')
import math

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

class AlexNet(nn.Module):
    def __init__(self, num_classes=24, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class MyDataset(Dataset):

    def __init__(self, txt, transform=None, target_transform=None, rot=0):
        super(MyDataset, self).__init__()
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
            if ro_u < 0.25:
                img = self.transform(img)
                rotation_labels = 0
            elif ro_u >= 0.25 and ro_u < 0.5:
                img = self.transform(img.rotate(90))
                rotation_labels = 1
            elif ro_u >= 0.5 and ro_u < 0.75:
                img = self.transform(img.rotate(180))
                rotation_labels = 2
            elif ro_u >= 0.75:
                img = self.transform(img.rotate(270))
                rotation_labels = 3
        else:
            img = self.transform(img)
            rotation_labels = 0
        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    def plot_loss(epoch, or_train_loss, or_test_loss, tar_test_loss, name):
        axis = np.linspace(0, epoch, epoch + 1)
        label = 'Loss'
        fig = plt.figure()
        plt.title('Loss')
        plt.plot(axis, np.array(or_train_loss), label="or_train_loss")
        plt.plot(axis, np.array(or_test_loss), label="or_test_loss")
        plt.plot(axis, np.array(tar_test_loss), label="tar_test_loss")
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('./model/' + name + '.jpg')
        plt.close(fig)


    # data_aug
    #    train_transforms = transforms.Compose([
    #            transforms.RandomHorizontalFlip(p=0.5),
    #            transforms.RandomAffine(10, translate=(0.1,0.1), scale=(0.8,1.2), shear=None, resample=False, fillcolor=0),
    #            transforms.ToTensor(),
    #            #transforms.Normalize([0.485], [0.229])
    #            ])
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485], [0.229]),
    ])
    # loss_fn = MarginLoss(0.9, 0.1, 0.5)
    # data_loader

    # XJ GK
    # or_trainset = MyDataset(txt="/home_lv/xinyu.fan/myUnsupervised/data/GK_gdata/data_224/train0.txt",
    #                         transform=train_transforms, rot=1)
    # or_testset = MyDataset(txt="/home_lv/xinyu.fan/myUnsupervised/data/GK_gdata/data_224/test0.txt",
    #                        transform=test_transform)
    # tar_testset = MyDataset(txt="/home_lv/xinyu.fan/myUnsupervised/data/XJ_data/data_224/test0.txt",
    #                         transform=test_transform)

    or_trainset = MyDataset(txt="/home_lv/xinyu.fan/myUnsupervised/data/XJ_data/data_224/train0.txt", transform=train_transforms,rot=1)
    or_testset = MyDataset(txt="/home_lv/xinyu.fan/myUnsupervised/data/XJ_data/data_224/test0.txt", transform=test_transform)
    tar_testset = MyDataset(txt="/home_lv/xinyu.fan/myUnsupervised/data/GK_gdata/data_224/test0.txt", transform=test_transform)

    # GK fish

    #    or_trainset = MyDataset(txt="/home/xianpeng.yi/muti_part/data/train1.txt", transform=train_transforms)  #fish->GK
    #    or_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data/test1.txt", transform=test_transform)
    #    tar_testset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test1.txt", transform=test_transform)

    #    or_trainset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/train1.txt", transform=train_transforms)
    #    or_testset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test1.txt", transform=test_transform)
    #    tar_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data/test1.txt", transform=test_transform)  #GK->fish

    # qband gk

    #    or_trainset = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/train1.txt", transform=train_transforms)  #qband->GK
    #    or_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/test1.txt", transform=test_transform)
    #    tar_testset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test1.txt", transform=test_transform)

    #    or_trainset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/train1.txt", transform=train_transforms)
    #    or_testset = MyDataset(txt="/home/xianpeng.yi/myUnsupervised/data/GK_gdata/data_224/test1.txt", transform=test_transform)
    #    tar_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/test1.txt", transform=test_transform)  #GK->qband

    # fish qband

    #    or_trainset = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/train1.txt", transform=train_transforms)  #qband->fish
    #    or_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/test1.txt", transform=test_transform)
    #    tar_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data/test1.txt", transform=test_transform)

    #    or_trainset = MyDataset(txt="/home/xianpeng.yi/muti_part/data/train1.txt", transform=train_transforms)
    #    or_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data/test1.txt", transform=test_transform)
    #    tar_testset = MyDataset(txt="/home/xianpeng.yi/muti_part/data_qband/test1.txt", transform=test_transform)  #fish->qband

    or_trainloader = torch.utils.data.DataLoader(or_trainset, batch_size=32, shuffle=True, num_workers=2,
                                                 pin_memory=True)
    or_testloader = torch.utils.data.DataLoader(or_testset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    tar_testloader = torch.utils.data.DataLoader(tar_testset, batch_size=64, shuffle=True, num_workers=2,
                                                 pin_memory=True)

    model = AlexNet()

    # model = nn.DataParallel(model)
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    # or_train_loss=[]
    # or_test_loss=[]
    # tar_test_loss=[]
    accplt = []
    endth = 0.0
    tmp1 = 0.1
    alpha = 0.1
    for epoch in range(100):
        correct = 0.0
        for i, data in enumerate(or_trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            or_trainloader.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                          100,
                                                                          loss)

        model.eval()
        if epoch % 1 == 0:
            # or_train_l = 0.0
            # or_test_l = 0.0
            # tar_test_l = 0.0
            with torch.no_grad():
                n = 0
                for i, data in enumerate(or_testloader, 0):
                    n += 1
                    test_data, target = data
                    test_data, target = test_data.cuda(), target.cuda()
                    test_data, target = Variable(test_data), Variable(target)
                    label = target.data.cpu().numpy()
                    output = model(test_data)
                    # loss = criterion(output, target)
                    # or_test_l+=float(loss)
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max probability
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # or_test_loss.append(or_test_l/n)
                print('\n*****or_test****Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(or_testloader.dataset),
                                                                             100. * correct / len(
                                                                                 or_testloader.dataset)))

                correct = 0.0
                n = 0
                for i, data in enumerate(or_trainloader, 0):
                    n += 1
                    test_data, target = data
                    test_data, target = test_data.cuda(), target.cuda()
                    test_data, target = Variable(test_data), Variable(target)
                    label = target.data.cpu().numpy()
                    output = model(test_data)
                    # loss = criterion(output, target)
                    # or_train_l+=float(loss)
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max probability
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # or_train_loss.append(or_train_l/n)
                print('\n*****train****Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(or_trainloader.dataset),
                                                                           100. * correct / len(
                                                                               or_trainloader.dataset)))

                correct = 0.0
                n = 0
                for i, data in enumerate(tar_testloader, 0):
                    n += 1
                    test_data, target = data
                    test_data, target = test_data.cuda(), target.cuda()
                    test_data, target = Variable(test_data), Variable(target)
                    label = target.data.cpu().numpy()
                    output = model(test_data)
                    # loss = criterion(output, target)
                    # tar_test_l+=float(loss)
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max probability
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # tar_test_loss.append(tar_test_l/n)
                print('\n*****tar_test****Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(tar_testloader.dataset),
                                                                              100. * correct / len(
                                                                                  tar_testloader.dataset)))
        # plot_loss(epoch,or_train_loss,or_test_loss,tar_test_loss,"X_2_G_resnet")
        if epoch % 10 == 0 and correct > tmp1:
            tmp1 = correct
            torch.save(model.state_dict(), './fxy/AlexNet/X_2_G_resnet.pkl')
