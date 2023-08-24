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

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:   # eval model lose this layer
            return x, aux2, aux1
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


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 淇濊瘉杈撳嚭澶у皬绛変簬杈撳叆澶у皬
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 淇濊瘉杈撳嚭澶у皬绛変簬杈撳叆澶у皬
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


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
        plt.savefig('./fxy/googlenet/' + name + '.jpg')
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
    or_trainset = MyDataset(txt="/home_lv/xinyu.fan/myUnsupervised/data/GK_gdata/data_224/train0.txt",
                            transform=train_transforms, rot=1)
    or_testset = MyDataset(txt="/home_lv/xinyu.fan/myUnsupervised/data/GK_gdata/data_224/test0.txt",
                           transform=test_transform)
    tar_testset = MyDataset(txt="/home_lv/xinyu.fan/myUnsupervised/data/XJ_data/data_224/test0.txt",
                            transform=test_transform)

    # or_trainset = MyDataset(txt="/home_lv/xinyu.Fan/myUnsupervised/data/XJ_data/data_224/train0.txt", transform=train_transforms,rot=1)
    # or_testset = MyDataset(txt="/home_lv/xinyu.Fan/myUnsupervised/data/XJ_data/data_224/test0.txt", transform=test_transform)
    # tar_testset = MyDataset(txt="/home_lv/xinyu.Fan/myUnsupervised/data/GK_gdata/data_224/test0.txt", transform=test_transform)

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


    net = GoogLeNet(num_classes=24, aux_logits=True, init_weights=True)


    # model = nn.DataParallel(model)
    model = net.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # or_train_loss = []
    # or_test_loss = []
    # tar_test_loss = []
    accplt = []
    endth = 0.0
    tmp1 = 0.1
    alpha = 0.1
    for epoch in range(100):
        net.train()
        running_loss = 0.0
        correct = 0.0
        for i, data in enumerate(or_trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = model(inputs)
            loss0 = criterion(logits, labels)
            loss1 = criterion(aux_logits1, labels)
            loss2 = criterion(aux_logits2, labels)
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3

            loss.backward()
            optimizer.step()
            running_loss += loss.item()


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
                    # loss0 = criterion(logits, labels)
                    # loss1 = criterion(aux_logits1, labels)
                    # loss2 = criterion(aux_logits2, labels)
                    # loss = loss0 + loss1 * 0.3 + loss2 * 0.3
                    # or_test_l += float(loss)
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max probability
                    # correct += torch.eq(pred, target.cpu()).sum().item()
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # or_test_loss.append(or_test_l / n)
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
                    # loss0 = criterion(logits, labels)
                    # loss1 = criterion(aux_logits1, labels)
                    # loss2 = criterion(aux_logits2, labels)
                    # loss = loss0 + loss1 * 0.3 + loss2 * 0.3
                    # or_train_l += float(loss)
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max probability
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # or_train_loss.append(or_train_l / n)
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
                    # loss0 = criterion(logits, labels)
                    # loss1 = criterion(aux_logits1, labels)
                    # loss2 = criterion(aux_logits2, labels)
                    # loss = loss0 + loss1 * 0.3 + loss2 * 0.3
                    # tar_test_l += float(loss)
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max probability
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # tar_test_loss.append(tar_test_l / n)
                print('\n*****tar_test****Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(tar_testloader.dataset),
                                                                              100. * correct / len(
                                                                                  tar_testloader.dataset)))
        # plot_loss(epoch, or_train_loss, or_test_loss, tar_test_loss, "G_2_X_googlenet")
        if epoch % 10 == 0 and correct > tmp1:
            tmp1 = correct
            torch.save(model.state_dict(), './fxy/googlenet/G_2_X_googlenet.pkl')
