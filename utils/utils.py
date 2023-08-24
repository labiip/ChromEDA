import torch
import numpy as np

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def to_cpu(x):
    return x.cpu()

def to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_onehot(label, num_classes):
    identity = torch.eye(num_classes).to(label.device)
    onehot = torch.index_select(identity, 0, label)
    return onehot

def accuracy(output, target):
    """Computes the precision"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct = correct[:1].view(-1).float().sum(0, keepdim=True)
    res = correct.mul_(100.0 / batch_size)
    return res


def accuracy_for_each_class(output, target, total_vector, correct_vector):
    """Computes the precision for each class"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1)).float().cpu().squeeze()
    for i in range(batch_size):
        total_vector[target[i]] += 1
        correct_vector[torch.LongTensor([target[i]])] += correct[i]

    return total_vector, correct_vector

def recall_for_each_class(output, target, total_vector, correct_vector):
    """Computes the recall for each class"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1)).float().cpu().squeeze()
    for i in range(batch_size):
        total_vector[pred[0][i]] += 1
        correct_vector[torch.LongTensor([pred[0][i]])] += correct[i]

    return total_vector, correct_vector

def process_one_values(tensor):
    if (tensor == 1).sum() != 0:
        eps = torch.FloatTensor(tensor.size()).fill_(0)
        eps[tensor.data.cpu() == 1] = 1e-6
        tensor = tensor - eps.cuda()
    return tensor

def process_zero_values(tensor):
    if (tensor == 0).sum() != 0:
        eps = torch.FloatTensor(tensor.size()).fill_(0)
        eps[tensor.data.cpu() == 0] = 1e-6
        tensor = tensor + eps.cuda()
    return tensor


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
import torch
import torch.nn as nn
import os

class BaseSolver:
    def __init__(self, net, dataloaders,args, **kwargs):
        self.net = net
        self.dataloaders = dataloaders
        self.CELoss = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.CELoss.cuda()
        self.epoch = 0
        self.iters = 0
        self.best_prec1 = 0
        self.iters_per_epoch = None
        self.build_optimizer()
        self.init_data(self.dataloaders)

    def init_data(self, dataloaders):
        self.train_data = {key: dict() for key in dataloaders if key != 'test'}
        for key in self.train_data.keys():
            if key not in dataloaders:
                continue
            cur_dataloader = dataloaders[key]
            self.train_data[key]['loader'] = cur_dataloader
            self.train_data[key]['iterator'] = None

        if 'test' in dataloaders:
            self.test_data = dict()
            self.test_data['loader'] = dataloaders['test']

    def build_optimizer(self):
        print('Optimizer built')


    def complete_training(self):
        if self.epoch > 2000:
            return True

    def solve(self):
        print('Training Done!')

    def get_samples(self, data_name):
        assert(data_name in self.train_data)
        assert('loader' in self.train_data[data_name] and \
               'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader']
        data_iterator = self.train_data[data_name]['iterator']
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'] = data_iterator
        return sample


    def update_network(self, **kwargs):
        pass
        
import torch
import torch.nn as nn
import torch.nn.functional as F        
import ipdb


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


class CrossEntropyClassWeighted(_Loss):

    def __init__(self, size_average=True, ignore_index=-100, reduce=None, reduction='elementwise_mean'):
        super(CrossEntropyClassWeighted, self).__init__(size_average)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target, weight=None):
        return F.cross_entropy(input, target, weight, ignore_index=self.ignore_index, reduction=self.reduction)


### clone this function from: https://github.com/krumo/swd_pytorch/blob/master/swd_pytorch.py. [Unofficial]
def discrepancy_slice_wasserstein(p1, p2):
    s = p1.shape
    if s[1] > 1:
        proj = torch.randn(s[1], 128).cuda()
        proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
        p1 = torch.matmul(p1, proj)
        p2 = torch.matmul(p2, proj)
    p1 = torch.topk(p1, s[0], dim=0)[0]
    p2 = torch.topk(p2, s[0], dim=0)[0]
    dist = p1 - p2
    wdist = torch.mean(torch.mul(dist, dist))

    return wdist


class McDalNetLoss(_WeightedLoss):

    def __init__(self, weight=None, size_average=True):
        super(McDalNetLoss, self).__init__(weight, size_average)

    def forward(self, input1, input2, dis_type='L1'):

        if dis_type == 'L1':
            prob_s = F.softmax(input1, dim=1)
            prob_t = F.softmax(input2, dim=1)
            loss = torch.mean(torch.abs(prob_s - prob_t))  ### element-wise
        elif dis_type == 'CE':  ## Cross entropy
            loss = - ((F.log_softmax(input2, dim=1)).mul(F.softmax(input1, dim=1))).mean() - (
                (F.log_softmax(input1, dim=1)).mul(F.softmax(input2, dim=1))).mean()
            loss = loss * 0.5
        elif dis_type == 'KL':  ##### averaged over elements, not the real KL div (summed over elements of instance, and averaged over instance)
            ############# nn.KLDivLoss(size_average=False) Vs F.kl_div()
            loss = (F.kl_div(F.log_softmax(input1), F.softmax(input2))) + (
                F.kl_div(F.log_softmax(input2), F.softmax(input1)))
            loss = loss * 0.5
        ############# the following two distances are not evaluated in our paper, and need further investigation
        elif dis_type == 'L2':
            nClass = input1.size()[1]
            prob_s = F.softmax(input1, dim=1)
            prob_t = F.softmax(input2, dim=1)
            loss = torch.norm(prob_s - prob_t, p=2, dim=1).mean() / nClass  ### element-wise
        elif dis_type == 'Wasse':  ## distance proposed in Sliced wasserstein discrepancy for unsupervised domain adaptation,
            prob_s = F.softmax(input1, dim=1)
            prob_t = F.softmax(input2, dim=1)
            loss = discrepancy_slice_wasserstein(prob_s, prob_t)

        return loss


class TargetDiscrimLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, num_classes=31):
        super(TargetDiscrimLoss, self).__init__(weight, size_average)
        self.num_classes = num_classes

    def forward(self, input):
        batch_size = input.size(0)
        prob = F.softmax(input, dim=1)

        if (prob.data[:, self.num_classes:].sum(1) == 0).sum() != 0:  ########### in case of log(0)
            soft_weight = torch.FloatTensor(batch_size).fill_(0)
            soft_weight[prob[:, self.num_classes:].sum(1).data.cpu() == 0] = 1e-6
            soft_weight_var = soft_weight.cuda()
            loss = -((prob[:, self.num_classes:].sum(1) + soft_weight_var).log().mean())
        else:
            loss = -(prob[:, self.num_classes:].sum(1).log().mean())
        return loss

class SourceDiscrimLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, num_classes=31):
        super(SourceDiscrimLoss, self).__init__(weight, size_average)
        self.num_classes = num_classes

    def forward(self, input):
        batch_size = input.size(0)
        prob = F.softmax(input, dim=1)

        if (prob.data[:, :self.num_classes].sum(1) == 0).sum() != 0:  ########### in case of log(0)
            soft_weight = torch.FloatTensor(batch_size).fill_(0)
            soft_weight[prob[:, :self.num_classes].sum(1).data.cpu() == 0] = 1e-6
            soft_weight_var = soft_weight.cuda()
            loss = -((prob[:, :self.num_classes].sum(1) + soft_weight_var).log().mean())
        else:
            loss = -(prob[:, :self.num_classes].sum(1).log().mean())
        return loss


class ConcatenatedCELoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, num_classes=31):
        super(ConcatenatedCELoss, self).__init__(weight, size_average)
        self.num_classes = num_classes

    def forward(self, input):
        prob = F.softmax(input, dim=1)
        prob_s = prob[:, :self.num_classes]
        prob_t = prob[:, self.num_classes:]

        prob_s = process_zero_values(prob_s)
        prob_t = process_zero_values(prob_t)
        loss = - (prob_s.log().mul(prob_t)).sum(1).mean() - (prob_t.log().mul(prob_s)).sum(1).mean()
        loss = loss * 0.5
        return loss



class ConcatenatedEMLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, num_classes=31):
        super(ConcatenatedEMLoss, self).__init__(weight, size_average)
        self.num_classes = num_classes

    def forward(self, input):
        prob = F.softmax(input, dim=1)
        prob_s = prob[:, :self.num_classes]
        prob_t = prob[:, self.num_classes:]
        prob_sum = prob_s + prob_t
        prob_sum = process_zero_values(prob_sum)
        loss = - prob_sum.log().mul(prob_sum).sum(1).mean()

        return loss

class MinEntropyConsensusLoss(nn.Module):
    def __init__(self, num_classes):
        super(MinEntropyConsensusLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, x, y):
        i = torch.eye(self.num_classes).unsqueeze(0).cuda()
        x = F.log_softmax(x, dim=1)
        y = F.log_softmax(y, dim=1)
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)

        ce_x = (- 1.0 * i * x).sum(1)
        ce_y = (- 1.0 * i * y).sum(1)

        ce = 0.5 * (ce_x + ce_y).min(1)[0].mean()
        return ce
        
def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1)- F.softmax(out2)))

def Weighted_CrossEntropy(input_,labels):
    input_s = F.softmax(input_)
    entropy = -input_s * torch.log(input_s + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    weight = 1.0 + torch.exp(-entropy)
    weight = weight / torch.sum(weight).detach().item()
    #print("cross:",nn.CrossEntropyLoss(reduction='none')(input_, labels))
    return torch.mean(weight * nn.CrossEntropyLoss(reduction='none')(input_, labels))

def Entropy_div(input_):
    epsilon = 1e-5
    input_ = torch.mean(input_, 0) + epsilon
    entropy = input_ * torch.log(input_)
    entropy = torch.sum(entropy)
    return entropy 

def Entropy_condition(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1).mean()
    return entropy 

def Entropy(input_):
    return Entropy_condition(input_) + Entropy_div(input_)        