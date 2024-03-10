
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torchvision.transforms as transforms
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from loguru import logger
import torch.nn.functional as F
import network
import loss
import itertools
from model_loader import load_model
from evaluate import mean_average_precision
from labelmodel import *
from torch.nn import Parameter
from torch.autograd import Variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'



class BaseClassificationLoss(nn.Module):
    def __init__(self):
        super(BaseClassificationLoss, self).__init__()
        self.losses = {}

    def forward(self, logits, code_logits, labels, onehot=True):
        raise NotImplementedError

def get_imbalance_mask(sigmoid_logits, labels, nclass, threshold=0.7, imbalance_scale=-1):
    if imbalance_scale == -1:
        imbalance_scale = 1 / nclass

    mask = torch.ones_like(sigmoid_logits) * imbalance_scale

    # wan to activate the output
    mask[labels == 1] = 1

    # if predicted wrong, and not the same as labels, minimize it
    correct = (sigmoid_logits >= threshold) == (labels == 1)
    mask[~correct] = 1

    multiclass_acc = correct.float().mean()

    # the rest maintain "imbalance_scale"
    return mask, multiclass_acc


class OrthoHashLoss(BaseClassificationLoss):
    def __init__(self,
                 ce=1,
                 s=8,
                 m=0.2,
                 m_type='cos',  # cos/arc
                 multiclass=False,
                 quan=0,
                 quan_type='cs',
                 multiclass_loss='label_smoothing',
                 **kwargs):
        super(OrthoHashLoss, self).__init__()
        self.ce = ce
        self.s = s
        self.m = m
        self.m_type = m_type
        self.multiclass = multiclass

        self.quan = quan
        self.quan_type = quan_type
        self.multiclass_loss = multiclass_loss
        assert multiclass_loss in ['bce', 'imbalance', 'label_smoothing']

    def compute_margin_logits(self, logits, labels):
        if self.m_type == 'cos':
            if self.multiclass:
                y_onehot = labels * self.m
                margin_logits = self.s * (logits - y_onehot)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                margin_logits = self.s * (logits - y_onehot)
        else:
            if self.multiclass:
                y_onehot = labels * self.m
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits

        return margin_logits

    def forward(self, logits, code_logits, labels, onehot=True):
        if self.multiclass:
            if not onehot:
                labels = F.one_hot(labels, logits.size(1))
            labels = labels.float()

            margin_logits = self.compute_margin_logits(logits, labels)

            if self.multiclass_loss in ['bce', 'imbalance']:
                loss_ce = F.binary_cross_entropy_with_logits(margin_logits, labels, reduction='none')
                if self.multiclass_loss == 'imbalance':
                    imbalance_mask, multiclass_acc = get_imbalance_mask(torch.sigmoid(margin_logits), labels,
                                                                        labels.size(1))
                    loss_ce = loss_ce * imbalance_mask
                    loss_ce = loss_ce.sum() / (imbalance_mask.sum() + 1e-7)
                    self.losses['multiclass_acc'] = multiclass_acc
                else:
                    loss_ce = loss_ce.mean()
            elif self.multiclass_loss in ['label_smoothing']:
                log_logits = F.log_softmax(margin_logits, dim=1)
                labels_scaled = labels / labels.sum(dim=1, keepdim=True)
                loss_ce = - (labels_scaled * log_logits).sum(dim=1)
                loss_ce = loss_ce.mean()
            else:
                raise NotImplementedError('unknown method: {self.multiclass_loss}')
        else:
            if onehot:
                labels = labels.argmax(1)
            margin_logits = self.compute_margin_logits(logits, labels)
            loss_ce = F.cross_entropy(margin_logits, labels)
            loss_ce_batch = F.cross_entropy(margin_logits, labels, reduction='none')


        if self.quan != 0:
            if self.quan_type == 'cs':
                quantization = (1. - F.cosine_similarity(code_logits, code_logits.detach().sign(), dim=1))
            elif self.quan_type == 'l1':
                quantization = torch.abs(code_logits - code_logits.detach().sign())
            else:  # l2
                quantization = torch.pow(code_logits - code_logits.detach().sign(), 2)
            quantization_batch = quantization
            quantization = quantization.mean()
        else:
            quantization_batch = torch.zeros_like(loss_ce_batch)
            quantization = torch.tensor(0.).to(logits.device)

        self.losses['ce'] = loss_ce
        self.losses['quan'] = quantization
        loss = self.ce * loss_ce + self.quan * quantization
        loss_batch = self.ce * loss_ce_batch + self.quan * quantization_batch
        return loss, loss_batch

def js_div(p_logits, q_logits):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    log_mean_output = ((p_logits + q_logits )/2).log()
    return (KLDivLoss(log_mean_output, p_logits) + KLDivLoss(log_mean_output, q_logits))/2

def MMD(x, y, kernel,device):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)


    return torch.mean(XX + YY - 2. * XY)

def train(train_s_dataloader,
          train_t_dataloader,
          query_dataloader,
          retrieval_dataloader,
          code_length,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          topk,
          num_class,
          evaluate_interval,
          tag,
          ):

    model = load_model(arch, code_length,num_class)
    # logger.info(model)
    #model = nn.DataParallel(model,device_ids=[0,1,2])
    model.to(device)
    ad_net = network.AdversarialNetwork(4096*code_length, 1024)
    ad_net.to(device)

    parameter_list = model.get_parameters() + ad_net.get_parameters()
    optimizer = optim.SGD(parameter_list, lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion_new = OrthoHashLoss()

    label_model = Label_net(num_class, code_length).to(device)
    labels = torch.zeros((num_class, num_class)).type(torch.FloatTensor).to(device)
    for i in range(num_class):
        labels[i, i] = 1
    labels = Variable(labels)
    I = Variable(torch.eye(num_class).type(torch.FloatTensor).to(device))
    one_hot = Variable(torch.ones((1, num_class)).type(torch.FloatTensor).to(device))
    optimizer_label = optim.SGD(label_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler_l = torch.optim.lr_scheduler.StepLR(optimizer_label, step_size=100, gamma=0.1, last_epoch=-1)

    # Training
    model.train()
    relu = nn.ReLU()
    alpha = 0.05
    beta = 0.01
    sigma = 20
    
    for i in range(200):
        
        scheduler_l.step()
        code = label_model(labels)

        loss1 = relu((code.mm(code.t()) - code_length * I))
        loss1 = loss1.pow(2).sum() / (num_class * num_class)
        loss_b = one_hot.mm(code).pow(2).sum() / num_class
        re = (torch.sign(code) - code).pow(2).sum() / num_class
        loss_ = loss1 + alpha * loss_b + beta * re
        optimizer_label.zero_grad()
        loss_.backward()
        optimizer_label.step()

    label_model.eval()
    code = label_model(labels)
    label_code = torch.sign(code) 
   
    for epoch in range(max_iter):

        for ((data_s, target_s, index), (data_t, target_t_gt, _)) in\
             zip(train_s_dataloader, train_t_dataloader):

            batch_size = data_t.shape[0]
            data_s = data_s.to(device)
            target_s = target_s.to(device)
            data_t = data_t.to(device)

            optimizer.zero_grad()
            logit_s, feature_s, code_s = model(data_s)

            logit_t, feature_t, code_t = model(data_t)
            temp = code_t.mm(label_code.t())
            p = F.normalize(nn.Softmax(dim=1)(sigma * temp) + 0.001)
            target_t = torch.argmax(logit_t, dim=1)
           
            # Source's cosine loss
            loss_s, _ = criterion_new(logit_s, code_s, target_s)
            mean_center = torch.zeros_like(label_code)
            num_cat = torch.ones((num_class,1)).to(device)
            filters = torch.zeros_like(num_cat)
            for i in range(batch_size):
                mean_center[target_t[i]] += code_t[i]
                num_cat[target_t[i]] += 1
                filters[target_t[i]] = 1
            mean_center *= 1/num_cat
            align_loss = torch.nn.MSELoss()(filters * mean_center, filters * label_code)

            features = torch.cat((feature_s, feature_t), dim=0)
            outputs = torch.cat((code_s, code_t), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, batch_size,device)

            maxs = torch.max(p,1)[0].unsqueeze(1)
            max_index = torch.max(p,1)[1]
            q = ((1 - maxs)/(num_class - 1)) * torch.ones_like(p)
            for i in range(temp.shape[0]):
                q[i][max_index[i]] = torch.max(p,1)[0][i]
            q = q.to(device)
            mmd = torch.exp(-1*MMD(p,q,'rbf',device))
            s_num = int(mmd*batch_size)

            confidence, indice_1 = torch.max(logit_t[:s_num] , 1)
            confidence = confidence.mean()
            w = js_div(p[:s_num],q[:s_num])
            para = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True).to(device)
            para = torch.sigmoid(para)
            loss_t = para*(1-confidence) + (1-para)*w

            total_loss = transfer_loss + loss_s + align_loss + 0.5 * loss_t
            total_loss.backward(retain_graph=True)
            optimizer.step()

        # Print log
        
        logger.info('[Epoch:{}/{}][loss:{:.4f}]'.format(epoch+1, max_iter, total_loss.item()))
        

        # Evaluate
        if (epoch % evaluate_interval == evaluate_interval-1):
            mAP = evaluate(model,
                            query_dataloader,
                            retrieval_dataloader,
                            code_length,
                            device,
                            topk,
                            )
            logger.info('[iter:{}/{}][map:{:.4f}]'.format(
                epoch+1,
                max_iter,
                mAP,
            ))

    # Evaluate and save 
    mAP = evaluate(model,
                   query_dataloader,
                   retrieval_dataloader,
                   code_length,
                   device,
                   topk,
                   )
    logger.info('Training finish, [iteration:{}][map:{:.4f}]'.format(epoch+1, mAP))


def evaluate(model, query_dataloader, retrieval_dataloader, code_length, device, topk):
    model.eval()

    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
    
    # One-hot encode targets

    onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
    onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)
   
    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )
    
    model.train()

    return mAP


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            _,_,outputs= model(data)
            code[index, :] = outputs.sign().cpu()

    return code








    


    
