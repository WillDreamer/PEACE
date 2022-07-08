import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def adv_loss(input_list, ad_net, batch_size, device, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1]
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1))) 

    if batch_size == ad_out.size(0) // 2 :
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    else:
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * (ad_out.size(0) - batch_size))).float().to(device)
    return nn.BCELoss()(ad_out, dc_target) 
