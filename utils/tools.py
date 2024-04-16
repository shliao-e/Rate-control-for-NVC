import torch

from unittest.mock import patch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import numpy as np


def interpolate_log(min_val, max_val, num, decending=False):
    assert max_val >= min_val
    if max_val == min_val:
        max_val += 0.0001
    assert min_val > 0
    if decending:
        values = np.linspace(np.log(max_val), np.log(min_val), num)
    else:
        values = np.linspace(np.log(min_val), np.log(max_val), num)
    values = np.exp(values)
    return values



import sys
import os
class Logger(object):
    def __init__(self, fileN="default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
        
        
def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def R_s_gloabl(rate , c, k,n_idx,y_q_scale_enc,gop_size): 
    # w=[1.0, 1.5, 1.25, 1.5, 0.8, 1.5, 1.25, 1.5 ]
    w=[1 for i in range(gop_size)]
    length = len(w)  
    a ,b = 1/y_q_scale_enc[3] , 1/ y_q_scale_enc[0]
    for num in range(50):
        scale = (a+b)/2
        sum  = 0
        for i in range(n_idx,length):
            sum+= c[i%4]*(w[i]*scale)**k[i%4]
        if sum > rate:
            a =scale
        else:
            b =scale
        if abs(b-a) < 0.0001: 
            break
        if abs(sum-rate) < 0.0001:
            break
    return w[n_idx]*scale


def R_s_gloabl_v2(rate , c, k,n_idx): 
    # w=[1.0, 1.5, 1.25, 1.5, 0.8, 1.5, 1.25, 1.5 ]

    w=[1, 1, 1, 1, 1, 1, 1, 1 ]
    w = w[n_idx:]
    length = len(w)  
    a ,b = 0.645,1.75
    for num in range(20):
        scale = (a+b)/2
        sum  = 0
        for i in range(n_idx,length):
            sum+= c[i%4]*(w[i]*scale)**k[i%4]
        if sum > rate:
            a =scale
        elif sum < rate:
            b =scale
        else:
            break
        if abs(b-a) < 0.0001: 
            break
        if abs(sum-rate) < 0.001:
            break
    return scale

def update_params(bpp, c,k,lambda_real,num):
    for i in range(num):
        lamda_comp = c*bpp**k
        if abs(lamda_comp - lambda_real)/lambda_real < 0.01:
            break 
        c_i =c- 0.1*(torch.log(lamda_comp) - torch.log(lambda_real))*c
        k_i = k- 0.05*(torch.log(lamda_comp) - torch.log(lambda_real))*torch.log(bpp)
        c_i = clamp(c_i,1/3*c, 5/3*c)
        k_i = clamp(k_i, 5/3*k, 1/3*k)
        c = float(c_i)
        k = float(k_i)
    return c, k
                
def update_params_v2(bpp, c,k,scale,num):
    for i in range(num):
        bpp_comp = c*scale**k
        if abs(bpp_comp - bpp)/bpp < 0.01:
            break 
        c_i =c- 0.1*(np.log(bpp_comp) - np.log(bpp))*c
        k_i = k- 0.05*(np.log(bpp_comp) - np.log(bpp))*np.log(scale)
        c_i = clamp(c_i,1/3*c, 5/3*c)
        k_i = clamp(k_i, 5/3*k, 1/3*k)
        c = float(c_i)
        k = float(k_i)
    return c, k           
                
def R_lamda(c,k,rate):
    a = float(c*rate**k) 
    return a


def Rate_estimate2(TR, coded_R, n_index,w):
    w_list = w[n_index:]
    rate = (TR - coded_R)*w_list[0]/ sum(w_list)

    return rate
                


