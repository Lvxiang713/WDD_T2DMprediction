# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:03:17 2022

@author: LX
"""


'''
MVT-WDD-DI model parameters:
gamma = 2**0, delta = 2**-2
tmpW: in file target_weight
op_m: in file target_point
thresholds: in file risk_thre
'''

import torch
import pandas as pd
import numpy as np
import argparse
import pickle

if __name__ == '__main__': 
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default = None)
    parser.add_argument('--delta', type=float, default = None)
    parser.add_argument('--parapath', type=str, default = None)
    parser.add_argument('--datapath', type=str, default = None)
    args = parser.parse_args()
    
    with open(args.parapath,'rb') as f:
        para =pickle.load(f)
        
    gamma = args.gamma
    delta = args.delta
    
    tmpW = para[2]
    tmpW = torch.squeeze(torch.tensor(np.array(tmpW)))
    
    op_m = para[1]
    op_m = torch.squeeze(torch.tensor(np.array(op_m)))
    
    thresholds = para[0]
    thresholds = torch.squeeze(torch.tensor(np.array(thresholds)))
    
    def val(point,op_m,tmpW,gamma,delta):
        ed = torch.nan_to_num((point-op_m)**2,nan = 0.0).double()
        num_notnan = torch.sum(~torch.isnan(ed),axis = 1)
        pro = torch.exp(-gamma*(torch.matmul(ed,tmpW)/(num_notnan**delta)))
        return pro
    
    ## example of prediction: 10 T2DM patients and 10 normal people
    data =  pd.read_csv(args.datapath,index_col=None)
    X_test = torch.tensor(np.array(data.iloc[:,:-1]))
    y_test = torch.squeeze(torch.tensor(np.array(data.iloc[:,-1])))
    risk_score = val(X_test,op_m,tmpW,gamma,delta)
    y_pred1 = (risk_score>thresholds).long()
    print('y_true:',y_test)
    print('y_pred:',y_pred1)
