# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:10:58 2022

@author: LX
"""


'''
WDD-KNN model parameters:
gamma = 2**-3
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
    parser.add_argument('--parapath', type=str, default = None)
    parser.add_argument('--datapath', type=str, default = None)
    args = parser.parse_args()
     
    with open(args.parapath,'rb') as f:
        para =pickle.load(f)

    gamma = args.gamma
    tmpW = para[2]
    tmpW = torch.squeeze(torch.tensor(np.array(tmpW)))
    
    op_m = para[1]
    op_m = torch.squeeze(torch.tensor(np.array(op_m)))
    
    thresholds = para[0]
    thresholds = torch.squeeze(torch.tensor(np.array(thresholds)))
    
    def val(point,op_m,tmpW,gamma):
        pro = torch.exp(-gamma*torch.matmul((point-op_m)**2, tmpW))
        return pro
    
    ## example of prediction: 10 T2DM patients and 10 normal people
    data =  pd.read_csv(args.datapath,index_col=None)
    X_test = torch.tensor(np.array(data.iloc[:,:-1]))
    y_test = torch.squeeze(torch.tensor(np.array(data.iloc[:,-1])))
    risk_score = val(X_test,op_m,tmpW,gamma)
    y_pred1 = (risk_score>thresholds).long()
    print('y_true:',y_test)
    print('y_pred:',y_pred1)
