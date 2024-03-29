# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 10:53:43 2022

@author: LX
"""

#MVT-WDD-BF model
import torch
import numpy as np
import torch.optim as optim
import time
import pandas as pd
import torch.nn as nn
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score
import random
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class myCMP(nn.Module):
    def __init__(self,input_features,gamma,lam,mean_p,var_p):##input_fetures is the dimension of features
        super(myCMP, self).__init__()
        self.input_features = input_features
        self.x = nn.Parameter((torch.tensor(mean_p)+2*torch.tensor(var_p)*(torch.rand(input_features)*2-1)).to(device)) ##generate init point by uniform generator
        self.w = nn.Parameter((torch.full((input_features,), 1/input_features)).double().to(device))##generate weight by features on average
        self.loss = None
        self.relu =  nn.ReLU()
        self.gamma = torch.tensor(gamma).to(device)
        self.tmpW = (self.relu(self.w) + 0.01)/torch.sum((self.relu(self.w) + 0.01))
        self.lam = lam
    
    def norm1(self,x):
        mean = torch.mean(x)
        std = torch.std(x)
        x_new = (x-mean)/std
        return x_new
    
    def cmp(self, bagP , bagN, groupIndexP, groupIndexN):
        batchSizeP = bagP.size(0)
        batchSizeN = bagN.size(0)
        feaLen = bagP.size(1)
        bagNumP = len(torch.unique(groupIndexP))
        bagNumN = len(torch.unique(groupIndexN))
        self.tmpW = self.relu(self.w) + 0.01
        self.tmpW = self.tmpW/torch.sum(self.tmpW)
        bagP_nan_rate = bagP.isnan().sum(axis=0)/bagP.shape[0]
        bagN_nan_rate = bagN.isnan().sum(axis=0)/bagN.shape[0]
        boolN =  (bagP_nan_rate> bagN_nan_rate)
    
        df_bagN =  pd.DataFrame(bagN.cpu())
        notnan_df_bagN = ~np.isnan(df_bagN)
        for idx,i in enumerate(boolN):
            if i == True:
                N_nan_rate = bagN_nan_rate[idx]
                P_nan_rate = bagP_nan_rate[idx]
                Dif_nan_rate = P_nan_rate-N_nan_rate
                Dif_number = (Dif_nan_rate*bagN.shape[0]).floor()
                not_nan_idx = notnan_df_bagN[notnan_df_bagN.iloc[:,idx]==True].index.tolist()
                random_index = random.sample( not_nan_idx, int(Dif_number))
                bagN[random_index,idx] = np.nan

        boolP =  ~(bagP_nan_rate> bagN_nan_rate)
        df_bagP =  pd.DataFrame(bagP.cpu())
        notnan_df_bagP = ~np.isnan(df_bagP)
        for idx,i in enumerate(boolP):
            if i == True:
                N_nan_rate = bagN_nan_rate[idx]
                P_nan_rate = bagP_nan_rate[idx]
                Dif_nan_rate = N_nan_rate-P_nan_rate
                Dif_number = (Dif_nan_rate*bagP.shape[0]).floor()
                not_nan_idx = notnan_df_bagP[notnan_df_bagP.iloc[:,idx]==True].index.tolist()
                random_index = random.sample( not_nan_idx, int(Dif_number))
                bagP[random_index,idx] = np.nan    

        is_nanP = bagP.isnan()
        bagP = bagP.clone().masked_fill(is_nanP,0)
        expP = torch.exp(-self.gamma*self.tmpW*torch.abs(bagP-self.x.view([1,feaLen])))
        expP = expP.clone().masked_fill(is_nanP,0)
        tmpScore = torch.exp(-(torch.sum(expP**self.lam,axis = 1))) #Nx1
        tmpScore = tmpScore.reshape(-1,1)
        groupIndexPOH = torch.zeros([batchSizeP,bagNumP]).to(device)
        groupIndexPOH.scatter_(1,groupIndexP.view([batchSizeP,1]).long(),1)
        tmpScore = torch.einsum('ni,np->np',tmpScore,groupIndexPOH) #N x numBagP
        tmpMask = groupIndexPOH == 0
        tmpScore = tmpScore.masked_fill(tmpMask,1)
        cp = torch.log(1 - torch.prod(tmpScore,axis=0)).sum()
        
        is_nanN = bagN.isnan()
        bagN = bagN.clone().masked_fill(is_nanN,0)
        expN = torch.exp(-self.gamma*self.tmpW*torch.abs(bagN-self.x.view([1,feaLen])))
        expN = expN.clone().masked_fill(is_nanN,0)
        tmpScore = torch.exp(-(torch.sum(expN**self.lam,axis = 1))) #Nx1    
        tmpScore = tmpScore.reshape(-1,1) 
        
        groupIndexNOH = torch.zeros([batchSizeN,bagNumN]).to(device)
        groupIndexNOH.scatter_(1,groupIndexN.view([batchSizeN,1]).long(),1)
        tmpScore = torch.einsum('ni,np->np',tmpScore,groupIndexNOH) #N x numBagP
        tmpMask = groupIndexNOH == 0
        tmpScore = tmpScore.masked_fill(tmpMask,1)
        cn = torch.log(torch.prod(tmpScore,axis=0)).sum()

        cp = cp/(bagNumP**1.4)
        cn = cn/(bagNumN**1.4)
        loss2 = -(cp + cn)
        self.loss = loss2
        return loss2
     
    def forward(self, bagP , bagN, groupIndexP, groupIndexN ):
        return self.cmp(bagP , bagN, groupIndexP, groupIndexN) 
    
    
    
    
def initPoint(bagP , bagN,groupIndexP, groupIndexN,gamma,lam,mean_p,var_p):
    myCMPlayer = myCMP(bagP.shape[-1],gamma,lam,mean_p,var_p)
    loss = myCMPlayer(bagP , bagN,groupIndexP, groupIndexN)
    while loss ==  torch.tensor(float('inf')):
        myCMPlayer = myCMP(bagP.shape[-1],gamma,lam,mean_p,var_p)
        loss = myCMPlayer(bagP , bagN, groupIndexP, groupIndexN)
        print(loss)
    return myCMPlayer


def run(cmpInit, bagP , bagN,groupIndexP,groupIndexN):
    stopThres = 1e-6
    lossOld = None
    round_1 = 30 
    round_2 = 5 ##update weight and delta 5 times per 30 epochs
    loss_list_1 = []
    for epoch in range(10000):
        if epoch%50 == 0 and epoch!=0 :
            print('epoch: %d' %epoch)
            print('loss:' ,loss.item())
        if epoch%round_1 < round_2:
            reCalThres = True 
            cmpInit.w.requires_grad = True
            if epoch%round_1 == 0: 
                optimizer = optim.Adam(cmpInit.parameters(), lr=0.01)
        else:
            reCalThres = False
            cmpInit.w.requires_grad = False
            if epoch%round_1 == round_2:
                optimizer = optim.Adam(cmpInit.parameters(), lr=0.1)
        cmpInit.train()
        optimizer.zero_grad()
        loss = cmpInit(bagP , bagN, groupIndexP,groupIndexN)
        loss_list_1.append(loss.item())
        if lossOld is None:
            lossOld = loss.item()
        else:
            if abs(loss.item() - lossOld) < stopThres:
                break
            lossOld = loss.item()
        loss.backward()
        optimizer.step()
    return cmpInit.x.detach().clone().data, cmpInit.tmpW.detach().clone().data ,loss, loss_list_1


def fit(X_train,y_train,gamma,lam):
    start_time = time.time()
    bagP = []
    bagN = []
    for idx,tmpArr in enumerate(X_train):
        if y_train[idx] == 1:
            bagP.append(np.array(tmpArr))
        else:
            bagN.append(np.array(tmpArr))
            
    bagP = np.array(bagP)
    bagN = np.array(bagN)
    
    groupIndexP = []
    for idx,i in enumerate(bagP):
        len_i = len(i)
        tmp_list = [idx]*len_i
        groupIndexP += tmp_list     
    groupIndexP = torch.tensor(groupIndexP).to(device)
    
    groupIndexN = []
    for idx,i in enumerate(bagN):
        len_i = len(i)
        tmp_list = [idx]*len_i
        groupIndexN += tmp_list  
    groupIndexN = torch.tensor(groupIndexN).to(device)   
    
    bagP = torch.tensor(bagP.reshape(-1,bagP.shape[-1])).to(device)
    bagN = torch.tensor(bagN.reshape(-1,bagN.shape[-1])).to(device)
    
    dt_p = pd.DataFrame(bagP.detach().clone().cpu().numpy())
    mean_p = []
    var_p = []
    for i in dt_p:
        dti_sort = dt_p[i].dropna().sort_values()
        Percentile = np.percentile(dti_sort,[0,25,50,75,100])
        IQR = Percentile[3] - Percentile[1]
        UpLimit = Percentile[3]+IQR*1.5
        DownLimit = Percentile[1]-IQR*1.5
        tmp_var = dti_sort[(dti_sort>=DownLimit) & (dti_sort<=UpLimit)]
        mean_p.append(np.mean(tmp_var))
        var_p.append(np.var(tmp_var))
    
    op_list = []
    w_list = []
    loss_list = []
    for i in range(1):
        ok_num = 1
        while ok_num:
            myCMP_init = initPoint(bagP , bagN,groupIndexP,groupIndexN,gamma,lam,mean_p,var_p)
            try:
                op_m,w_m,loss_m,loss_list1 = run(myCMP_init, bagP, bagN,groupIndexP,groupIndexN)
                # plt.plot([i for i in range(len(loss_list1))],loss_list1)
                op_list.append(op_m)
                w_list.append(w_m)
                loss_list.append(loss_m)  
            except RuntimeError:
                print("Runtime try again!")
                ok_num = 0
            else:
                ok_num = 0
    # plt.show()
    # print('')
    # print('')
    # print('')
    cost_time = time.time() - start_time
    return op_list,w_list

def val(point,op_m,tmpW,gamma,lam):
    is_nanP = point.isnan()
    point = point.clone().masked_fill(is_nanP,0)
    expP = torch.exp(-gamma*tmpW*torch.abs(point-op_m))
    expP = expP.clone().masked_fill(is_nanP,0)
    pro = torch.sum(expP**lam,axis = 1)
    return pro

def CV_score(X_train,y_train,X_test,y_test,gamma,lam,op_list,w_list,device = 'cuda:0',dtype=torch.float32):
    score_list = []
    score_list1 = []    
    auc_list = []
    y_pred_list = []
    
    
    for i in range(1):
        op_m = op_list[i]
        tmpW = w_list[i]
        y_train_pred=[]
        y_pred = []
        glu_score = []
        
        for idx,i in enumerate(X_train):
            tmp = torch.tensor(i,dtype=dtype)
            y_train_pred.append(max(val(tmp.to(device),op_m,tmpW,gamma,lam)).cpu().detach().numpy())


        for idx,i in enumerate(X_test):
            tmp = torch.tensor(i,dtype=dtype)
            y_pred.append(max(val(tmp.to(device),op_m,tmpW,gamma,lam)).cpu().detach().numpy())
        
        
    y_train_pred = np.array(y_train_pred)
    
       
        
    y_pred = np.array(y_pred)
    y_pred_list.append(y_pred)
    
    
    
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_train_pred, pos_label=1,drop_intermediate=False)


    arg = tpr-fpr
    Youden_index = np.argmax(arg) 
    
    optimal_threshold = thresholds[Youden_index]
    
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1,drop_intermediate=False)
    Auc = metrics.auc(fpr, tpr)
    auc_list.append(Auc)
    
    
    y_pred1 = (y_pred>optimal_threshold)
    Acc = accuracy_score(y_test, y_pred1)
    Pre = precision_score(y_test, y_pred1)
    Recall = recall_score(y_test, y_pred1)
    F1_score = f1_score(y_test, y_pred1)

    
    score_list.append(('Auc:%f'%Auc, 'Acc:%f'%Acc, 'Pre:%f'%Pre, 'Recall:%f'%Recall, 'F1_score:%f'%F1_score))
    score_list1.append((Auc,Acc,Pre,Recall,F1_score))
    return score_list,score_list1,fpr,tpr,optimal_threshold

        
