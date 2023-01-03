# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:15:10 2022

@author: LX
"""

#MVT-WDD-BF train
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from BFmodel import *
from sklearn import preprocessing
import argparse
import pickle
import os

if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default = None)
    parser.add_argument('--lam', type=float, default = None)
    parser.add_argument('--data_path', type=str, default = None)
    parser.add_argument('--out_path', type=str, default = None)
    args = parser.parse_args()
    skf = StratifiedKFold(n_splits=10, shuffle = True,random_state =48)
    def normali(df_train,df_test,dealoutlier = True):
        df_train = pd.DataFrame(df_train)
        df_test = pd.DataFrame(df_test)
        mean_tmp = []
        std_tmp = []
        for i in df_train:
            if dealoutlier == True:
                dti_sort = df_train[i].dropna().sort_values()
                Percentile = np.percentile(dti_sort,[0,25,50,75,100])
                IQR = Percentile[3] - Percentile[1]
                UpLimit = Percentile[3]+IQR*1.5
                DownLimit = Percentile[1]-IQR*1.5
                tmp_var = dti_sort[(dti_sort>=DownLimit) & (dti_sort<=UpLimit)]
                df_train[i][df_train[i]>UpLimit] = np.nan  
                df_train[i][df_train[i]<DownLimit] = np.nan
                df_test[i][df_test[i]>UpLimit] = np.nan
                df_test[i][df_test[i]<DownLimit]  = np.nan
                mean_tmp.append(np.mean(tmp_var))
                std_tmp.append(np.std(tmp_var)) 
            else:
                mean_tmp.append(np.mean(df_train[i]))
                std_tmp.append(np.std(df_train[i]))
        df_train_norm = (df_train- np.array(mean_tmp))/std_tmp
        df_test_norm = (df_test - np.array(mean_tmp))/std_tmp
        return df_train_norm,df_test_norm
    out_path = args.out_path
    data = pd.read_csv(args.data_path)
    X = np.array(data.iloc[:,:-1])
    y = np.array(data.iloc[:,-1])
    '''
    X is a 2D array(Number of samples*number of features),for example:
        array([[ 3.26, 32.3 ,   nan, ..., 13.8 , 47.5 ,  4.36],
               [ 3.65, 21.1 ,   nan, ...,  7.  , 35.8 ,  5.59]])
    
    y is a 1D array(the label of X),for example:
        array([1,0])
    
    '''
        

    score_list = []
    score_list1 = []
    fpr_list = []
    tpr_list = []
    op_list1 = []
    w_list1 = []
    optimize_thre_list = []
    gamma= args.gamma
    lam= args.lam
    idx=0
    for train_index, test_index in skf.split(X, y):
        idx+=1
        print('fold:',idx)
        X_train, X_test = X[train_index], X[test_index]
        X_train, X_test = normali(X_train, X_test,dealoutlier = True)
        train_use_list = X_train[(np.sum(X_train.isna(),axis=1)) != X_train.shape[1]].index.tolist()
        test_use_list = X_test[(np.sum(X_test.isna(),axis=1)) != X_test.shape[1]].index.tolist()
        X_train = X_train.iloc[train_use_list,:]
        X_test = X_test.iloc[test_use_list,:]
        X_train = np.expand_dims(X_train,axis=1)
        X_test = np.expand_dims(X_test,axis=1)
        y_train, y_test = y[train_index], y[test_index]
        y_train = y_train[train_use_list]
        y_test = y_test[test_use_list]
        op_list,w_list = fit(X_train,y_train,gamma,lam)
        op_list1.append(op_list)
        w_list1.append(w_list)
        if len(op_list)!=0:
            cv_final = CV_score(X_train,y_train,X_test,y_test,gamma,lam,op_list,w_list)
            score_list = score_list+ cv_final[0]
            score_list1 = score_list1+ cv_final[1]
            fpr_list.append(cv_final[2])
            tpr_list.append(cv_final[3])  
            optimize_thre_list.append(cv_final[4]) 
            print(cv_final[0])
            
    with open(os.path.join(out_path,'score.txt'),'a') as f:
        f.write('\n gamma=%f lam=%f: '%(gamma,lam))
        if len(score_list1)!=0:
            f.write('\n')
            for i in np.average(np.array(score_list1),axis=0):
                f.write(str(i) + ' ')  
            f.write('\n')
            for j in np.std(np.array(score_list1),axis=0):
                f.write(str(j) + ' ')  
     
    ##all data      
    score_list = []
    score_list1 = []
    fpr_list = []
    tpr_list = []
    op_list1 = []
    w_list1 = []
    optimize_thre_list = []
    X_train, X_test = X,X
    X_train, X_test = normali(X_train, X_test,dealoutlier = True)
    train_use_list = X_train[(np.sum(X_train.isna(),axis=1)) != X_train.shape[1]].index.tolist()
    test_use_list = X_test[(np.sum(X_test.isna(),axis=1)) != X_test.shape[1]].index.tolist()
    X_train = X_train.iloc[train_use_list,:]
    X_test = X_test.iloc[test_use_list,:]
    X_train = np.expand_dims(X_train,axis=1)
    X_test = np.expand_dims(X_test,axis=1)
    y_train, y_test = y,y
    y_train = y_train[train_use_list]
    y_test = y_test[test_use_list]
    op_list,w_list = fit(X_train,y_train,gamma,lam)
    op_list1.append(op_list)
    w_list1.append(w_list)
    if len(op_list)!=0:
        cv_final = CV_score(X_train,y_train,X_test,y_test,gamma,lam,op_list,w_list)
        score_list = score_list+ cv_final[0]
        score_list1 = score_list1+ cv_final[1]
        fpr_list.append(cv_final[2])
        tpr_list.append(cv_final[3])  
        optimize_thre_list.append(cv_final[4]) 
                
   
    para_list = []
    df_optimize_thre_list= pd.DataFrame(np.array(optimize_thre_list))
    para_list.append(df_optimize_thre_list)
    df_op =  pd.DataFrame(np.array([i[0].cpu().numpy() for i in op_list1]))
    para_list.append(df_op)
    df_w =  pd.DataFrame(np.array([i[0].cpu().numpy() for i in w_list1]))
    para_list.append(df_w)
    
    
    with open(os.path.join(out_path,'BFpara.pk'), 'wb') as f:
        pickle.dump(para_list, f)