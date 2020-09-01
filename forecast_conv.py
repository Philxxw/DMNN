# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:42:35 2019

@author: xxw
"""

import time
from eval import calcMAPE,calcCOSine,calcMSE
import torch
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt 
from utils import load_data
from model import *
from loader import *
import pandas as pd
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%

def test_single_model(net_name,origin_data,filename,data_order,Select_num,batch_size,l_forward,l_back,
                       n,train_count,parapath,forepath):

    #########################
    ##### load data    
    #########################
    
    T0_train,T1_train,T2_train, \
    Forw_train,Pred_train,\
    T0_test,T1_test,T2_test,\
    Forw_test,Pred_test = load_data(filename =filename ,split_rate = 1/2,train_count =train_count,
                                    data_order = data_order,city_path =origin_data,store_seg ="data/seg", 
                                    week_num = 24,store_path = "data/train",select_num =Select_num,step =4,
                                    T = 24 ,n = n,l_back = l_back,m =1,l_forward =l_forward )


    te_f_dataset = forw_Data(Forw_test, Pred_test)
    te_f_dataloader = torch.utils.data.DataLoader(te_f_dataset, batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=0)
    



#%%  
#####################
#  model
#####################

    if net_name == 'short_LSTM':
        net1 = short_LSTM(input_dim =l_forward, hidden_dim=2, num_layers =1,batch_size= batch_size).cuda()
    elif net_name == 'BiLSTM':
        net1 = BiLSTM(input_dim =l_forward, hidden_dim=2, num_layers =1,batch_size= batch_size).cuda() 
    elif net_name == 'conv_LSTM':
        net1 = conv_LSTM(input_dim =20, hidden_dim=2, num_layers =1,batch_size= batch_size).cuda()
    elif net_name == 'conv_BiLSTM':
        net1 = conv_BiLSTM(input_dim =20, hidden_dim=2, num_layers =1,batch_size= batch_size).cuda()
#    elif net_name == 'LSTM_with_Attention':
#        net1 = LSTM_with_Attention(input_dim =l_forward, hidden_dim=2, num_layers =1,batch_size= batch_size).cuda()
#    elif net_name == 'conv_LSTM_with_Attention':
#        net1 = conv_LSTM_with_Attention(input_dim =20, hidden_dim=2, num_layers =1,batch_size= batch_size).cuda()
#    
    loss_MAPE  = calcMAPE()
    net1.load_state_dict(torch.load(parapath+'%s.pkl'%"net1",map_location=torch.device('cuda:0')))
    net1.eval()
#%%
#####################
# test model
#####################
    t3 = time.time()
    count_fore = 0
    result = []
    num = []
    for data in te_f_dataloader:
        x = data[0].view(-1,1,l_forward)
        y  = data[1].view(-1,1,l_back)
        count_fore += 1
        x,y = x.to(device),y.to(device)
        result1 = []
        for j in range(l_back):  
            #insert predicted value into window data
            x0,g_truth = x.view(batch_size,-1,1).cuda(),(y[:,:,j]).cuda()
            if j == 0:
                x_f = x0
            else:
                x_f = torch.cat([x_f,pred_f],dim = 2)
                    
            if len(x_f[0][0]) > l_forward :
                x_mid = x_f
                x_f = x_mid[:,:,1:len(x_f[0][0])]
            x_f = (torch.tensor(x_f,dtype =torch.float64).cuda()).view(-1,1,l_forward)
            
            preds_f = net1(x_f)
            pred_f = (torch.tensor(preds_f,dtype =torch.float64).cuda()).view(batch_size,-1,1)
            #build predicted sequence
            result1.append(preds_f)

        r = torch.cat([s for s in result1], 1)
        #calculate MAPE of predicting setting
        loss = loss_MAPE(y.view(-1,l_back),r.view(-1,l_back)) 

#        plt.title("%f"%loss+"%")
#        plt.plot((pd.Series(np.array(list(r.view(l_back,-1))).T)), label="Pred")
#        plt.plot(pd.Series(np.array(list(y.view(l_back,-1))).T), label="Data")
#        plt.legend()
#        plt.savefig(forepath+"/%i.png"%count_fore)
#        plt.show()
        #print("Hybrid:","data_order:%i"%count_fore,loss,"%")
        num.append(count_fore)
        result.append(loss)

    num.append("average")
    result.append(torch.mean(torch.tensor(result)))
    loss_mean = torch.mean(torch.tensor(result))
    t4 = time.time()
    print("  mean loss   |test time (s)")
    print("%s|%s"%(loss_mean,t4-t3))
    dataframe = pd.DataFrame({'num':num,'loss':result})
    dataframe.to_csv(forepath+"/forecast"+".csv",index=False,sep=',')
    # return average MAPE
    return loss_mean