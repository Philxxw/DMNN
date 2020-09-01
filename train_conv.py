# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:48:40 2019

@author: xxw
"""
from model import *
import time
from eval import calcMAPE,calcCOSine
import torch
import torch.utils
import torch.utils.data
from torch.nn import MSELoss,CosineEmbeddingLoss
from utils import load_data
from loader import forw_Data,period_Data,TrainLog

import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(1)
log = TrainLog().getlog()

#%%
def train_single_model(lr_,net_name,origin_data,filename,data_order,Select_num,batch_size,num_epochs,l_forward,l_back,
                       n,train_count,parapath):

    '''
    #########################
    ##### load data     #####
    #########################
    '''
    split_rate = 1/2
    T0_train,T1_train,T2_train, \
    Forw_train,Pred_train,\
    T0_test,T1_test,T2_test,\
    Forw_test,Pred_test = load_data(filename =filename ,split_rate = split_rate,train_count =train_count,
                                    data_order = data_order,city_path =origin_data,store_seg ="data/seg", 
                                    week_num = 24,store_path = "data/train",select_num =Select_num,step =4,
                                    T = 24 ,n = n,l_back = l_back,m =1,l_forward =l_forward )
#    
    
    tr_f_dataset = forw_Data(Forw_train, Pred_train)
    tr_f_dataloader = torch.utils.data.DataLoader(tr_f_dataset, batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=0)
   

#%%  
    '''
    #####################
    #  model 
    #####################
    '''
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
#    elif net_name == 'MLP':    
#        net1 = MLP(input_dim = l_forward,common_dim =2).cuda()
    loss_MSE = torch.nn.MSELoss(size_average=False)
    loss_MAPE  = calcMAPE()
    
    optimizer1 = torch.optim.Adam(list(net1.parameters()), lr=lr_)

    #%% 
    #####################
    # Train model
    #####################
    t1 = time.time()
    net1.train()
    list_train = []
    num = []
    for i in range(num_epochs):
        count = 0
#        t3 = time.time()
        train_loss = 0 
        for data in (tr_f_dataloader):
            x = data[0]
            y  = data[1].view(-1,1,l_back)
            count += 1
            x,y = x.cuda(),y.cuda()
            result = []
            for j in range(l_back):  
                # insert predicted value into window data
                x0,g_truth = x.view(batch_size,-1,1).cuda(),(y[:,:,j]).cuda()
                if j ==0:
                    x_f = x0
                else:
                    x_f = torch.cat([x_f,pred_f],dim = 2)
                    
                if len(x_f[0][0]) > l_forward :
                    x_mid = x_f
                    x_f = x_mid[:,:,1:len(x_f[0][0])]
                x_f = (torch.tensor(x_f,dtype =torch.float64).cuda()).view(-1,1,l_forward)
            
        #forward 
                preds_f = net1(x_f.cuda())
                pred_f = (torch.tensor(preds_f,dtype =torch.float64).cuda()).view(batch_size,-1,1)
        #calculate loss
                g_truth =g_truth.float()

                loss = loss_MAPE(g_truth,preds_f )
                #loss = loss_MSE(preds_f,g_truth)
                optimizer1.zero_grad()     
        #backward 
                loss.backward()
        #update parameters
                optimizer1.step()
                result.append(preds_f)
            r = torch.cat([s for s in result], 1)
            train_loss =train_loss + loss_MAPE(y.view(-1,l_back),r.view(-1,l_back)) 
            
#        t4 = time.time()
        print("epoch:%i"%i,"loss:",train_loss/num_epochs)
        list_train.append(train_loss/num_epochs)
#        print("training time:",t4-t3)
        num.append(i)
    
    df = pd.DataFrame({"num":num,"train_MAPE":list_train})
    df.to_csv(parapath+"train_MAPE"+".csv",index=False,sep=',')
    torch.save(net1.state_dict(),parapath+'%s.pkl'%"net1")
    t2 = time.time()
    print("training time:",t2-t1)

