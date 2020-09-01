# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:58:46 2019

@author: xxw
"""

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F

#%%
#####################
# Conv LSTM
#####################
class conv_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim,num_layers,
                 batch_size, output_dim=1 ):
        super(conv_LSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        # Define the conv layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.fc = nn.Linear(20, self.input_dim)
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,dropout = 0.4)

        # Define the output layer
        self.linear1 = nn.Linear(self.hidden_dim*32,self.hidden_dim*4)
        self.linear2 = nn.Linear(self.hidden_dim*4,output_dim)
        
    def forward(self, x):
        
        # Forward pass through conv-LSTM layer
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        lstm_out, self.hidden = self.lstm(x.view(1,-1,len(x[0][0])))
        #out  = self.linear1(lstm_out[-1].view(1, -1))
        out  = self.linear1(lstm_out[-1].view(-1,self.hidden_dim*32))
        
        return self.linear2(out.view(-1,self.hidden_dim*4))
#%%
#####################
# Conv BiLSTM
#####################
class conv_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim,num_layers,
                 batch_size, output_dim=1 ):
        super(conv_BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        # Define the conv layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.fc = nn.Linear(20, self.input_dim)
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,bidirectional=True,dropout = 0.4)

        # Define the output layer
        self.linear1 = nn.Linear(self.hidden_dim*64,self.hidden_dim*4)
        self.linear2 = nn.Linear(self.hidden_dim*4,output_dim)
        
    def forward(self, x):
        
        # Forward pass through conv-LSTM layer
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        lstm_out, self.hidden = self.lstm(x.view(1,-1,len(x[0][0])))
        #out  = self.linear1(lstm_out[-1].view(1, -1))
#        print(lstm_out.shape)
        out  = self.linear1(lstm_out[-1].view(-1,self.hidden_dim*64))
        
        return self.linear2(out.view(-1,self.hidden_dim*4))
#%%
#####################
# Conv LSTM with Attention
#####################
class conv_LSTM_with_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim,num_layers,
                 batch_size, output_dim=1,use_bidirectional =False):
        super(conv_LSTM_with_Attention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.use_bidirectional = use_bidirectional
        # Define the conv layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.fc = nn.Linear(20, self.input_dim)
        # Define the LSTM layer
        if self.use_bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,bidirectional=self.use_bidirectional,dropout = 0.4)

        # Define the output layer
        self.linear1 = nn.Linear(self.hidden_dim*64*self.num_directions,self.hidden_dim*4)
        self.linear2 = nn.Linear(self.hidden_dim*4,output_dim)
        
    def forward(self, x):
        
        # Forward pass through conv-LSTM layer
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        lstm_out, self.hidden = self.lstm(x.view(1,-1,len(x[0][0])))
        #out  = self.linear1(lstm_out[-1].view(1, -1))
#        print(lstm_out.shape)
        out  = self.linear1(lstm_out[-1].view(-1,self.hidden_dim*64))
        
        return self.linear2(out.view(-1,self.hidden_dim*4))
#%%
#####################
# LSTM
#####################   
class short_LSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim,num_layers,
                 batch_size, output_dim=1 ):
        super(short_LSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,dropout = 0.4)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
       
    def forward(self, x):
        x = x.float()
        lstm_out, self.hidden = self.lstm(x.view(1,-1,len(x[0][0])))
        
        return self.linear(lstm_out[-1].view(self.batch_size, -1))
#%%
#####################
# BiLSTM
#####################   
class BiLSTM(nn.Module):

     def __init__(self, input_dim, hidden_dim,num_layers,
                 batch_size, output_dim=1 ):
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,bidirectional=True,dropout = 0.4)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim*2, output_dim)
       
     def forward(self, x):
        x = x.float()
        lstm_out, self.hidden = self.lstm(x.view(1,-1,len(x[0][0])))

        return self.linear(lstm_out[-1].view(self.batch_size, -1))
#%%
#####################
# LSTM with Attention
#####################   
class LSTM_with_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,batch_size, output_dim = 1, 
                 use_bidirectional=False):
        super(LSTM_with_Attention,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.use_bidirectional = use_bidirectional
        # Define the LSTM layer
        if self.use_bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,bidirectional=self.use_bidirectional,dropout = 0.4)

        # Define the output layer
        self.linear = nn.Linear((self.hidden_dim*2),output_dim)
    
    def attention_net(self, lstm_output, final_state):

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output,soft_attn_weights)

        return new_hidden_state
    
    def attention(self, lstm_output, final_state):
 
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.permute(0,2,1)  
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, x):
        x = x.float()
        lstm_out, hidden = self.lstm(x.view(1,-1,len(x[0][0])))
        # attn_output = self.attention_net(output, hidden)
        attn_output = self.attention(lstm_out, hidden)
        return self.linear(attn_output.view(self.batch_size,-1))
#%%
#####################
# Period series model LSTM
#####################
class period_net(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers,batch_size, output_dim=1):
        super(period_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm0 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,dropout = 0.4)
        self.lstm1 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,dropout = 0.4)
        self.lstm2 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,dropout = 0.4)
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.coefficient  = torch.nn.Parameter(torch.Tensor(1))
        self.P_weight_0 = torch.nn.Parameter(torch.Tensor(1))
        self.P_weight_1 = torch.nn.Parameter(torch.Tensor(1))
        self.P_weight_2 = torch.nn.Parameter(torch.Tensor(1))
    def forward(self, T0,T1,T2):

        T0 = T0.float()
        T1 = T1.float()
        T2 = T2.float()

        lstm_out_0, self.hidden = self.lstm0(T0.view(1,-1,len(T0[0])))
        lstm_out_1, self.hidden = self.lstm1(T1.view(1,-1,len(T1[0])))
        lstm_out_2, self.hidden = self.lstm2(T2.view(1,-1,len(T2[0]))) 

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        a = self.P_weight_0
        b = self.P_weight_1
        c = self.P_weight_2           
        y_pred_0 = self.linear(lstm_out_0[-1].view(-1,self.hidden_dim))
        y_pred_1 = self.linear(lstm_out_1[-1].view(-1,self.hidden_dim))
        y_pred_2 = self.linear(lstm_out_2[-1].view(-1,self.hidden_dim))

        return ((a*y_pred_0.view(-1)+b*y_pred_1.view(-1)+c*y_pred_2.view(-1))*self.coefficient)
        
#%%
#####################
# Fully coonected linear model
#####################  
class metrix(nn.Module):
   def __init__(self, in_dim, n_hidden_1,n_hidden_2, out_dim):
        super(metrix, self).__init__(),

        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
#        self.relu = nn.ReLU()
   def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        return self.layer3(y)

#%%
#####################
#   MLP
#####################           
class MLP(nn.Module):
   def __init__(self, input_dim, common_dim):
        super(MLP, self).__init__(),

        self.layer1 = nn.Linear(input_dim, input_dim // 2)
        
        self.layer2 = nn.Linear(input_dim // 2, input_dim // 4)
        
        self.layer3 = nn.Linear(input_dim // 4, common_dim)
        self.relu = nn.ReLU()
 
   def forward(self, x):
        x= x.float()
        y = self.layer1(x)
        y = self.relu(y)
        y = self.layer2(y)
        y = self.relu(y)
        y = self.layer3(y)
        return y
    
    
    
