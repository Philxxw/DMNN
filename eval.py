#encoding=utf-8
#from sklearn.metrics import mean_squared_error,mean_absolute_error
import torch
import torch.nn as nn

# calculate MAPE
class calcMAPE(nn.Module):
    def __init__(self, epsion = 0.0000000):
        super(calcMAPE,self).__init__()
        self.epsion = epsion
        
    def forward(self,true, pred):
        true += self.epsion
        loss =  torch.mean(torch.abs((true-pred)/true))*100
        
        return loss
# calculate SMAPE

class calcSMAPE(nn.Module):
    def __init__(self):
        super(calcMAPE,self).__init__()
        
    def forward(self,true, pred):
        delim = (torch.abs(true)+torch.abs(pred))/2
        loss =  torch.mean(torch.abs((true-pred)/delim))*100
        
        return loss
    
#calculate cosine distance    
class calcCOSine(nn.Module):
    def __init__(self):
        super(calcCOSine,self).__init__()
    def forward(self,true,pred):
        s = torch.sum(torch.mul(true,pred))
        m = torch.mul(true.norm(2),pred.norm(2))
        loss = torch.div(s,m)
        return loss


#calculate MSE        
class calcMSE(nn.Module):
    def __init__(self):
        super(calcMSE,self).__init__()
    def forward(self,true,pred):
        s = torch.sub(true,pred)
        loss = torch.mean(torch.mul(s,s))
        return loss




