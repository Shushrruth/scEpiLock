import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MaxPool1d


class scEpiLock(nn.Module):
    def __init__(self,n_class):
        super(scEpiLock, self).__init__()

        self.Conv1 = nn.Conv1d(in_channels=4, out_channels = 100, kernel_size=10)

        self.Maxpool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.Conv2 = nn.Conv1d(in_channels = 50, out_channels = 100, kernel_size = 8)

        self.Maxpool2 = nn.MaxPool1d(kernel_size=3, stride=3)

        #self.Conv3 = nn.Conv1d(in_channels = 50, out_channels =100, kernel_size = 6)

        #self.Maxpool3 = nn.MaxPool1d(kernel_size=8, stride=6)

        self.Drop1 = nn.Dropout(p=0.3)

        self.Linear1 = nn.Linear(5350*2, 500)

        #self.Linear2 = nn.Linear(500, 500)

        self.Linear3 = nn.Linear(500,7)


    def forward(self, input):

        x = self.Conv1(input)

        x = F.relu(x)
        x = self.Maxpool1(x)
 
        x = self.Conv2(x)

        x = F.relu(x)
        x = self.Maxpool2(x)

        #x = self.Conv3(x)

        #x = F.relu(x)
        #x = self.Maxpool3(x)

        x = torch.flatten(x,1)

        x = self.Drop1(x)
        #print(x.shape)
        x = self.Linear1(x)

        x = F.relu(x)

        x = self.Drop1(x)

        #x = self.Linear2(x)

        x = F.relu(x)

        x = self.Linear3(x)

        #x = torch.sigmoid(x)

        return x

class scEpiLock_Siam(nn.Module):
    def __init__(self,n_class):
        super(scEpiLock_Siam, self).__init__()

        self.Conv1 = nn.Conv1d(in_channels=4, out_channels = 100, kernel_size=10)
        #1970
        self.Maxpool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        #985
        self.Conv2 = nn.Conv1d(in_channels = 100, out_channels = 100, kernel_size = 8)
        #966
        self.Maxpool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        #483
        #self.Conv3 = nn.Conv1d(in_channels = 50, out_channels = 50, kernel_size = 6)
        #476
        #self.Maxpool3 = nn.MaxPool1d(kernel_size=8, stride=6)
        #79
        self.Drop1 = nn.Dropout(p=0.3)

        self.Linear1 = nn.Linear(5350*2, 500)

        #self.Linear2 = nn.Linear(500, 500)

        self.Linear3 = nn.Linear(500,7)


    def forward_one(self, input):
        
        x = self.Conv1(input)

        x = F.relu(x)
        x = self.Maxpool1(x)
 
        x = self.Conv2(x)

        x = F.relu(x)
        x = self.Maxpool2(x)

        #x = self.Conv3(x)

        #x = F.relu(x)
        #x = self.Maxpool3(x)

        x = torch.flatten(x,1)

        x = self.Drop1(x)
        #print(x.shape)
        x = self.Linear1(x)

        x = F.relu(x)

        x = self.Drop1(x)

        #x = self.Linear2(x)

        x = F.relu(x)

        x = self.Linear3(x)

        return x
    
    def forward(self,x1,x2):

        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        
        out = (out1+out2)/2

        return torch.sigmoid(x) 