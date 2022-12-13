from torch import nn
import torch.nn.functional as F
import torch


class scEpiLock(nn.Module):
    def __init__(self, n_class):
        super(scEpiLock, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=100, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=8)
        self.Conv3 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=4)
        self.Conv4 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=4)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.Linear1 = nn.Linear(26*100, 925)
        self.Linear2 = nn.Linear(925, n_class)

    def forward(self, input):
        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 4, 1000]
        # Output Tensor Shape: [batch_size, 100, 993]
        x = self.Conv1(input)
        x = F.relu(x)
        # # Pooling Layer 1
        # # Input Tensor Shape: [batch_size, 100, 993]
        # # Output Tensor Shape: [batch_size, 100, 248]
        # x = self.Maxpool(x)
        # x = self.Drop1(x)

        # Convolution Layer 2
        # Input Tensor Shape: [batch_size, 100, 993]
        # Output Tensor Shape: [batch_size, 100, 986]
        x = self.Conv2(x)
        x = F.relu(x)
        # Pooling Layer 1100
        # Input Tensor Shape: [batch_size, 100, 986]
        # Output Tensor Shape: [batch_size, 100, 246]
        x = self.Maxpool(x)
        x = self.Drop1(x)

        # Convolution Layer 3
        # Input Tensor Shape: [batch_size, 100, 246]
        # Output Tensor Shape: [batch_size, 100, 243]
        x = self.Conv3(x)
        x = F.relu(x)
        # Pooling Layer 2
        # Input Tensor Shape: [batch_size, 100, 243]
        # Output Tensor Shape: [batch_size, 100, 60]
        x = self.Maxpool(x)
        x = self.Drop2(x)

        # Convolution Layer 4
        # Input Tensor Shape: [batch_size, 100, 60]
        # Output Tensor Shape: [batch_size, 100, 57]
        x = self.Conv4(x)
        x = F.relu(x)
        x = self.Drop2(x)

        # Pooling Layer 3
        # Input Tensor Shape: [batch_size, 100, 57]
        # Output Tensor Shape: [batch_size, 100, 14]
        #x = self.Maxpool(x)
        #print(x.shape)
        x = x.view(-1, 26*100)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x



from torch import nn
import torch.nn.functional as F
import torch


class scEpiLock_Siam(nn.Module):
    def __init__(self, n_class):
        super(scEpiLock_Siam, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=100, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=8)
        self.Conv3 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=4)
        self.Conv4 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=4)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.Linear1 = nn.Linear(26*100, 925)
        self.Linear2 = nn.Linear(925, n_class)

    def forward_one(self, input):
        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 4, 1000]
        # Output Tensor Shape: [batch_size, 100, 993]
        x = self.Conv1(input)
        x = F.relu(x)
        # # Pooling Layer 1
        # # Input Tensor Shape: [batch_size, 100, 993]
        # # Output Tensor Shape: [batch_size, 100, 248]
        # x = self.Maxpool(x)
        # x = self.Drop1(x)

        # Convolution Layer 2
        # Input Tensor Shape: [batch_size, 100, 993]
        # Output Tensor Shape: [batch_size, 100, 986]
        x = self.Conv2(x)
        x = F.relu(x)
        # Pooling Layer 1100
        # Input Tensor Shape: [batch_size, 100, 986]
        # Output Tensor Shape: [batch_size, 100, 246]
        x = self.Maxpool(x)
        x = self.Drop1(x)

        # Convolution Layer 3
        # Input Tensor Shape: [batch_size, 100, 246]
        # Output Tensor Shape: [batch_size, 100, 243]
        x = self.Conv3(x)
        x = F.relu(x)
        # Pooling Layer 2
        # Input Tensor Shape: [batch_size, 100, 243]
        # Output Tensor Shape: [batch_size, 100, 60]
        x = self.Maxpool(x)
        x = self.Drop2(x)

        # Convolution Layer 4
        # Input Tensor Shape: [batch_size, 100, 60]
        # Output Tensor Shape: [batch_size, 100, 57]
        x = self.Conv4(x)
        x = F.relu(x)
        x = self.Drop2(x)

        # Pooling Layer 3
        # Input Tensor Shape: [batch_size, 100, 57]
        # Output Tensor Shape: [batch_size, 100, 14]
        #x = self.Maxpool(x)
        #print(x.shape)
        x = x.view(-1, 26*100)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x

    def forward(self,x1,x2):

        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        
        out = (out1+out2)/2

        return torch.sigmoid(out)