from torch import nn
import torch.nn.functional as F
import torch
import numpy


class scEpiLock(nn.Module):
    def __init__(self, n_class,input_dim,cnn_kernel_1,cnn_kernel_2 ,cnn_channel_1,cnn_channel_2,cnn_channel_3,cnn_channel_4,max_kernel, max_stride,linear,drop):
        super(scEpiLock, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=cnn_channel_1, kernel_size=cnn_kernel_1)
        self.Conv2 = nn.Conv1d(in_channels=cnn_channel_1, out_channels=cnn_channel_2, kernel_size=cnn_kernel_1)
        self.Maxpool = nn.MaxPool1d(kernel_size= max_kernel, stride= max_stride)
        #done
        self.Conv3 = nn.Conv1d(in_channels=cnn_channel_2, out_channels=cnn_channel_3, kernel_size=cnn_kernel_2)
        self.Maxpool = nn.MaxPool1d(kernel_size= max_kernel, stride= max_stride)
        self.Conv4 = nn.Conv1d(in_channels=cnn_channel_3, out_channels=cnn_channel_4, kernel_size=cnn_kernel_2)
        
        self.Drop = nn.Dropout(p=drop)


        self.dense_1 = int(((input_dim - 2*cnn_kernel_1 + 2) - max_kernel)//max_stride) + 1
        self.dense_1 = int(((self.dense_1 - cnn_kernel_2 +1) - max_kernel)//max_stride) - cnn_kernel_2 + 2
        #print(self.dense_1)

        self.Linear1 = nn.Linear(self.dense_1*cnn_channel_4, linear)
        self.Linear2 = nn.Linear(linear, n_class)

    def forward(self, input):

        x = self.Conv1(input)
        x = F.relu(x)

        x = self.Conv2(x)
        x = F.relu(x)

        x = self.Maxpool(x)
        x = self.Drop(x)

        x = self.Conv3(x)
        x = F.relu(x)

        x = self.Maxpool(x)
        x = self.Drop(x)

        x = self.Conv4(x)
        x = F.relu(x)
        x = self.Drop(x)
        #print(x.shape)
        x = torch.flatten(x,1)
        #print(x.shape)
        x = self.Drop(x)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)

        return x

class scEpiLock_Siam(nn.Module):
    def __init__(self, n_class,input_dim,cnn_kernel_1,cnn_kernel_2 ,cnn_channel_1,cnn_channel_2,cnn_channel_3,cnn_channel_4,max_kernel, max_stride,linear,drop):
        super(scEpiLock_Siam, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=cnn_channel_1, kernel_size=cnn_kernel_1)
        self.Conv2 = nn.Conv1d(in_channels=cnn_channel_1, out_channels=cnn_channel_2, kernel_size=cnn_kernel_1)
        self.Maxpool = nn.MaxPool1d(kernel_size= max_kernel, stride= max_stride)
        #done
        self.Conv3 = nn.Conv1d(in_channels=cnn_channel_2, out_channels=cnn_channel_3, kernel_size=cnn_kernel_2)
        self.Maxpool = nn.MaxPool1d(kernel_size= max_kernel, stride= max_stride)
        self.Conv4 = nn.Conv1d(in_channels=cnn_channel_3, out_channels=cnn_channel_4, kernel_size=cnn_kernel_2)
        
        self.Drop = nn.Dropout(p=drop)


        self.dense_1 = int(((input_dim - 2*cnn_kernel_1 + 2) - max_kernel)//max_stride) + 1
        self.dense_1 = int(((self.dense_1 - cnn_kernel_2 +1) - max_kernel)//max_stride) - cnn_kernel_2 + 2
        #print(self.dense_1)

        self.Linear1 = nn.Linear(self.dense_1*cnn_channel_4, linear)
        self.Linear2 = nn.Linear(linear, n_class)

    def forward_one(self, input):

        x = self.Conv1(input)
        x = F.relu(x)

        x = self.Conv2(x)
        x = F.relu(x)

        x = self.Maxpool(x)
        x = self.Drop(x)

        x = self.Conv3(x)
        x = F.relu(x)

        x = self.Maxpool(x)
        x = self.Drop(x)

        x = self.Conv4(x)
        x = F.relu(x)
        x = self.Drop(x)
        #print(x.shape)
        x = torch.flatten(x,1)
        #print(x.shape)
        x = self.Drop(x)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)

        return x

    def forward(self,x1,x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)

        outout = (out1+out2)/2

        return torch.sigmoid(out)

