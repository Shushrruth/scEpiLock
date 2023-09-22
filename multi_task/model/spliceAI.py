import torch
import torch.nn as nn



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return x + self.fn(x)

def ResidualBlock(in_channels, out_channels, kernel_size, dilation):
    return Residual(nn.Sequential(
        nn.BatchNorm1d(in_channels),
        nn.ReLU(),
        nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding='same'),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding='same')
    ))


class SpliceAI_400nt(nn.Module):
    S = 400

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(4, 32, 1, dilation=1, padding='same')
        self.res_conv1 = nn.Conv1d(32, 32, 1, dilation=1, padding='same')

        self.block1 = nn.Sequential(
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
        )

        self.res_conv2 = nn.Conv1d(32, 32, 1, dilation=1, padding='same')

        self.block2 = nn.Sequential(
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            nn.Conv1d(32, 32, 1, dilation=1, padding='same'),
        )

        self.conv_last = nn.Conv1d(32, 3, 1, dilation=1, padding='same')
    
    def forward(self, x):
        x = self.conv1(x)
        detour = self.res_conv1(x)

        x = self.block1(x)
        detour += self.res_conv2(x)

        x = self.block2(x) + detour
        x = self.conv_last(x)

        return x#rearrange(x[..., 200:5000 + 200], 'b c l -> b l c')



class SpliceAI_10k(nn.Module):
    S = 10000

    def __init__(self,n_class):
        super().__init__()

        self.conv1 = nn.Conv1d(4, 32, 1, dilation=1, padding='same')
        self.res_conv1 = nn.Conv1d(32, 32, 1, dilation=1, padding='same')

        self.block1 = nn.Sequential(
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
        )

        self.res_conv2 = nn.Conv1d(32, 32, 1, dilation=1, padding='same')

        self.block2 = nn.Sequential(
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
        )

        self.res_conv3 = nn.Conv1d(32, 32, 1, dilation=1, padding='same')

        self.block3 = nn.Sequential(
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
        )

        self.res_conv4 = nn.Conv1d(32, 32, 1, dilation=1, padding='same')

        self.block4 = nn.Sequential(
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
        )

        self.conv_last = nn.Conv1d(32, 3, 1)
        self.max = torch.nn.MaxPool1d(10,3)

        self.linear1 = nn.Linear(9993,500)
        self.linear2 = nn.Linear(500,n_class)
    
    def forward(self, x):
        x = self.conv1(x)
        detour = self.res_conv1(x)

        x = self.block1(x)
        detour += self.res_conv2(x)

        x = self.block2(x)
        detour += self.res_conv3(x)

        x = self.block3(x)
        detour += self.res_conv4(x)

        x = self.block4(x) + detour
        #x = nn.ReLU(x)

        x = self.conv_last(x)
        x = self.max(x)
        #x = nn.ReLU(x)
        #print(x.shape)
        x = torch.flatten(x,1)
        #print(x.shape)

        x = self.linear1(x)
        

        x = self.linear2(x)

        return x #torch.sigmoid(x)

    def forward_one(self,x1,x2):

        out1 = self.forward(x1)
        out2 = self.forward(x2)
        out = (out1 + out2)/2
        return torch.sigmoid(out)

