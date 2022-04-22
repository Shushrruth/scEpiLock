import numpy as np
import torch
from torch.utils.data import Dataset
#from datetime import datetime

class DataLoader(Dataset):

    # The __init__ function is run once when instantiating the Dataset object.
    def __init__(self, data, label, subset ="True"):
        # (n,) array, each element is string, dtype=object
        self.data = data # fasta of forward, no chr title, 1d np.array, shape is n
        self.label = label # cell cluster has been handled in pre-process
        print("-----------------shape before add RC -------------") # TODO Delete the print -YG
        print(self.data.shape)
        print(self.label.shape)


        # add reverse complement
        temp = []
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N' : 'N',
                      'a': 't', 't': 'a', 'g': 'c', 'c': 'g', 'n' : 'n'}
        #print("reverse complement start time: ", datetime.now())
        for seq in self.data:
            #print(seq)
            complement_seq = ''
            for base in seq: 
                complement_seq = complement[base] + complement_seq

            temp.append(complement_seq)

        #print("reverse complement end time: ", datetime.now())
        temp = np.array(temp, dtype=object)
        self.data = np.append(self.data, temp, axis=0)
        self.label = np.append(self.label, self.label, axis=0)
        print("-----------------shape after subset and add RC-------------")
        print(self.data.shape)
        print(self.label.shape)

    # The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return self.data.shape[0] ## check

    # The __getitem__ function loads and returns a sample from the dataset at the given index idx.
    def __getitem__(self, index):
        seq = self.data[index]
        row_index = 0
        temp = np.zeros((len(seq), 4))
        #print(seq)
        for base in seq:
            if base == 'A' or base == 'a':
                temp[row_index, 0] = 1
            elif base == 'T' or base == 't':
                temp[row_index, 1] = 1
            elif base == 'G' or base == 'g':
                temp[row_index, 2] = 1
            elif base == 'C' or base == 'c':
                temp[row_index, 3] = 1
            row_index += 1

        X = torch.tensor(temp).float().permute(1,0) # change the dim to 4, 1000
        y = torch.tensor(self.label[index]).float()

        return X, y


class DataLoader_Siam(Dataset):

    # The __init__ function is run once when instantiating the Dataset object.
    def __init__(self, data, label, subset ="True"):
        # (n,) array, each element is string, dtype=object
        self.data = data # fasta of forward, no chr title, 1d np.array, shape is n
        self.label = label # cell cluster has been handled in pre-process
        print("-----------------shape before add RC -------------") # TODO Delete the print -YG
        print(self.data.shape)
        print(self.label.shape)

    def RC(self,seq):
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N' : 'N',
                      'a': 't', 't': 'a', 'g': 'c', 'c': 'g', 'n' : 'n'}

        complement_seq = ''
        for base in seq: 
            complement_seq = complement[base] + complement_seq

        return complement_seq



    # The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return self.data.shape[0] ## check


    # The __getitem__ function loads and returns a sample from the dataset at the given index idx.
    def __getitem__(self, index):

        seq = self.data[index]
        rc_seq = self.RC(seq)
        row_index = 0
        temp1 = np.zeros((len(seq), 4))
        #print(seq)
        for base in seq:
            if base == 'A' or base == 'a':
                temp1[row_index, 0] = 1
            elif base == 'T' or base == 't':
                temp1[row_index, 1] = 1
            elif base == 'G' or base == 'g':
                temp1[row_index, 2] = 1
            elif base == 'C' or base == 'c':
                temp1[row_index, 3] = 1
            row_index += 1

        temp2 = np.zeros((len(seq), 4))
        #print(seq)
        for base in rc_seq:
            if base == 'A' or base == 'a':
                temp2[row_index, 0] = 1
            elif base == 'T' or base == 't':
                temp2[row_index, 1] = 1
            elif base == 'G' or base == 'g':
                temp2[row_index, 2] = 1
            elif base == 'C' or base == 'c':
                temp2[row_index, 3] = 1
            row_index += 1

        X1 = torch.tensor(temp1).float().permute(1,0) # change the dim to 4, 1000
        X2 = torch.tensor(temp2).float().permute(1,0)
        y = torch.tensor(self.label[index]).float()

        return X1, X2, y