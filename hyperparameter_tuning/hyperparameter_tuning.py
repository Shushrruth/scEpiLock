import os
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
#from torchvision import datasets
#from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import copy
import os
import numpy as np

###### Getting and Loading the Data ########

class PreProcessor():

    def __init__(self, data_dir):

        self.pos_fasta_path = data_dir + "union_peaks_1000.fa" # X-centered pos
        #self.neg_fasta_path = data_dir + "x-centered-negatives-ENCODE-5000bpAway.fa" 
        self.label_path = data_dir + "new_lables.tsv" # y-labels

    def get_ratio(self,df):
        weights = []
        for col in df.columns:
            array = df[col].values

            weights.append(sum(array==0)/sum(array==1))
  
        return weights

    def concat_data(self):

        pos_fasta = pd.read_csv(self.pos_fasta_path,sep=">chr*",header=None, engine='python').values[1::2][:,0] 
        #neg_fasta = pd.read_csv(self.neg_fasta_path,sep=">chr*",header=None, engine='python').values[1::2][:,0] 

        label = pd.read_csv(self.label_path, sep='\t', header=0)

        pos_label = label.to_numpy()

        train_weights = self.get_ratio(label)

        #len_p = round(len(pos_fasta)/5)
        
        #np.random.seed(42)
        #np.random.shuffle(neg_fasta)
        #neg_fasta = neg_fasta[:len_p]



        #neg_label = np.zeros((len_p, 7))

        #print(pos_label.shape)
        #print(neg_label.shape)
                
        #data_1 = np.concatenate([pos_fasta,neg_fasta])
        #label_1 = np.concatenate([pos_label,neg_label])
        

        #print("positive size = ", pos_fasta.shape)  
        #print("positive label size = ",pos_label.shape) 
        
        #print("total data size = " , neg_fasta.shape)
        #print("Total label size = ", neg_label.shape)


        return pos_fasta, pos_label, train_weights

    def split_train_test(self, data, label, test_size = 0.1):
        data_train_temp, data_test, label_train_temp, label_test = train_test_split(data, label, test_size=test_size, random_state=12)
        data_train, data_eval, label_train, label_eval = train_test_split(data_train_temp, label_train_temp, test_size=test_size, random_state=12)

        #print("-----------train size concate------------")
        #print(data_train.shape) #(173775,)
        #print(label_train.shape) #(173775, 1)

        #print("-----------eval size concate------------")
        #print(data_eval.shape) #(19309,)
        #print(label_eval.shape) #(19309, 1)

        #print("-----------test size concate------------")
        #print(data_test.shape) #(21454,)
        #print(label_test.shape) #(21454,) no encode: (11454, 8)

        #test_size = data_test.shape[0]

        return data_train, data_eval, data_test, label_train, label_eval, label_test



class DataLoader(Dataset):

    # The __init__ function is run once when instantiating the Dataset object.
    def __init__(self, data, label, subset ="True"):
        # (n,) array, each element is string, dtype=object
        self.data = data # fasta of forward, no chr title, 1d np.array, shape is n
        self.label = label # cell cluster has been handled in pre-process
        #print("-----------------shape before add RC -------------") # TODO Delete the print -YG
        #print(self.data.shape)
        #print(self.label.shape)


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
        #print("-----------------shape after subset and add RC-------------")
        #print(self.data.shape)
        #print(self.label.shape)

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


##### Define the Model #####

def define_model(trial):

	in_features = 4*1000
	cnn_kernel_1 = trial.suggest_int("cnn_kernel_1", 4, 16)
	cnn_kernel_2 = trial.suggest_int("cnn_kernel_2", 4, 16)
	channel_1 = trial.suggest_int("cnn_channel_1", 100, 1000)
	channel_2 = trial.suggest_int("cnn_channel_2", 100, 1000)
	channel_3 = trial.suggest_int("cnn_channel_3", 100, 1000)
	channel_4 = trial.suggest_int("cnn_channel_4", 100, 1000)
	max_kernel = trial.suggest_int("max_kernel", 2, 8)
	max_stride = trial.suggest_int("max_kernel", 2, 8)
	linear = trial.suggest_int("max_kernel", 500, 1000)
	drop = trial.suggest_float("dropout_l", 0.2, 0.5)


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


def train(network, optimizer):
    """Trains the model.
    Parameters:
        - network (__main__.Net):              The CNN
        - optimizer (torch.optim.<optimizer>): The optimizer for the CNN
    """
    criterion = nn.BCEWithLogitsLoss().to(device)
    network = network.to(device)
    network.train()  # Set the module in training mode (only affects certain modules)
    for (data, target) in train_loader:  # For each batch


        optimizer.zero_grad()                                 # Clear gradients
        output = network(data.to(device))                     # Forward propagation
        loss = criterion(output, target.to(device))          # Compute loss (negative log likelihood: −log(y))
        loss.backward()                                       # Compute gradients
        optimizer.step()  

def prc_value(y_true, y_proba):

    lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_proba)
    lr_auco = auc(lr_recall, lr_precision)

    return lr_auco

def test(network):
    """Tests the model.
    Parameters:
        - network (__main__.Net): The CNN
    Returns:
        - accuracy_test (torch.Tensor): The test accuracy
    """
    y_pred = np.empty((0, 7))
    y_true = np.empty((0, 7))
    y_proba = np.empty((0, 7))
    network = network.to(device)
    network.eval()         # Set the module in evaluation mode (only affects certain modules)
    with torch.no_grad():  # Disable gradient calculation (when you are sure that you will not call Tensor.backward())
        for (data, target) in test_loader:  # For each batch



            output = network(data.to(device))               
            target = target.float()

            y_true = np.concatenate((y_true, target.cpu().numpy()))
            y_pred = np.concatenate((y_pred, output.cpu().numpy()))

    prc = prc_value(y_true.flatten(),y_pred.flatten())

    return prc

def objective(trial):

	n_epochs = 20
	n_class = 7
	input_dim = 1000
	cnn_kernel_1 = trial.suggest_int("cnn_kernel_1", 4, 16,1)
	cnn_kernel_2 = trial.suggest_int("cnn_kernel_2", 4, 16,1)
	cnn_channel_1 = trial.suggest_int("cnn_channel_1", 100, 1000,100)
	cnn_channel_2 = trial.suggest_int("cnn_channel_2", 100, 1000,100)
	cnn_channel_3 = trial.suggest_int("cnn_channel_3", 100, 1000,100)
	cnn_channel_4 = trial.suggest_int("cnn_channel_4", 100, 1000,100)
	max_kernel = trial.suggest_int("max_kernel", 2, 6,1)
	max_stride = trial.suggest_int("max_stride", 2, 6,1)
	linear = trial.suggest_int("dense", 500, 1000,50)
	drop = trial.suggest_float("dropout_l", 0.2, 0.5)

	model = scEpiLock(n_class,input_dim,cnn_kernel_1, cnn_kernel_2,cnn_channel_1,cnn_channel_2,cnn_channel_3,cnn_channel_4,max_kernel, max_stride,linear,drop)
	optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
	lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)  
	optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

	for epoch in range(n_epochs):
		train(model, optimizer)
		accuracy = test(model) 
		trial.report(accuracy, epoch)
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

	return accuracy


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    number_of_trials = 100
    data_dir = "/scratch/share/Sai/data/"
    data_class = PreProcessor(data_dir)
    data, label,train_weights = data_class.concat_data()
    data_train, data_eval, data_test, label_train, label_eval, label_test = data_class.split_train_test(data,label)

    train_data_loader = DataLoader(data_train, label_train)
    eval_data_loader = DataLoader(data_eval, label_eval)
    batch_size = 128

    train_loader = torch.utils.data.DataLoader(train_data_loader, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(eval_data_loader, batch_size=batch_size, shuffle=True)
    print('Data loaded')
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=number_of_trials)

    #-------------------------
    # Results
    #-------------------------
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv('/scratch/share/Sai/projects/scEpiLock/hyperparameter_tuning/results_full.tsv', sep='\t',index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(study, target=None)

    # Display the most important hyperparameters
    print('\nMost important hyperparameters:')
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))

