from data.data_processor import PreProcessor
from data.data_loader import DataLoader, DataLoader_Siam
from model.model import scEpiLock, scEpiLock_Siam
from model.spliceAI import SpliceAI_10k
from train.trainer import Trainer
from train.tester import Tester 
import torch
from datetime import datetime
from sklearn.metrics import classification_report
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt

#1 Choosing cell type

parser = argparse.ArgumentParser(description='Training a binary classififer')
parser.add_argument('run', type=str, help='Enter run')
args = parser.parse_args()

run = args.run

ct_path = "/scratch/share/Sai/projects/scEpiLock/multi_task/result/" + run + "/"
wt_path = "/scratch/share/Sai/data/scEpiLock_weights/"

if not os.path.exists(wt_path):
    os.makedirs(wt_path)

if not os.path.exists(ct_path):
    os.makedirs(ct_path)
#2 Reading and Processing and Loading the data 

data_dir = "/scratch/share/Sai/data/"
print(int(run))
data_class = PreProcessor(data_dir,int(run))
data, label,train_weights = data_class.concat_data()
data_train, data_eval, data_test, label_train, label_eval, label_test = data_class.split_train_test(data,label)

print("Seq len is", len(data[0]))
#3 Loading and one-hot-encoding



train_data_loader = DataLoader(data_train, label_train)
eval_data_loader = DataLoader(data_eval, label_eval)
print('Loading')

print(train_data_loader[1])
print(eval_data_loader[1])


test_data_loader = DataLoader_Siam(data_test, label_test)
print('done')

print(test_data_loader[1])

#4 Trainer
model_wt_path = wt_path + "model_"+run+ ".pt"
learning_rate = 5e-5
epochs = 1
batch_size = 64
weight_decay = 0


#### Model hyperparams #####
n_class = 7
input_dim = int(run) # or 1000
cnn_kernel_1 = 10
cnn_kernel_2 = 4
cnn_channel_1 = 320
cnn_channel_2 = 480
cnn_channel_3 = 960
cnn_channel_4 = 1024
max_kernel = 4
max_stride = 4
linear = 925
drop = 0.5

model = scEpiLock(n_class,input_dim,cnn_kernel_1,cnn_kernel_2 ,cnn_channel_1,cnn_channel_2,cnn_channel_3,cnn_channel_4,max_kernel, max_stride,linear,drop)
#model = scEpiLock(n_class)
#model = SpliceAI_10k(n_class)
############################

trainer = Trainer(train_data_loader, eval_data_loader, model_wt_path, epochs, batch_size, learning_rate, weight_decay , '/', n_class,model, ct_path,'Yes',train_weights)


print("train start time: ", datetime.now())
trainer.train()
print("train end time: ", datetime.now())


# Testing

test_out_dir = ct_path

model = scEpiLock_Siam(n_class,input_dim,cnn_kernel_1,cnn_kernel_2 ,cnn_channel_1,cnn_channel_2,cnn_channel_3,cnn_channel_4,max_kernel, max_stride,linear,drop)
tester = Tester(model, model_wt_path, test_data_loader,10, n_class)

y_true, y_pred = tester.test()
print('testing done')
print('y_true')
print(y_true)
print('y_pred')
print(y_pred)

if not os.path.exists(test_out_dir+'plots/'):
    os.makedirs(test_out_dir+'plots/')

Tester.plot_prc_curve(y_true.flatten(), y_pred.flatten(),test_out_dir+'plots/PRC_plot.png')
Tester.plot_roc_curve(y_true.flatten(), y_pred.flatten(),test_out_dir+'plots/ROC_plot.png')

target_names = ['Negatives', 'Positives']
print(classification_report(y_true.flatten(), y_pred.round().flatten(), target_names=target_names))

print('.')
print('.')
columns = ['Ast','End','Ex' ,'In', 'Mic' ,'Oli','OPC' ]
print(classification_report(y_true.round(), y_pred.round(), target_names=columns))


np.save(ct_path + 'y_pred.npy', y_pred)
np.save(ct_path + 'y_true.npy', y_true)



# Plotting the loss
train_loss = np.load(ct_path + 'train_loss.npy')
val_loss = np.load(ct_path +'val_loss.npy')


def plot_loss():

    plt.figure(dpi=600)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('loss curves')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train loss' , 'val loss'])
    plt.savefig(ct_path + 'plots/loss_curves.png')

plot_loss()


#
print("Done")
