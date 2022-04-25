from data.data_processor import PreProcessor
from data.data_loader import DataLoader, DataLoader_Siam
from model.model import scEpiLock, scEpiLock_Siam
from train.trainer import Trainer
from train.tester import Tester 
import torch
from datetime import datetime

#1 Reading and Processing and Loading the data 

data_dir = "/gpfs/gibbs/pi/girgenti/JZhang/commonData/scEpiLock/traning/"
data_class = PreProcessor(data_dir)
data, label = data_class.concat_data()
data_train, data_eval, data_test, label_train, label_eval, label_test = data_class.split_train_test(data,label)

# Loading and one-hot-encoding

train_data_loader = DataLoader(data_train, label_train)
eval_data_loader = DataLoader(data_eval, label_eval)


test_data_loader = DataLoader_Siam(data_test, label_test)


# Trainer

model_wt_path = "/home/sss253/project/scEpiLock_weights/04-18-model_v2.pt"
learning_rate = 5e-4
epochs = 10
batch_size = 64
weight_decay = 5e-4
n_class = 7

trainer = Trainer(train_data_loader, eval_data_loader, model_wt_path, epochs, batch_size, 
	learning_rate, weight_decay, '_' , '/', n_class,scEpiLock)


print("train start time: ", datetime.now())
trainer.train()
print("train end time: ", datetime.now())


# Testing

test_out_dir = '/home/sss253/project/scEpiLock/binary_classifier/result/plots/'

tester = Tester(scEpiLock_Siam, model_wt_path, test_data_loader, batch_size, n_class)
y_true, y_pred = tester.test()
print('testing done')
print('y_true')
print(y_true)
print('y_pred')
print(y_pred)

Tester.plot_prc_curve(y_true.flatten(), y_pred.flatten(),test_out_dir+'PRC_plot_v2.png')
Tester.plot_roc_curve(y_true.flatten(), y_pred.flatten(),test_out_dir+'ROC_plot_v2.png')


