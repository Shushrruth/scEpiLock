from data.data_processor import PreProcessor
from data.data_loader import DataLoader
from model.model import scEpiLock
from train.trainer import Trainer 
import torch

#1 Reading and Processing and Loading the data 

data_dir = "/gpfs/gibbs/pi/girgenti/JZhang/commonData/scEpiLock/traning/"
data_class = PreProcessor(data_dir)
data, label = data_class.concat_data()
data_train, data_eval, data_test, label_train, label_eval, label_test = data_class.split_train_test(data,label)

# Loading and one-hot-encoding

train_data_loader = DataLoader(data_train, label_train)
eval_data_loader = DataLoader(data_eval, label_eval)



# Trainer

model_wt_path = "binary_classifier/result/weights/04-18-model.pt"
epochs = 5
batch_size = 64
learning_rate = 1e-4
weight_decay = 5e-4
n_class = 7

trainer = Trainer(train_data_loader, eval_data_loader, model_wt_path, epochs, batch_size, 
	learning_rate, weight_decay, '_' , '/', n_class,scEpiLock)

#trainer.train()
