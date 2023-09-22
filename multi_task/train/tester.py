from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import os
import numpy as np



class Tester():
    def __init__(self,model, model_weight_path,test_data, batch_size, n_class):

        self.model = model
        self.model_weight_path = model_weight_path
        self.test_data = test_data
        self.batch_size = batch_size
        self.n_class = n_class

    def test(self):

        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        model.load_state_dict(torch.load(self.model_weight_path))


        test_loss = 0
        correct = 0
        y_pred = np.empty((0, self.n_class))
        y_true = np.empty((0, self.n_class))
        y_proba = np.empty((0, self.n_class))
        model.eval() 

        with torch.no_grad(): 
            for X1,X2, y in tqdm(test_loader):
                 
                X1,X2, y = X1.to(device), X2.to(device),y.to(device)

                output = model(X1.float(), X2.float())

                y_hat = output
                y = y.float()

                y_true = np.concatenate((y_true, y.cpu().numpy()))
                y_pred = np.concatenate((y_pred, y_hat.cpu().numpy()))


        return y_true, y_pred



    def plot_prc_curve(y_true, y_proba, plot_destination_path):

        pyplot.figure(dpi=600)
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_proba)
        lr_auco = auc(lr_recall, lr_precision)
        pyplot.plot(lr_recall, lr_precision)
        pyplot.xlabel('Recall', fontsize = 13)
        pyplot.ylabel('Precision', fontsize = 13)
        pyplot.title('Precision-Recall curve',fontsize = 15)
        pyplot.legend(['Overall (AUC = %0.3f)' % lr_auco])
        pyplot.savefig(plot_destination_path, bbox_inches='tight')


    def plot_roc_curve(y_true, y_proba, plot_destination_path):

        pyplot.figure(dpi=600)
        lr_auco = roc_auc_score(y_true,y_proba)
        lr_fpr, lr_tpr, _ = roc_curve(y_true, y_proba)
        pyplot.plot(lr_fpr, lr_tpr)
        pyplot.xlabel('False Positive Rate', fontsize = 13)
        pyplot.ylabel('True Positive Rate', fontsize = 13)
        pyplot.title('Receiver Operating Characteristic curve',fontsize = 15)
        pyplot.legend(['Overall (AUC = %0.3f)' % lr_auco])
        pyplot.savefig(plot_destination_path, bbox_inches='tight')





