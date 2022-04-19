from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
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
        self.test_data = test_loader
        self.batch_size = batch_size
        self.n_class = n_class

    def tester(self):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = self.model(n_class).to(device)
        model.load_state_dict(torch.load(self.model_weight_path))
        model.eval()

        test_loss = 0
        correct = 0
        y_pred = []
        y_true = []
        y_proba = []
        with torch.no_grad():

            for data1, data2, target in tqdm(self.testloader):
                data1, data2, target = data1.to(device), data2.to(device), target.to(device)

                output = model(data1,data2).squeeze()
                test_loss += nn.BCELoss()(output, target).item()*1e+1  # sum up batch loss
                prob = (output)
                y_probas=prob.cpu().numpy()
                
                prob[prob>=0.5] = 1
                prob[prob<0.5] = 0
                correct += torch.sum(prob==target)
                for i in range(len(prob)):
                    y_proba.append(float(y_probas[i]))
                    y_pred.append(float(prob[i]))
                    y_true.append(float(target[i]))


        return y_true, y_proba, y_pred



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





