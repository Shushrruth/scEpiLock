import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import os
import numpy as np

class Trainer:
    def __init__(self, train_data, eval_data, model_path, num_epochs, batch_size,
                 learning_rate, weight_decay, pos_weight, plot_path, n_class,model):

        '''
        :param model: the model
        :param train_data: complete train data, include x and y
        :param model_path: path the save the trained model
        '''

        
        self.train_data = train_data
        self.eval_data = eval_data
        self._model_path = model_path
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.plot_path = plot_path
        self.n_class = n_class
        self.model = model
        


    def train(self):
        """ train """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = self.model(self.n_class).to(device)


        train_loss_hist = []
        eval_loss_hist = []

        train_acc_hist = []
        eval_acc_hist = []

        best_loss = 1000


        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay, amsgrad=True)

        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self._batch_size, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(self.eval_data, batch_size=self._batch_size, shuffle=True)

        for epoch in range(self._num_epochs):
            print('Epoch {}/{}'.format(epoch, self._num_epochs - 1))
            print('-' * 10)

            # Train
            train_loss = 0.0
            train_acc = 0.0

            model.train()  # set to train mode, use dropout and batchnorm ##

            for train_step, (X, y) in tqdm(enumerate(train_loader)):

                X = X.to(device)
                y = y.to(device)

                # Forward pass: Compute predicted y by passing x to the model
                y_pred_logit = model(X.float()) # predicted value

                loss = criterion(y_pred_logit.float(), y.float())

                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step() 

                # statistics
                y_pred_prob = 1 / (1 + np.exp(-y_pred_logit.cpu().detach().numpy())) # calculate prob from logit
                y_pred = y_pred_prob.round() ## 0./1.
                y = y.cpu().detach().numpy()

                train_loss += loss.item() * X.size(0) #loss.item() has be reducted by batch size
                if epoch == 0 and train_step == 0:
                    initial_loss = loss.item()
                    initial_acc = np.sum(y_pred == y)/float(self._batch_size*self.n_class)

                train_acc += np.sum(y_pred == y)

            train_loss = train_loss/len(train_loader.dataset)
            train_acc = train_acc/ float(len(train_loader.dataset)*self.n_class)
            print('Epoch {} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, 'train', train_loss, train_acc))


            # Evaluation
            eval_loss = 0
            eval_acc = 0

            model.eval() 
            with torch.no_grad(): 
                for X, y in tqdm(eval_loader):
                    optimizer.zero_grad() 
                    X, y = X.to(device), y.to(device)
                    y_pred_logit = model(X.float())
                    eva_loss = criterion(y_pred_logit.float(), y.float())

                    y_pred_prob = 1 / (1 + np.exp(-y_pred_logit.cpu().detach().numpy()))  
                    y_pred = y_pred_prob.round() ## 0./1.
                    y = y.cpu().detach().numpy()

                    # statistics
                    eval_loss += eva_loss.item() * X.size(0)
                    eval_acc += np.sum(y_pred == y)

                eval_loss = eval_loss / len(eval_loader.dataset)
                eval_acc = eval_acc / float(len(eval_loader.dataset)*self.n_class)

            print("Eval loss" , eval_loss)

            eval_loss_hist.append(eval_loss)
            eval_acc_hist.append(eval_acc)

            if eval_loss < best_loss:
                best_loss = eval_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(best_model_wts, self._model_path)