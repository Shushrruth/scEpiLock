import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PreProcessor():

    def __init__(self, data_dir):

        self.pos_fasta_path = "/scratch/share/Sai/swarup_lab/data_1/picks_disease.fa" # X-centered pos
        self.neg_fasta_path = data_dir + "x-centered-negatives-Random-2000bpAway.fa" 
        self.label_path = "/scratch/share/Sai/swarup_lab/data_1/label.tsv" # y-labels

    def get_ratio(self,df):
        weights = []
        for col in df.columns:
            array = df[col].values

            weights.append(sum(array==0)/sum(array==1))
  
        return weights

    def concat_data(self):

        pos_fasta = pd.read_csv(self.pos_fasta_path,sep=">chr*",header=None, engine='python').values[1::2][:,0] 
        neg_fasta = pd.read_csv(self.neg_fasta_path,sep=">chr*",header=None, engine='python').values[1::2][:,0] 
    
        label = pd.read_csv(self.label_path, sep='\t', header=0)

        pos_label = label.to_numpy()

        #train_weights = self.get_ratio(label)

        len_p = round(len(pos_fasta))
        
        np.random.seed(42)
        np.random.shuffle(neg_fasta)
        neg_fasta = neg_fasta[:len_p]



        neg_label = np.zeros((len_p, 7))

        #print(pos_label.shape)
        #print(neg_label.shape)
                
        data_1 = np.concatenate([pos_fasta,neg_fasta])
        label_1 = np.concatenate([pos_label,neg_label])
        train_weights = self.get_ratio(pd.DataFrame(label_1))

        #print("positive size = ", pos_fasta.shape)  
        #print("positive label size = ",pos_label.shape) 
        
        #print("total data size = " , neg_fasta.shape)
        #print("Total label size = ", neg_label.shape)


        return data_1, label_1, train_weights

    def split_train_test(self, data, label, test_size = 0.1):
        data_train_temp, data_test, label_train_temp, label_test = train_test_split(data, label, test_size=test_size, random_state=12)
        data_train, data_eval, label_train, label_eval = train_test_split(data_train_temp, label_train_temp, test_size=test_size, random_state=12)

        print("-----------train size concate------------")
        print(data_train.shape) #(173775,)
        print(label_train.shape) #(173775, 1)

        print("-----------eval size concate------------")
        print(data_eval.shape) #(19309,)
        print(label_eval.shape) #(19309, 1)

        print("-----------test size concate------------")
        print(data_test.shape) #(21454,)
        print(label_test.shape) #(21454,) no encode: (11454, 8)

        #test_size = data_test.shape[0]

        return data_train, data_eval, data_test, label_train, label_eval, label_test
