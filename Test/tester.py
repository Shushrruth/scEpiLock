import pandas as pd
import numpy as np

data_dir = "/gpfs/gibbs/pi/girgenti/JZhang/commonData/scEpiLock/traning/"

ATAC_labels = pd.read_csv(data_dir + "y-labels-positives.for-trainging.txt", header = 0, sep ='\t')

cell_types  = ["Ast", "End", "Ex" , "In" , "Mic" , "Oli", "OPC"]

ct = []

for cell_type in cell_types:

	ct.append (sum(ATAC_labels[cell_type].values))


print(ct)