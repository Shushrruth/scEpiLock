import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_dir = "/gpfs/gibbs/pi/girgenti/JZhang/commonData/scEpiLock/traning/"
ATAC_labels = pd.read_csv(data_dir + "y-labels-positives.for-trainging.txt", header = 0, sep ='\t')

cell_types  = ["Ast", "End", "Ex" , "In" , "Mic" , "Oli", "OPC"]

ct = []

for cell_type in cell_types:

	ct.append (sum(ATAC_labels[cell_type].values))


#print(ct)

# Generate the plot // Cell type count 

fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(cell_types, ct, color ='maroon',
        width = 0.4)
 
plt.xlabel("Cell Type")
plt.ylabel("Count")
#plt.title("Data countsplit")

plt.savefig('Test/test_slrum_check.png', bbox_inches='tight')