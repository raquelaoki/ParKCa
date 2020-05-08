#plots

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set()
from ast import literal_eval


#read from roc_table

def roc_plot(filename):
    #https://abdalimran.github.io/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot
    #roc_table1 = pd.read_csv(filename,delimiter=';')
    #coef_table1 = pd.read_csv('results\\coef_15.txt',delimiter=',')
    #roc_table1.columns = ['learners','fpr','tpr','auc']
    
    roc_table1 = pd.read_pickle(filename)
    roc_table1.set_index('learners', inplace=True)
    
    fig = plt.figure(figsize=(8,6))
    
    for i in roc_table1.index:
        plt.plot(roc_table1.loc[i]['fpr'], 
                 roc_table1.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, roc_table1.loc[i]['auc']))
      

      
    plt.plot([0,1], [0,1], color='orange', linestyle='--')
    
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)
    
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    
    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    
    plt.show()
    fig.savefig('results//plot_'+filename.split('//')[-1].split('.')[0]+'.png')

