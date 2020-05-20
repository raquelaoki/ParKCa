#plots

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set()
from ast import literal_eval
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix,f1_score
sns.palplot(sns.color_palette("colorblind", 10))



def roc_table_creation(filenames,modelname):
    '''
    Read from roc_table.txt
    From predicted values, create roc_table
    "obs";"pred"
    '''
    roc_table = pd.DataFrame(columns=['learners', 'fpr','tpr','auc'])

    for f in filenames:
        table = pd.read_csv(f,sep=';')
        y_pred = 1-table['pred']
        y_ = table['obs']
        y_pred01 = [1 if i>=0.5 else 0 for i in y_pred]
        print('F1:',f1_score(y_,y_pred01),sum(y_pred01),sum(y_))
        fpr, tpr, _ = roc_curve(y_,y_pred)
        auc = roc_auc_score(y_,y_pred)
        name =  f.replace('.txt', '')
        name = name.replace('results//','')
        roc = {'learners': name,
               'fpr':fpr,
               'tpr':tpr,
               'auc':auc}
        roc_table = roc_table.append(roc,ignore_index=True)
    roc_table.to_pickle('results//roc_'+modelname+'.txt')

def roc_plot(filename):
    '''
    From filenama with roc table with tp,tn, acc and others, fit the plot
    '''
    #https://abdalimran.github.io/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot
    #roc_table1 = pd.read_csv(filename,delimiter=';')
    #coef_table1 = pd.read_csv('results\\coef_15.txt',delimiter=',')
    #roc_table1.columns = ['learners','fpr','tpr','auc']
    roc_table1 = pd.read_pickle(filename)
    roc_table1.set_index('learners', inplace=True)

    fig = plt.figure(figsize=(8,6))

    for i in roc_table1.index:
        label = i.replace('dappcalr','da')
        plt.plot(roc_table1.loc[i]['fpr'],
                 roc_table1.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(label, roc_table1.loc[i]['auc']))

    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':12}, loc='lower right', ncol=1)

    plt.show()
    fig.savefig('results//plots_realdata//plot_'+filename.split('//')[-1].split('.')[0]+'.png')



def roc_plot_all(filenames):
    '''
    From filenama with roc table with tp,tn, acc and others, fit the plot
    '''
    #https://abdalimran.github.io/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot

    for f in filenames:
        roc_table0 = pd.read_pickle(f)
        if f==filenames[0]:
            roc_table1 = roc_table0
        else:
            roc_table1 = pd.concat([roc_table1, roc_table0], axis = 0)

    roc_table1.set_index('learners', inplace=True)

    fig = plt.figure(figsize=(8,6))

    roc_table1.index

    for i in roc_table1.index:
        label = i.replace('dappcalr','da')
        plt.plot(roc_table1.loc[i]['fpr'],
                 roc_table1.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(label, roc_table1.loc[i]['auc']))

    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right', ncol=1)

    plt.show()
    fig.savefig('results//plots_realdata//plot_roc_all_realdata.png')

def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()

def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


def first_level_asmeta(colb,colda, data1):
    #Learners   
    y = data1['y_out']
    X = data1.drop(['y_out'], axis=1)
    from sklearn.model_selection import train_test_split,  GridSearchCV, StratifiedKFold
    y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.33,random_state=22)
    
    #BART
    #col = ['bart_all',  'bart_FEMALE',  'bart_MALE' ]
    for i in colb: 
        X_train[i] = np.abs(X_train[i])
        q = X_train[i].quantile(0.9)
        X_train[i] = [1 if j>q else 0 for j in X_train[i]]
        X_test[i] = [1 if j>q else 0 for j in X_test[i]]
    
    
    #DA
    #col = ['dappcalr_15_LGG','dappcalr_15_SKCM','dappcalr_15_all','dappcalr_15_FEMALE','dappcalr_15_MALE']
    for i in colda: 
        X_train[i] = [1 if j!=0 else 0 for j in X_train[i]]
        X_test[i] = [1 if j!=0 else 0 for j in X_test[i]]
        
    roc_table = pd.DataFrame(columns=['metalearners', 'precision','recall','auc','f1','f1_'])
    for i in X_train.columns:            
        pr = precision(1,confusion_matrix(y_test,X_test[i]))
        re = recall(1,confusion_matrix(y_test,X_test[i]))
        auc = roc_auc_score(y_test,X_test[i])        
        f1_ = f1_score(y_test,X_test[i])
        
        y_full = np.hstack([X_test[i],X_train[i]])
        f1 = f1_score(np.hstack([y_test,y_train]),y_full)

        roc = {'metalearners': i,'precision':pr ,'recall':re,'auc':auc,'f1':f1,'f1_':f1_}
        roc_table = roc_table.append(roc,ignore_index=True)
        
    return roc_table

#exp_plot('results\\cgc_baselines.txt','results\\eval_metalevel1.txt','results\\eval_metalevel0.txt')
#path_baselines, path_ex0, path_ex1 = 'results\\cgc_baselines.txt','results\\eval_metalevel1.txt','results\\eval_metalevel0.txt'

