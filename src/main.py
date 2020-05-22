import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
import time
import matplotlib.pyplot as plt


path = 'C://Users//raoki//Documents//GitHub//ParKCa'
#path = 'C://Users//raque//Documents//GitHub//ParKCa'

sys.path.append(path+'//src')
sys.path.append(path+'//extra')
import datapreprocessing as dp
#import CEVAE as cevae
import train as models
import eval as eval
import numpy.random as npr
from os import listdir
from os.path import isfile, join
os.chdir(path)
import random
randseed = 123
print("random seed: ", randseed)
random.seed(randseed)
np.random.seed(randseed)

pd.set_option('display.max_columns', 500)

from scipy.stats import ttest_ind,ttest_rel

SIMULATION = False
EVALUATION_A = False
EVALUATION_S = True
#cuda test torch.cuda.FloatTensor(2)

'''
Real-world application
level 0 data: gene expression of patients with cancer
level 0 outcome: metastasis
'''
#models.learners(APPLICATIONBOOL=True,DABOOL=True, BARTBOOL=False, CEVAEBOOL=False,path = path)

if EVALUATION_A:
    features_bart =  pd.read_csv("results\\coef_bart.txt",sep=';')
    features_da15 = pd.read_pickle("results\\coef_15.txt")
    features_da15c = pd.read_pickle("results\\coefcont_15.txt")

    level1data = features_bart.merge(features_da15,  left_on='gene', right_on='genes').drop(['genes'],1)
    level1datac = features_bart.merge(features_da15c,  left_on='gene', right_on='genes').drop(['genes'],1)

    cgc_list = dp.cgc('extra\\cancer_gene_census.csv')
    level1data = cgc_list.merge(level1data, left_on='genes',right_on='gene',how='right').drop(['genes'],1)
    level1datac = cgc_list.merge(level1datac, left_on='genes',right_on='gene',how='right').drop(['genes'],1)

    level1data['y_out'].fillna(0,inplace=True)
    level1datac['y_out'].fillna(0,inplace=True)

    level1data.set_index('gene', inplace = True, drop = True)
    level1datac.set_index('gene', inplace = True, drop = True)


    data1 = dp.data_norm(level1data)
    data1c = dp.data_norm(level1datac)

    #DIVERSITY
    qav, q_ = eval.diversity(['bart_all',  'bart_FEMALE',  'bart_MALE' ],
                        ['dappcalr_15_LGG','dappcalr_15_SKCM','dappcalr_15_all','dappcalr_15_FEMALE','dappcalr_15_MALE'],
                        data1)
    print('DIVERSITY: ', qav)
    #Metalearners
    experiments1 = models.meta_learner(data1, ['adapter','upu','lr','rf','random'])
    experiments1c = models.meta_learner(data1c, ['adapter','upu','lr','rf','random'])

    experiments0 = eval.first_level_asmeta(['bart_all',  'bart_FEMALE',  'bart_MALE' ],
                        ['dappcalr_15_LGG','dappcalr_15_SKCM','dappcalr_15_all','dappcalr_15_FEMALE','dappcalr_15_MALE'],
                        data1)
    #models.meta_learner(data1, ['lr'])


    experiments1.to_csv('results\\eval_metalevel1.txt', sep=';')
    experiments1c.to_csv('results\\eval_metalevel1c.txt', sep=';')
    experiments0.to_csv('results\\eval_metalevel0.txt', sep=';')

#more layers and nodes improved
from sklearn.model_selection import train_test_split,  GridSearchCV, StratifiedKFold


count_class_0, count_class_1 = data1.y_out.value_counts()

# Divide by class
df_class_0 = data1[data1['y_out'] == 0]
df_class_1 = data1[data1['y_out'] == 1]

df_class_0_under = df_class_0.sample(4000)
df_class_1_over = df_class_1.sample(4000, replace=True)
data2 = pd.concat([df_class_0_under, df_class_1_over], axis=0)


y = data1['y_out']
X = data1.drop(['y_out'], axis=1)


y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.33,random_state=22)
models.nn_classifier(y_train, y_test, X_train, X_test, 1000,10,0.001)


'''
Simulation
level 0 data: Binary treatments
level 0 outcome: Binary
'''


#SAVING 10 datasets
n_units = 5000
n_causes = 10000# 10% var
sim = 10

#dp.generate_samples(sim,n_units, n_causes)


#CEVAE CODE ON NOTEBOOK - CHANGE NAME
#sim1_cevae_out = dp.join_simulation('results\\simulations\\cevae_output_sim1_',['a','b','c'])
#sim1 = join_simulation('results\\simulations\\cevae_output_sim1_',['a','b','c'])

#DA
pathtc = 'data_s\\snp_simulated1_truecauses.txt'
pathy01 = 'data_s\\snp_simulated1_y01.txt'
tc  = pd.read_pickle(pathtc)
y01 = pd.read_pickle(pathy01)



#RUN AGAIN BECAUSE ROC
dp.sim_level1data([0,1,3],tc,y01,'sim_roc_a')
#CHECK PREDICTIVE CHECK
done = [0,1,3] #pred_check = [0.53]

out_meta = pd.DataFrame(columns=['metalearners', 'precision','recall','auc','f1','f1_','prfull','refull','version'])
out_metac = pd.DataFrame(columns=['metalearners', 'precision','recall','auc','f1','f1_','prfull','refull','version'])
out_level0 = pd.DataFrame(columns=['metalearners', 'precision','recall','auc','f1','f1_','version'])
pehe = pd.DataFrame(columns=['method','pehe_noncausal', 'pehe_causal','pehe_overall','version'])

out_diversity = []

def pehe_calc(true_cause,pred_cause, name,version):
    pehe = [0,0,0]
    count = [0,0,0]
    for j in range(len(true_cause)):
        if true_cause[j]== 0:
            pehe[0] += pow(true_cause[j]-pred_cause[j],2)
            count[0] += 1
        else:
            pehe[1] += pow(true_cause[j]-pred_cause[j],2)
            count[1] += 1

        pehe[2] += pow(true_cause[j]-pred_cause[j],2)
        count[2] += 1
    pehe_ = {'method':name,'pehe_noncausal':pehe[0]/count[0],
         'pehe_causal':pehe[1]/count[1],'pehe_overall':pehe[2]/count[2],
         'version':version}
    return pehe_

for i in done:

    data = pd.read_csv('results\\level1data_sim_'+str(i)+'.txt',sep=';')
    #Meta-learners
    exp1 = models.meta_learner(data.iloc[:,[1,2,3]],['rf','lr','random','upu','adapter'])
    exp1c = models.meta_learner(data.iloc[:,[1,4,3]],['rf','lr','random','upu','adapter'])
    exp0 = eval.first_level_asmeta(['cevae' ],['coef'],data.iloc[:,[1,2,3]])

    exp1['version'] = str(i)
    exp1c['version'] = str(i)
    exp0['version'] = str(i)

    qav, q_ = eval.diversity(['cevae' ],['coef'], data.iloc[:,[1,2,3]])


    #Continuos
    #pehe = pd.DataFrame(columns=['method','pehe_noncausal', 'pehe_causal','pehe_overall'])

    X = np.matrix(data.iloc[:,[1,4]])
    y = np.array(data.iloc[:,5])
    y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.33,random_state=22)

    #model = sm.Logit(y,X).fit_regularized(method='l1')
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_full = model.predict(X)

    pehe_ = pehe_calc(np.array(data.iloc[:,5]), np.array(data.iloc[:,1]),'CEVAE',str(i))
    pehe = pehe.append(pehe_,ignore_index=True)
    pehe_ = pehe_calc(np.array(data.iloc[:,5]), np.array(data.iloc[:,2]),'DA',str(i))
    pehe = pehe.append(pehe_,ignore_index=True)
    pehe_ = pehe_calc(np.array(data.iloc[:,5]),y_full,'Meta-learner (Full set)',str(i))
    pehe = pehe.append(pehe_,ignore_index=True)
    pehe_ = pehe_calc(y_test,y_pred,'Meta-learner (Testing set)',str(i))
    pehe = pehe.append(pehe_,ignore_index=True)


    out_meta = out_meta.append(exp1,ignore_index=True)
    out_metac = out_meta.append(exp1c,ignore_index=True)
    out_level0 = out_level0.append(exp1c,ignore_index=True)
    out_diversity.append(qav)


diversity = pd.DataFrame({'diversity':out_diversity})
diversity['version'] = done

out_meta.to_csv('results\\eval_sim_metalevel1.txt', sep=';')
out_metac.to_csv('results\\eval_sim_metalevel1c.txt', sep=';')
out_level0.to_csv('results\\eval_sim_metalevel0.txt', sep=';')
diversity.to_csv('results\\eval_sim_diversity.txt', sep=';')
pehe.to_csv('results\\eval_sim_pehe.txt', sep=';')
