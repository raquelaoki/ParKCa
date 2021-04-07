# ParKCa

Causality for Computational Biology

Citation:
Aoki, Raquel, and Martin Ester. "ParKCa: Causal Inference with Partially Known Causes." Pac Symp Biocomput. 2021 ([link](https://arxiv.org/abs/2003.07952))

TODO: Update train.py

## Usage of causal inference methods: 
To use a particular method, check this [tutorial](https://github.com/raquelaoki/ParKCa/blob/new_structure/CausalInferenceMethods.md).


## Code Instructions 

### 1. DATA 

a. Real-world dataset (TODO - Missing TCGA)
b. Simulated dataset ([link](https://github.com/raquelaoki/CompBioAndSimulated_Datasets))

#### Example simulation: 
```python
from CompBioAndSimulated_Datasets.simulated_data_multicause import *

sdata_gwas = gwas_simulated_data(prop_tc=0.05, pca_path='/content/CompBioAndSimulated_Datasets/data/tgp_pca2.txt')
X, y, y01, treatement_columns, treatment_effects  = sdata_gwas.generate_samples()
```


### 2. LEARNERS

The following causal inference methods were implemented:
- BART (very time-consuming for experiments with large number of treatments)
- CEVAE
- Deconfounder Algorithm

#### Example: 
```python
colnamesX = ['Col'+str(i) for i in range(X.shape[1])]
level1data = learners(['CEVAE','DA','BART'],pd.DataFrame(X),y01, TreatCols = None, colnamesX=colnamesX)
```

### 3. META-LEARNERS
- from train.py, meta_learner(data, meta-learners, prob): run the meta-learners in the level 1 dataset + known causes. 
If working on the real-world datset, set prob = 1.

#### Example: 
```python
level1data['y'] = 0
level1data['y'] = [1 if i in known_causes else 0 for i in level1data['causes'].values]

roc, output, y_full_prob = parkca.meta_learner(level1data.fillna(0), ['lr','nn','upu','rf'],'y')
output 

  metalearners   pr_test   re_test   auc_test  f1_test   f1_full   pr_full   re_full
0           rf   0.112211  0.202381  0.510435  0.252295  0.144374  0.173531  0.461988   
1           lr   0.102883  0.467262  0.502693  0.126300  0.168636  0.072530  0.488304  
2       random   0.136508  0.127976  0.518104  0.063189  0.132104  0.063516  0.062865 
3          upu   0.103728  0.571429  0.505856  0.127821  0.175583  0.071643  0.592105 
4      adapter   0.101818  1.000000  0.500000  0.128042  0.184818  0.068400  1.000000 
5           nn   0.103462  0.773810  0.506844  0.125644  0.182520  0.068434  0.766082   
6     ensemble   0.103504  0.571429  0.505181  0.125747  0.175262  0.070447  0.584795  
```

### 4.EVALUATION 
- eval.py: ROC plots, ROC curve for learners evaluation, simulation evaluation (pehe, roc curve to find new causes, diversity, and others)
- eval.R + pickle_reader.py: code to generate the plots in the paper 

#### Example:
```python
qav, q_ = eval.diversity(['cevae' ],['coef'], data) #qav: average value, q_: array with the pairwise diversity
qav
-0.0482939868370808
```
