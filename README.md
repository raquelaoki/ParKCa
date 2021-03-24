# ParKCa

Causality for Computational Biology

Citation:
Aoki, Raquel, and Martin Ester. "ParKCa: Causal Inference with Partially Known Causes." Pac Symp Biocomput. 2021 ([link](https://arxiv.org/abs/2003.07952))

TODO: Update train.py

## Code Instructions 

### 1. DATA PRE-PROCESSING 

- datapreprocessing.py: simulated dataset generation, pre-processing of cgc list, and other operations  
- script_realdata_preparation.R: real-dataset download, filtering, merging

#### Example: 
```python
n_units = 5000
n_variables = 10000 
n_datasets = 10
dp.generate_samples(n_datasets,n_units, n_variables) 
#output: 10 datasets 5000 x 10000 in pickle format 'data_s//snp_simulated1_0.txt',...,'data_s//snp_simulated1_9.txt'
#+1 dataset 5000 x 10 in pickle format with the simulated targets 'data_s//snp_simulated1_y01.txt'; col 0 has the targets of 'snp_simulated1_0.txt' dataset. 
#+1 dataset 10000 x 10 in pickle format with the true causes 'data_s//snp_simulated1_truecauses.txt'; col 0 has the true causes of 'snp_simulated1_0.txt' dataset. 
```


### 2. LEARNERS  
- bart.R: Fit BART to real-world dataset. Average time for full dataset: 6h 
- cevae.ipynb + CEVAE_FUNCTIONS.py + CEVAE.py: Google Colab notebook. Fit CEVAE for simulated dataset. Average time for full dataset: 10h
- from train.py, learners(APPLICATIONBOOL,DABOOL,path): run the DA for the application and simulated dataset

#### Example: 
```python
train.learners(APPLICATIONBOOL=True,DABOOL=True, path = path)

tcga_train_gexpression_cgc_7k_abr_HNSC.txt :  192
skip HNSC #HNSC is skiped because there are only 192 patients with this cancer type

tcga_train_gexpression_cgc_7k_abr_LGG.txt :  254
Pass Predictive Check: dappcalr_15_LGG ( 0.7182358928313352 )
F1: 0.3333333333333333 10 20
Confusion Matrix [[59  5] [15  5]]

tcga_train_gexpression_cgc_7k_abr_LIHC.txt :  269
Pass Predictive Check: dappcalr_15_LIHC ( 0.7042992933416174 )
F1: 0.3111111111111111 17 28
Confusion Matrix [[51 10] [21  7]]
#F1-score and confusion matrix to predict level 0 target (metastasis) 
```

### 3. META-LEARNERS
- from train.py, meta_learner(data, meta-learners, prob): run the meta-learners in the level 1 dataset + known causes. 
If working on the real-world datset, set prob = 1.

#### Example: 
```python
experiments = train.meta_learner(data,['rf','lr','random','upu','adapter','nn'],0.5)
experiments 

  metalearners  precision    recall       auc        f1       f1_    prfull  refull
0           rf   0.112211  0.202381  0.510435  0.252295  0.144374  0.173531  0.461988   
1           lr   0.102883  0.467262  0.502693  0.126300  0.168636  0.072530  0.488304  
2       random   0.136508  0.127976  0.518104  0.063189  0.132104  0.063516  0.062865 
3          upu   0.103728  0.571429  0.505856  0.127821  0.175583  0.071643  0.592105 
4      adapter   0.101818  1.000000  0.500000  0.128042  0.184818  0.068400  1.000000 
5           nn   0.103462  0.773810  0.506844  0.125644  0.182520  0.068434  0.766082   
6     ensemble   0.103504  0.571429  0.505181  0.125747  0.175262  0.070447  0.584795  

#precision, recall, aux, f1_ were calculated using testing set; 
#prfull, refull, f1 were calculated using the full set.
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
### 5. RUNNING EXPERIMENTS 
- main.py: 

* real-world application: after running script_realdata_preparation.R and bart.R, this code run the DA learner, join the results, add the known causes from cgc, run the meta-learners and save the evaluation outputs. 
* simulatation: run the dp.generate_samples(sim,n_units, n_causes) and cevae.ipynb. Then, this code run the DA learner, join the results, add proportions of known causes, run the meta-learners and save the evaluation outputs. 

