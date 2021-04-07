# Tutorial 

This tutorial explains how to use the Causal Inference methods in other projects. 
Methods available: 

Use as [reference](https://github.com/raquelaoki/Summer2020MultipleCauses/blob/master/parkca/train.py)


1. The Deconfounder Algorithm
```python
from ParKCa.src.deconfounder import deconfounder_algorithm as DA
model_da = DA(X_train, X_test, y_train, y_test, k=10)
coef, coef_continuos, roc = model_da.fit()
```
2. BART

```python
from ParKCa.src.bart import BART
model_bart = BART(X_train, X_test, y_train, y_test)
model_bart.fit()
```

2. CEVAE

```python
from ParKCa.src.cevae import CEVAE 
model_cevae = CEVAE(X_train, X_test, y_train, y_test, TreatCols,
                                binfeats=binfeatures, contfeats=confeatures)
cate = model_cevae.fit_all()
```
