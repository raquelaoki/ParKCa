# Tutorial 

This tutorial explains how to use the Causal Inference methods in other projects. 
Methods available: 

Use as [reference](https://github.com/raquelaoki/Summer2020MultipleCauses/blob/master/parkca/train.py)

1. The Deconfounder Algorithm
2. BART

```python
from ParKCa.src.bart import BART
model_bart = BART(X_train, X_test, y_train, y_test)
model_bart.fit()
```

4. CEVAE
