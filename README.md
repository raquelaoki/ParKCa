# ParKCa

Causality for Computational Biology

## Env
* conda create --name parkca python=3.7
* conda activate parkca
* conda remove --name parkca --all

### Packages: 
* conda install git
* conda install -c anaconda pandas
* conda install -c anaconda numpy
* conda install -c anaconda scikit-learn
* conda install -c conda-forge matplotlib
* conda install -c anaconda seaborn
* conda install -c anaconda statsmodels
* conda install -c conda-forge scikit-allel
* conda install -c conda-forge progressbar
* conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
* conda install pytorch=0.4.1 cuda92 -c pytorch (Old GPU)

Optional:
* pip install -e "git+https://github.com/google/edward2.git#egg=edward2"
* conda install -c conda-forge tensorflow=1.15
* conda install -c conda-forge tensorflow-probability

