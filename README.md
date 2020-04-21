# ParKCa
Causality for Computational Biology
conda create --name parkca python=3.7
conda activate parkca
conda remove --name parkca --all

conda install git
pip install "git+https://github.com/google/edward2.git#egg=edward2"
pip install -e "git+https://github.com/google/edward2.git#egg=edward2"
conda install -c conda-forge tensorflow


import edward2 as ed

normal_rv = ed.Normal(loc=0., scale=1.)
