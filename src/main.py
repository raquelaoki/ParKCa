import sys
import os
import pandas as pd

path = path = 'C:\\Users\\raoki\\Documents\\GitHub\\ParKCa'
sys.path.append(path+'\\scr')
import data_preprocessing as data
os.chdir(path)
pd.set_option('display.max_columns', 500)

APPLICATION1 = True #driver genes application
PREPROCESSING = True
if PREPROCESSING and APPLICATION1:
    df_ge, df_cl = data.load_GE('data\\tcga_rna_old.txt','data\\tcga_cli_old.txt')

if APPLICATION1:
    train,colnames, y01, cli = data.read_GE('data\\train_ge.txt','data\\train_cl.txt')
#exec(open("main.py").read())
