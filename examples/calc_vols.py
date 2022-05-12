import streusel
from streusel.gaussian_cube import *
from tqdm import tqdm
import pandas as pd
import glob
import os

df = pd.DataFrame()

for mpath in glob.glob('*.cube'):
  el = mpath.split('/')
  molecule = el[len(el)-2]
  print(molecule)
  m = el[len(el)-1]
  print(m)
 
  mol = Molecule(mpath)
  vecs = mol.vecs
  ngs = mol.ngs
  mol.get_efield()
  mol.sample_efield()
  streusel = mol.vol
  print('volume is -------')
  print(streusel) 
  row = pd.DataFrame(data=[[molecule, streusel]])
  tempdf = pd.concat([df, row], axis=0)
  df = pd.DataFrame(data=tempdf)
  
  print('--------------------------------------------------------------------------')

df.columns = ['molecule','vol']
print(df)
df.to_csv('final_volumes.csv', index=False)


