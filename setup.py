from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()
  
setup(
  name='STREUSEL',
  version='1.0.0',
  description='Pore volume and surface area calcs for porous materials',
  url='https://github.com/hmsoregon/STREUSEL',
  author='Austin Mroz',
  classifiers=[
    'License :: MIT License',
    'Programming Language :: 2.7',
    'Programming Language :: 3.6',
  ],
  
  packages=['streusel'],
  install_requires['tqdm','numpy','pandas','os','math','ase','pickle','shutil','time']
)
