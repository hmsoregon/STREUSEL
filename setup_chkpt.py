from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()
  
setup(
  name='streusel',
  version='1.0.13',
  description='Pore volume and surface area calcs for porous materials',
  url='https://github.com/austin-mroz/STREUSEL',
  author='Austin Mroz',
  packages=find_packages(where='streusel'),
  package_dir={'': 'streusel'},
  install_requires=['tqdm','numpy','pandas','ase']
)
