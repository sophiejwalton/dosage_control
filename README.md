# dosage_control

Repository for Dosage Control project. Each circuit is contained in its own jupyter notebook and can be run to recreate plots. 
The sandbox folder contains scratch work that was created during the development of this project. 

To run notebooks, the packages listed in requirements.txt must be installed. The instructions for installation are adapted from (http://justinbois.github.io/bootcamp/2020/lessons/l00_configuring_your_computer.html). The bootcamp_utils package is on pip, but we have also included a copy of the code in the folder bootcamp_utils, which was forked from https://github.com/justinbois/bootcamp_utils. We thank Justin Bois for developing this code for the Caltech Python Bootcamp (BE/Bi/NB 203: Intro. to programming for the bio. sciences bootcamp). It can be installed by either using pip (included in the requiremennts file) or through python setup.py install in the bootcamp_utils folder. 

We recommend installing anaconda and using the conda package manager (https://www.anaconda.com/products/individual). Please install the version for python 3.7. 


Once you have installed anaconda, you can install all necessary plotting packages and configure jupyter notebooks through the following: 

conda install -c pyviz holoviz

conda install nodejs

pip install -r requirements.txt

jupyter labextension install --no-build @pyviz/jupyterlab_pyviz

jupyter lab build

