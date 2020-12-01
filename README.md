# dosage_control

Repository for Dosage Control project. Each circuit is contained in its own jupyter notebook and can be run to recreate plots. 


To run notebooks, the packages listed in requirements.txt must be installed. The instructions for installation are adapted from (http://justinbois.github.io/bootcamp/2020/lessons/l00_configuring_your_computer.html). 

We recommend installing anaconda and using the conda package manager (https://www.anaconda.com/products/individual). Please install the version for python 3.7. 


Once you have installed anaconda, you can install all necessary plotting packages and configure jupyter notebooks through the following: 

conda install -c pyviz holoviz

conda install nodejs

pip install -r requirements.txt

jupyter labextension install --no-build @pyviz/jupyterlab_pyviz

jupyter lab build

