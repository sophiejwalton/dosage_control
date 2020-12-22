# dosage_control

Repository for Dosage Control project. Each circuit is contained in its own jupyter notebook and can be run to recreate plots. 
The sandbox folder contains scratch work that was created during the development of this project. 

The following report describes the work in this project: https://doi.org/10.1101/2020.12.18.423556


Citations for this project can be found in dosage_control_references.pdf. The following reference was especially influential; 

Arthur Prindle, Jangir Selimkhanov, Howard Li, Ivan Razinkov, Lev S. Tsimring, and Jeff
Hasty. Rapid and tunable post-translational coupling of genetic circuits. Nature, 508(7496):
387–391, Apr 2014. ISSN 1476-4687. doi: 10.1038/nature13238.


L. Pasotti, M. Bellato, N. Politi, M. Casanova, S. Zucca, M. G. Cusella De Angelis, and
P. Magni. A synthetic close-loop controller circuit for the regulation of an extracellular
molecule by engineered bacteria. IEEE Transactions on Biomedical Circuits and Systems,
13(1):248–258, 2019. - Most parameters are derived from this reference. 

Susanna Zucca, Lorenzo Pasotti, Nicolò Politi, Michela Casanova, Giuliano Mazzini,
Maria Gabriella Cusella De Angelis, and Paolo Magni. Multi-faceted characterization of a novel luxr-repressible promoter library for escherichia coli. PLOS ONE, 10(5):1–26, 05
2015. doi: 10.1371/journal.pone.0126264.

Xinying Ren and Richard M. Murray. Layered feedback control improves robust functionality
across heterogeneous cell populations. bioRxiv, 2020. doi: 10.1101/2020.03.24.006528.

A. Vignoni, D. A. Oyarzún, J. Picó, and G. . Stan. Control of protein concentrations in heterogeneous
cell populations. pages 3633–3639, 2013. doi: 10.23919/ECC.2013.6669828.


To run notebooks, the packages listed in requirements.txt must be installed. The instructions for installation are adapted from (http://justinbois.github.io/bootcamp/2020/lessons/l00_configuring_your_computer.html). The bootcamp_utils package is on pip, but we have also included a copy of the code in the folder bootcamp_utils, which was forked from https://github.com/justinbois/bootcamp_utils. We thank Justin Bois for developing this code for the Caltech Python Bootcamp (BE/Bi/NB 203: Intro. to programming for the bio. sciences bootcamp). It can be installed by either using pip (included in the requiremennts file) or through python setup.py install in the bootcamp_utils folder. 

We recommend installing anaconda and using the conda package manager (https://www.anaconda.com/products/individual). Please install the version for python 3.7. 


Once you have installed anaconda, you can install all necessary plotting packages and configure jupyter notebooks through the following: 

conda install -c pyviz holoviz

conda install nodejs

pip install -r requirements.txt

jupyter labextension install --no-build @pyviz/jupyterlab_pyviz

jupyter lab build

