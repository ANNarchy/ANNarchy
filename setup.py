#!/usr/bin/env python

################################################
# Check if the required packages are installed
################################################
from pkg_resources import parse_version

# check python version
import sys
if not sys.version_info[:2] >= (2, 6):
    print('Error : ANNarchy requires at least Python 2.6.')
    exit(0) 

# setuptools
try:
    from setuptools import setup, find_packages
    print('Checking for setuptools... OK')
except:
    print('Checking for setuptools... NO')
    print('Error : Python package "setuptools" is required.')
    print('You can install it from: http://pypi.python.org/pypi/setuptools')
    exit(0)

# sympy
try:
    import sympy
    
    if parse_version(sympy.__version__) > parse_version('0.7.4'):
        print('Checking for sympy... OK')
    else:
        print parse_version(sympy.__version__), 'is not sufficient, expected >= 0.7.4' 
except:
    print('Checking for sympy... NO')
    print('Error : Python package "sympy" is required.')
    exit(0)
    
# numpy
try:
    import numpy
    print('Checking for numpy... OK')
except:
    print('Checking for numpy... NO')
    print('Error : Python package "numpy" is required.')
    print('You can install it from: http://www.numpy.org')
    exit(0)

# scipy
try:
    import scipy
    print('Checking for scipy... OK')
except:
    print('Checking for scipy... NO')
    print('Warning : Python package "scipy" is needed by some functions, but not required.')
    print('You can install it from: http://www.scipy.org')

################################################
# Perform the installation
################################################
print('Installing ANNarchy on your system')
setup(  name='ANNarchy',
		version='4.1.0.gamma',
		license='GPLv2 or later',
		platforms='GNU/Linux',
		description='Artificial Neural Networks architect',
		long_description='ANNarchy (Artificial Neural Networks architect) is a simulator for distributed mean-rate neural networks. The core of the library is written in C++ and distributed using openMP. It provides an interface in Python for the definition of the networks.',
		author='Julien Vitay and Helge Uelo Dinkelbach',
		author_email='julien.vitay@informatik.tu-chemnitz.de',
		url='http://www.tu-chemnitz.de/informatik/KI/projects/ANNarchy/index.php',
        packages=find_packages(),
        package_data={'ANNarchy': ['data/compile.sh', 'data/compiled.sh', 'data/setup.py', 'data/core/*', 'data/openmp/*', 'data/pyx/*', 'core/cy_functions.pyx']}
 )

