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
    print('You can install it from pip or: http://pypi.python.org/pypi/setuptools')
    exit(0)

# sympy
try:
    import sympy
    
    if parse_version(sympy.__version__) > parse_version('0.7.4'):
        print('Checking for sympy... OK')
    else:
        print 'Sympy', sympy.__version__, 'is not sufficient, expected >= 0.7.4' 
except:
    print('Checking for sympy... NO')
    print('Error : Python package "sympy" is required.')
    print('You can install it from pip or: http://www.sympy.org')
    exit(0)
    
# numpy
try:
    import numpy
    print('Checking for numpy... OK')
except:
    print('Checking for numpy... NO')
    print('Error : Python package "numpy" is required.')
    print('You can install it from pip or: http://www.numpy.org')
    exit(0)

# cython
try:
    import cython
    from Cython.Build import cythonize
    print('Checking for cython... OK')

except:
    print('Checking for cython... NO')
    print('Error : Python package "cython" is required.')
    print('You can install it from pip or: http://www.cython.org')
    exit(0)

# extend installer class to support makefiles
from setuptools.command.install import install
from setuptools.command.develop import develop

def cuda_supported():
    import os
    if os.system("which nvcc")==0:
        return True
    else:
        return False

class MyInstall(install):

    def run(self):
        install.run(self)
        print("\n\n\n\nI did it!!!!\n\n\n\n")

class MyDevelop(develop):

    def run(self):
        if cuda_supported():
            import os
            cwd = os.getcwd()
            print("Building cuda_check ...")
            os.chdir(cwd+"/ANNarchy/generator/CudaCheck")
            os.system("make clean && make")
            os.chdir(cwd)

        print("Building ANNarchy ...")
        develop.run(self)

################################################
# Perform the installation
################################################
print('Installing ANNarchy on your system')
setup(  name='ANNarchy',
        version='4.3.2',
        license='GPLv2 or later',
        platforms='GNU/Linux',
        description='Artificial Neural Networks architect',
        long_description='ANNarchy (Artificial Neural Networks architect) is a simulator for distributed rate-coded or spiking neural networks. The core of the library is written in C++ and distributed using openMP. It provides an interface in Python for the definition of the networks.',
        author='Julien Vitay and Helge Uelo Dinkelbach',
        author_email='julien.vitay@informatik.tu-chemnitz.de',
        url='http://www.tu-chemnitz.de/informatik/KI/projects/ANNarchy/index.php',
        packages=find_packages(),
        package_data={'ANNarchy': ['data/core/*', 'core/cython_ext/*.pxd']},
        ext_modules = cythonize(
            [   "ANNarchy/core/cython_ext/Connector.pyx", 
                "ANNarchy/core/cython_ext/Coordinates.pyx",
                "ANNarchy/core/cython_ext/Transformations.pyx" ]
        ),
        include_dirs=[numpy.get_include()],
        zip_safe = False,
        cmdclass={'develop': MyDevelop,
                  'install': MyInstall },
)

