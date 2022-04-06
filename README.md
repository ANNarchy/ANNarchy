# ANNarchy 

[![DOI](https://zenodo.org/badge/57382690.svg)](https://zenodo.org/badge/latestdoi/57382690)


ANNarchy (Artificial Neural Networks architect) is a parallel and hybrid simulator for distributed rate-coded or spiking neural networks. The core of the library is written in C++ and distributed using openMP or CUDA. It provides an interface in Python for the definition of the networks. It is released under the [GNU GPL v2 or later](http://www.gnu.org/licenses/gpl.html).

The source code is available at:

<http://bitbucket.org/annarchy/annarchy>

The documentation is available online at:

<http://annarchy.readthedocs.io>

A forum for discussion is set at:

<https://groups.google.com/forum/#!forum/annarchy>

### Citation

If you use ANNarchy for your research, we would appreciate if you cite the following paper:

> Vitay J, Dinkelbach HÜ and Hamker FH (2015). ANNarchy: a code generation approach to neural simulations on parallel hardware. *Frontiers in Neuroinformatics* 9:19. [doi:10.3389/fninf.2015.00019](http://dx.doi.org/10.3389/fninf.2015.00019)

### Authors

* Julien Vitay (julien.vitay@informatik.tu-chemnitz.de).
* Helge Ülo Dinkelbach (helge-uelo.dinkelbach@informatik.tu-chemnitz.de).
* Fred Hamker (fred.hamker@informatik.tu-chemnitz.de).


## Installation

Using pip, you can install the latest stable release:

```
pip install ANNarchy
```

## Platforms

* GNU/Linux
* MacOS X (with limitations)

## Dependencies

* g++ >= 4.8 or clang++ >= 3.4
* python >= 3.6 with development files
* cython >= 0.20
* setuptools >= 40.0
* numpy >= 1.13
* sympy >= 1.6
* scipy >= 0.19

Recommended:

* matplotlib
* lxml 
* PyQtGraph 
* pandoc 
* tensorboardX
