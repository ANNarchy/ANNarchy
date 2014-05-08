***********************************
External documentation
***********************************

The core of ANNarchy is written in C++, but is completely interfaced by Python. Knowledge of the Python ecosystem is the only requirement to use ANNarchy.

Python
======================

ANNarchy uses currently the version 2.7 of Python. Python3 will be supported in the near future.

The documentation for the Python2 language can be found on the official `Python <http://docs.python.org/2/>`_ website. It includes the language and library references, description of available packages as well as nice tutorials and FAQ.

A very nice and complete book is freely available on the `Dive into Python <http://www.diveintopython.net/>`_ website in various formats and languages.

Cython
=======================

Cython (project homepage www.cython.org) is an optimised static compiler for both the Python programming language and the extended Cython programming language (based on Pyrex). It renders writing C extensions for Python as easy as Python itself. Although most users won't need knowledge about Cython for using ANNarchy, it may be needed for speeding up consuming functions or for linking with external C/C++ libraries.

NumPy
=======================

Although Python is *batteries included* (meaning many useful libraries are shipped with the Python interpreter), there are several external libraries which may be needed. ANNarchy makes use of the `NumPy <http://numpy.scipy.org/>`_ library for numerical computations, which is part of the SciPy package. It provides an efficient array/matrix representation, linear algebra functions (including matrix operations), statistical analysis, signal processing, etc.  Using Python+SciPy advantageously replaces the proprietary software Matlab(R). Although the interface is in Python, the core operations are written in C, leading to very fast and efficient computations.

Documentation for NumPy and SciPy can be found on the `NumPy <http://numpy.scipy.org/>`_ website, including reference guides and tutorials. Of particular interest is the `NumPy for Matlab Users <http://www.scipy.org/NumPy_for_Matlab_Users>`_ page, which compares the syntax of NumPy with Matlab.

ANNarchy relies on the ``array`` class rather than ``matrix``. Good practice is to import NumPy in your Python code as *np*, to avoid overlapping naming. This is done automatically when importing ANNarchy.

As an example, if you want to create a 10*10 matrix filled with zeros, you simply have to type::

    import numpy as np
    A=np.zeros((10, 10))
    
NumPy is particularly useful when defining connection matrices or accessing variables in ANNarchy.

Matplotlib
=======================

To visualize the results of your simulations, one possibility is the `matplotlib <http://matplotlib.org/>`_ 2D plotting library. It combines nicely with NumPy in order to plot 1D or 2D arrays. Matplotlib is very simple to use, powerful and customizable. It can work in interactive mode and save hardcopies in various formats.

The reference manual, tutorials and many useful examples can be found on the `matplotlib <http://matplotlib.org/>`_ website. Nicolas Rougier has also written a `tutorial for beginners <http://www.loria.fr/~rougier/teaching/matplotlib/>`_.


