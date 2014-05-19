ANNarchy (Artificial Neural Networks architect) is a simulator for distributed rate-coded or spiking neural networks. The core of the library is written in C++ and distributed using openMP. It provides an interface in Python for the definition of the networks. It is released under the `GNU GPL v2 or later <http://www.gnu.org/licenses/gpl.html>`_.

The latest source code is available at:

    http://bitbucket.org/annarchy/annarchy

The documentation is available online at:

    http://annarchy.readthedocs.org

**Authors**:

* Julien Vitay (julien.vitay@informatik.tu-chemnitz.de). 

* Helge Ülo Dinkelbach (helge-uelo.dinkelbach@informatik.tu-chemnitz.de). 

**Installation**:

* With adminitrator permissions::

    > sudo python setup.py install

* In the home directory::

    > python setup.py install --user
    
* To install it in another repertory /path/to/repertory::

    > export PYTHONPATH=$PYTHONPATH:/path/to/repertory/lib/python2.7/dist-packages
    > python setup.py install --prefix=/path/to/repertory

**Platforms**:

* GNU/Linux

**Dependencies**:

* g++ >= 4.6

* Python 2.7

* Cython >= 0.19

* Setuptools >= 0.6

* Numpy >= 1.8

* Sympy >= 0.7.4

Optionally:

* lxml >= 3.0

* Scipy >= 0.17

* Matplotlib >= 1.1

* PyQtGraph >= 0.9.8
