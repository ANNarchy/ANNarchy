*************************
Installation of ANNarchy
*************************

ANNarchy is designed to run on GNU/Linux and OSX. It relies mostly on a C++ compiler, Cython (C for Python extension) and Python (NumPy, Sympy) libraries. Installation on Windows is not yet possible.

Download
===========

The source code of ANNarchy can be downloaded on Bitbucket::

    git clone http://bitbucket.org/annarchy/annarchy.git

As ANNarchy is under heavy development, you should update the repository regularly::

    git pull origin master

Installation on GNU/Linux systems
=============================================
   
Dependencies
--------------------

ANNarchy depends on a number of packages which should be easily accessible on recent GNU/Linux distributions. The classical way to install these dependencies is through your favourite package manager. Older versions of these packages may work but have not been tested.

    * g++ >= 4.6 (4.7 or above is recommended) 
    * make >= 3.0
    * python == 2.7 or >= 3.3 (with the development files)
    * cython >= 0.19
    * setuptools >= 0.6
    * numpy >= 1.8
    * sympy >= 0.7.4
    
Additionally, the following packages are optional but strongly recommended:

    * pyqtgraph >= 0.9.8 (to visualize the provided examples)
    * matplotlib >= 1.3.0 (for your own visualizations)
    * lxml >= 3.0 (to save the networks in .xml format)
    * scipy >= 0.12 (to save the networks in Matlab format)
    
For CUDA:

    * the CUDA-SDK is available on the official website: https://developer.nvidia.com/cuda-downloads (we recommend to use at least a SDK version > 6.x)
    
The version requirement on Sympy is rather newand may not be available on all distributions. The Python packages would benefit strongly from being installed using ``easy_install`` (provided by setuptools) or ``pip`` (to be installed through ``setuptools``)::

    sudo easy_install pip
    sudo pip install cython numpy sympy pyqtgraph matplotlib lxml scipy
    
.. note::

     On fresh installs of Ubuntu 14.04 and **Linux Mint Debian Edition** 64 bits, the following commands successfully install everything you need::
     
        sudo apt-get install g++ gfortran git python-dev python-setuptools \
            python-numpy python-scipy python-matplotlib cython python-opengl \
            python-qt4-gl python-lxml python-pip python-tk

        sudo pip install sympy
        
        wget http://www.pyqtgraph.org/downloads/python-pyqtgraph_0.9.8-1_all.deb
        
        sudo dpkg -i python-pyqtgraph_0.9.8-1_all.deb
        
    You should replace the PyQtGraph version with the latest one on `pyqtgraph.org <www.pyqtgraph.org>`_.


Installation
---------------

Installation of ANNarchy is possible through one of the three following methods: 

**Local installation in home directory** 

If you want to install ANNarchy in your home directory, type::

    user@Machine:~/annarchy-4.0$ python setup.py install --user
    
The ANNarchy egg will be installed in ``$HOME/.local/lib/python2.7/site-packages/`` (at least on Debian systems) and automatically added to your ``PYTHONPATH``.


**Global installation**

If you have superuser permissions, you can install ANNarchy in ``/usr/local`` by typing in the top-level directory::

    user@Machine:~/annarchy-4.0$ sudo python setup.py install
    
This simply installs a Python egg in ``/usr/local/lib/python2.7/dist-packages/`` (replace '2.7' with your Python version). 

        
**Specific installation**

If you want to install ANNarchy in another directory (let's say in ``/path/to/repertory``), you should first set your Python path to this directory::

    user@Machine:~/annarchy-4.0$ export PYTHONPATH=$PYTHONPATH:/path/to/repertory/lib/python2.7/site-packages
    
Again, replace '2.7' with your Python version. If this directory does not exist, you should create it now. Don't forget to set this value in your ``~/.profile`` or ``~/.bashrc`` to avoid typing this command before every session. You can then install ANNarchy by typing::

    user@Machine:~/annarchy-4.0$ python setup.py install --prefix=/path/to/repertory
    

.. note::

    Sometimes the detection of CUDA fails during installation as some environment variables are not set. In this case try::
    
        sudo env "PATH=$PATH" "LIBRARY_PATH=$LIBRARY_PATH" python setup.py ...


Installation on OSX systems
============================

.. note::

    Installation should be similar to Linux. ANNarchy should be able to use clang instead of gcc. Beware that OpenMP is not available by default on OSX...
