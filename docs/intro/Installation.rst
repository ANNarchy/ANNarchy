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

ANNarchy depends on a number of packages which should be easily accessible on recent GNU/Linux distributions. The classical way to install these dependencies is through your package manager, or using full Python distributions such as Anaconda. Older versions of these packages may work but have not been tested.

* g++ >= 4.6 (4.7 or above is recommended) 
* make >= 3.0
* python == 2.7 or >= 3.3 (with the development files, e.g. ``python-dev`` or ``python-devel``)
* cython >= 0.19
* setuptools >= 0.6
* numpy >= 1.8
* sympy >= 0.7.4
* scipy >= 0.12
    
Additionally, the following packages are optional but strongly recommended:

* pyqtgraph >= 0.9.8 (to visualize some of the provided examples)
* matplotlib >= 1.3.0 (for the rest of the visualizations)
* lxml >= 3.0 (to save the networks in .xml format)
    
To use the CUDA backend:

* the CUDA-SDK is available on the official `website <https://developer.nvidia.com/cuda-downloads>`_ (we recommend to use at least a SDK version > 6.x)
    
The version requirement on Sympy is rather new and may not be available on all distributions. The Python packages would benefit strongly from being installed using ``easy_install`` (provided by setuptools) or ``pip`` (to be installed through ``setuptools``)::

    sudo easy_install pip
    sudo pip install cython numpy sympy pyqtgraph matplotlib lxml scipy
    
.. note::

     On fresh installs of Ubuntu 14.04 and **Linux Mint Debian Edition** 64 bits, the following commands successfully install everything you need::
     
        sudo apt-get install g++ gfortran git python-dev python-setuptools \
            python-numpy python-scipy python-matplotlib cython python-opengl \
            python-qt4-gl python-lxml python-pip python-tk

        sudo pip install sympy pyqtgraph


Installation
---------------

Using pip
_________

Stable releases of ANNarchy are available on PyPi::

    pip install ANNarchy

or::

    pip install ANNarchy --user

if you do not have administrator permissions.

Using the source code
______________________

Installation of ANNarchy is possible through one of the three following methods: 

**Local installation in home directory** 

If you want to install ANNarchy in your home directory, type::

    python setup.py install --user
    
The ANNarchy egg will be installed in ``$HOME/.local/lib/python2.7/site-packages/`` (at least on Debian systems) and automatically added to your ``PYTHONPATH``.


**Global installation**

If you have superuser permissions, you can install ANNarchy in ``/usr/local`` by typing in the top-level directory::

    sudo python setup.py install
    
This simply installs a Python egg in ``/usr/local/lib/python2.7/dist-packages/`` (replace '2.7' with your Python version). 

        
**Specific installation**

If you want to install ANNarchy in another directory (let's say in ``/path/to/repertory``), you should first set your Python path to this directory::

    export PYTHONPATH=$PYTHONPATH:/path/to/repertory/lib/python2.7/site-packages
    
Again, replace '2.7' with your Python version. If this directory does not exist, you should create it now. Don't forget to set this value in your ``~/.bash_profile`` or ``~/.bashrc`` to avoid typing this command before every session. You can then install ANNarchy by typing::

    python setup.py install --prefix=/path/to/repertory
    

.. note::

    Sometimes the detection of CUDA fails during installation as some environment variables are not set. In this case try::
    
        sudo env "PATH=$PATH" "LIBRARY_PATH=$LIBRARY_PATH" python setup.py ...


If you have multiple Python installations on your system (e.g. through Anaconda), you should not forget to update your ``LD_LIBRARY_PATH`` environment variable in ``.bashrc`` or ``bash_profile`` to point at the location of ``libpython2.7.so``::

    export LD_LIBRARY_PATH=$HOME/anaconda2/lib:$LD_LIBRARY_PATH

Installation on MacOS X systems
================================

Installation on MacOS X is in principle similar to GNU/Linux::

    python setup.py install (--user or --prefix)


We advise using a full Python distribution such as `Anaconda <https://www.continuum.io/why-anaconda>`_, which installs automatically all dependencies of ANNarchy, rather than using the old python provided by Apple.

The only problem with Anaconda (and potentially other Python distributions, not tested) is that the compiler will use by default the Python shared library provided by Apple, leading to the following crash when simulating::

    Fatal Python error: PyThreadState_Get: no current thread
    Abort trap: 6

The solution is to set the environment variable ``DYLD_FALLBACK_LIBRARY_PATH`` to point at the correct library ``libpython2.7.dylib`` in your ``.bash_profile``. For a standard Anaconda installation, this should be::

    export DYLD_FALLBACK_LIBRARY_PATH=$HOME/anaconda/lib:$DYLD_FALLBACK_LIBRARY_PATH

.. note::

    The default compiler on OS X is clang-llvm. You should install the *command_line_tools* together with XCode in order to use it.

    For some reasons, this compiler is not compatible with OpenMP, so the models will only run sequentially.
