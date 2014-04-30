*************************
Installation of ANNarchy
*************************

ANNarchy is designed to run on GNU/Linux. It relies mostly on a C++ compiler, Cython (C for Python extension) and Python (NumPy, Sympy) libraries. Installation on MacOS is theoretically possible, but not tested yet. Installation on Windows is not yet possible.

Download
===========

The source code of ANNarchy can be downloaded on Bitbucket::

    git clone http://bitbucket.org/annarchy/annarchy.git

As ANNarchy is under heavy development, you should update the repository regularly::

    git pull

Installation on GNU/Linux systems
=============================================
   

Dependencies
--------------------

ANNarchy depends on a number of packages which should be easily accessible on recent GNU/Linux distributions. The classical way to install these dependencies is through your favourite package manager. Older versions of these packages may work but have not been tested.

    * g++ >= 4.6 (4.7 or above is recommended) 
    * make >= 3.0
    * python == 2.7 (with the development files)
    * cython >= 0.19
    * setuptools >= 0.6
    * numpy >= 1.8
    * sympy >= 0.7.4
    
Additionally, the following packages are optional but strongly recommended:

    * pyqtgraph >= 0.9.8 (for visualizing the provided examples)
    * matplotlib >= 1.3.0 (for your own visualizations)
    * lxml >= 3.0 (for saving the networks in .xml format)
    * scipy >= 0.17 (for saving the networks in Matlab format)
    
    
The version requirement on Sympy is rather new (as of May 2014) and may not be available on all distributions. The Python packages would benefit strongly from being installed using ``pip`` (to be installed)::

    pip install cython numpy sympy pyqtgraph matplotlib lxml scipy


Installation
---------------


**Global installation**

If you have superuser permissions, you can install ANNarchy in ``/usr/local`` by typing in the top-level directory::

    user@Machine:~/annarchy-4.0$ sudo python setup.py install
    
This simply installs a Python egg in ``/usr/local/lib/python2.7/dist-packages/`` (replace '2.7' with your Python version). 


**Installation in home directory** 

If you want to install ANNarchy in your home directory, type::

    user@Machine:~/annarchy-4.0$ python setup.py install --user
    
The ANNarchy egg will be installed in ``$HOME/.local/lib/python2.7/site-packages/`` (at least on Debian systems) and automatically added to your ``PYTHONPATH``.
        
**Specific installation**

If you want to install ANNarchy in another directory (let's say in ``/path/to/repertory``), you should first set your python path to this directory::

    user@Machine:~/annarchy-4.0$ export PYTHONPATH=$PYTHONPATH:/path/to/repertory/lib/python2.7/site-packages
    
Again, replace '2.7' with your Python version. If this directory does not exist, you should create it now. Don't forget to set this value in your ``~/.profile`` or ``~/.bashrc`` to avoid typing this command before every session. You can then install ANNarchy by typing::

    user@Machine:~/annarchy-4.0$ python setup.py install --prefix=/path/to/repertory
    

    
.. Installation on Windows systems
.. ============================================

.. As usual, dependencies are much more complicated to satisfy on Windows systems than on GNU/Linux. We detail here a procedure which *should* lead to a successful installation. But we recommend to use ANNarchy on UNIX systems.

.. Dependencies
.. ---------------------

.. **C++ compiler** 

.. ANNarchy needs a C++ compiler adapted to your platform. It has been successfully tested on 32 and 64 architectures with the `Microsoft Visual C++ 2012 Express <http://www.microsoft.com/visualstudio/eng/products/visual-studio-2010-express>`_ compiler, available for free (as in beer). Other versions of the compiler should work, but it has not been tested yet.

.. `MinGW (Minimalist GNU for Windows) <http://www.mingw.org/>`_ is another option, as it is a Windows implementation of the GNU gcc compiler, but has not been tested yet. Same story for the Intel C compiler (theoretically better than the other ones, but expensive).

.. In this case you need to attach an argument to the install command:

..    > python setup.py install --compiler=mingw32
    
.. **Cython**

.. Cython is available either as source on www.cython.org or as python package through easy_install::

..     > easy_install cython

.. Installation
.. ---------------

.. Once all dependencies are satisfied, simply unpack ANNarchy's source code somewhere, and type::

..    > python setup.py install

.. in the top-level directory.
