*************************
Installation of ANNarchy
*************************

ANNarchy is designed to run on GNU/Linux. It relies mostly on a C++ compiler, Cython (C for Python extension) and Python (NumPy, SciPy, Sympy) libraries. Installation on MacOS is theoretically possible if openMP support is disabled, but not tested yet.

Installation on GNU/Linux systems
=============================================
    

Dependencies
--------------------

ANNarchy depends on a number of packages which should be easily accessible on recent GNU/Linux distributions. The safest way to install these dependencies is through your favourite package manager. Older versions of these packages may work but have not been tested.

    * g++ >= 4.6 (4.7 is recommended) 
    * Python == 2.7
    * Cython >= 0.19
    * Setuptools >= 0.6
    * NumPy >= 1.5
    * SymPy >= 0.7.4
    
On Debian/Ubuntu systems, these packages are in the normal repositories under the names (python 2.x version):

    * g++
    * python-dev
    * python-setuptools
    * python-numpy
    * python-scipy 
    * python-lxml
    * python-sympy

.. For python3.x support of ANNarchy the needed packages are:

    * python3-dev
    * python3-setuptools
    * python3-numpy
    * python3-scipy
    * python3-lxml

.. For the GUI version you need further these packages:

    * python-qscintilla2
    * python-opengl
    * python-qt4
    * python-qt4-gl

.. hint: python3-matplotlib seems to be available only for versions above 3.3

Cython is available either as source on www.cython.org or as python package through easy_install or pip (requires super user permissions for global installation) ::

    user@Machine: easy_install cython

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

    > python setup.py install --compiler=mingw32
    
.. **Cython**

.. Cython is available either as source on www.cython.org or as python package through easy_install::

..     > easy_install cython

.. Installation
.. ---------------

.. Once all dependencies are satisfied, simply unpack ANNarchy's source code somewhere, and type::

    > python setup.py install

.. in the top-level directory.
