This is the documentation for the current version of ANNarchy.

The documentation can be generated using the `Sphinx <http://sphinx-doc.org/>`_ documentation tool (version 1.1 or above).

It can be simply installed through pip, with the relevant theme::

	pip install sphinx sphinx_bootstrap_theme

The examples are displayed as IPython/Jupyter notebooks. You have to install it, plus `pandoc` and the python module `nbsphinx`::

    pip install nbsphinx

Once Sphinx is installed, you only need to use the provided Makefile in the current directory.

* To generate the html version of the documentation, just type::

	make html

It will be accessible in ``_build/html/``.

* To generate the pdf version of the documentation (requires a LaTeX distribution), just type::

	make latexpdf

It will be accessible in ``_build/latex/``.

* Other output formats are possible, check the Makefile or type::

	make help

