************************************
Class Population
************************************

A Population object represents a single population in the Network. `Population` is used as a generic term intended to include layers, columns, nuclei, etc., of cells. Each population is identified through his name and bases on one neuron type. 

Population is a pure Python class, which has an attribute referencing the corresponding C++ Population object through a cython wrapper.

Spatial structure of a population
====================================

The geometry is one of the parameter forwared to population. It represents the spatial structure of the population in space, where neurons are aligned along a 3D grid. The parameters height and depth are optional (they default to 1), so you can define 1D, 2D or 3D geometries with this set of values.

Please note, the geometry has no influence on the simulation as all objects are stored internally in an one dimensional array. The spatial structure might be relevant when accessing individual neuron of a population. However, adding a geometry allows better visualization as well as an easier description of distance-based connectivity patterns.

Available functions
====================================

.. autoclass:: ANNarchy.Population
   :members:
   :show-inheritance:
   :inherited-members:

