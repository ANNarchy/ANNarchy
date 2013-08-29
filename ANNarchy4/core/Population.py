"""
Population.py
"""
import Global 
from ANNarchy4 import generator
from ANNarchy4.core.Variable import Descriptor

import traceback
import numpy as np

class Population(Descriptor):
    """
    Represents a neural network population in ANNarchy.
    """

    def __init__(self, geometry, neuron, name=None, debug=False):
        """
        Constructor.
        
        Parameter:
        
        * *name*: unique name of the population.
        * *geometry*: population geometry as tuple (width, height, depth)
        * *neuron*: instance of ``ANNarchy4.Neuron``
        * *debug*: print some debug information to standard out ( by default *False* )
        """
        self.debug = debug
        self.pop_geometry = geometry
        self.neuron = neuron
        self.id = len(Global._populations)
        if name:
            self.name = name
        else:
            self.name = 'Population_'+str(self.id)
        

        Global._populations.append(self)
        self.generator = generator.Population(self)

    @property
    def variables(self):
        """
        Returns a list of all variable names.
        """
        neur_var = self.generator.neuron_variables
        ret_var=[] 
        
        for var in neur_var:
            if 'var' in var.keys():
                ret_var.append(var['name'])
        
        return ret_var

    @property
    def parameters(self):
        """
        Returns a list of all parameter names.
        """
        neur_var = self.generator.neuron_variables
        ret_par=[] 
        
        for var in neur_var:
            if not 'var' in var.keys():
                ret_par.append(var['name'])
        
        return ret_par
        
    @property
    def size(self):
        """
        Number of neurons in the population.
        """
        size = 1

        for i in xrange(len(self.pop_geometry)):
            size *= self.pop_geometry[i]

        return size

    @property
    def width(self):
        """
        Width of the population.
        """
        return self.pop_geometry[0]

    @property
    def height(self):
        """
        Height of the population.
        """
        return self.pop_geometry[1]    

    @property
    def depth(self):
        """
        Depth of the population.
        """        
        return self.pop_geometry[2]

    @property
    def geometry(self):
        """
        Geometry of the population as tuple (w, h, d).
        """
        return self.pop_geometry

    @property
    def dimension(self):
        """
        Dimension of the population (1, 2 or 3)
        """
        if self.depth != 1:
            return 3
        if self.height != 1:
            return 2
        else: 
            return 1
        
    def get_variable(self, variable):
        """
        Returns the value of the given variable for all neurons in the population, as a NumPy array having the same geometry as the population.
        
        Parameter:
        
            * *variable*: should be a string representing the variables's name.
        """
        
        if hasattr(self, variable):
            var = eval('self.'+variable)
            return self._reshape_vector(var)
        else:
            print 'Error: variable',variable,'does not exist in this population.'
            print traceback.print_stack()

    def get_parameter(self, parameter):
        """
        Returns the value of the given variable for all neurons in the population, as a NumPy array having the same geometry as the population.
        
        Parameter:
        
            * *parameter*: should be a string representing the variables's name.
        """
        
        if hasattr(self, parameter):
            return eval('self.'+parameter)
        else:
            print 'Error: parameter',parameter,'does not exist in this population.'
            print traceback.print_stack()
    
    def rank_from_coordinates(self, w, h, d=0):
        """
        Returns rank corresponding to (w, h, d) coordinates. The depth d is 0 by default.
        """
        return d*self.width*self.height + h * self.width + w

    def coordinates_from_rank(self, rank):
        """
        Returns a tuple (w, h, d) represents the spatial coordinates corresponding to rank.
        """
        coord = [0, 0, 0]

        if(self.depth==1):
            if(self.height==1):
                coord[0] = rank
            else:
                coord[1] = rank / self.width
                coord[0] = rank - coord[1]*self.height
        else:
            coord[2] = rank / ( self.width * self.height )

            plane_rank = rank - coord[2] * ( self.width * self.height )
            coord[1] = plane_rank / self.width
            coord[0] = plane_rank - coord[1]*self.height

        return coord

    def normalized_coordinates_from_rank(self, rank, norm=1):
        """
        Returns (w, h, d) coordinates normalized to norm corresponding to rank.
        """
        
        coord = [0, 0, 0]

        if(self.depth==1):
            if(self.height==1):
                coord[0] = rank/(self.size()-norm)
            else:
                w = rank / self.width
                h = rank - coord[1]*self.height
                coord[0] = w / (self.width-norm)
                coord[1] = h / (self.height-norm)
        else:
            d = rank / ( self.width * self.height )
            #coord in plane
            pRank = rank - coord[2] * ( self.width * self.height )
            h = pRank / self.width
            w = pRank - coord[1]*self.height

            coord[0] = w / (self.width-norm)
            coord[1] = h / (self.height-norm)
            coord[2] = d / (self.depth-norm)

        return coord

    def set(self, **keyValueArgs):
        """
        update neuron variable/parameter definition
        """
        self.neuron.set(keyValueArgs)
        
    def _reshape_vector(self, vector):
        """
        Transfers a list or a 1D np.array (indiced with ranks) into the correct 1D, 2D, 3D np.array
        """
        vec = np.array(vector) # if list transform to vec
        
        if self.dimension == 1:
            return vec
        elif self.dimension == 2:
            return vec.reshape(self.height, self.width)
        elif self.dimension == 3:
            return vec.reshape(self.depth, self.height, self.width)

