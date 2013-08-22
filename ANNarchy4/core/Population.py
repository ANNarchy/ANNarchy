"""
Population.py
"""

import Global 
from ANNarchy4 import generator
from ANNarchy4.core.Variable import Descriptor

class Population(Descriptor):
    """
    Represents a neural network population
    """

    def __init__(self, name, geometry, neuron, debug=False):
        self.debug = debug
        self.geometry = geometry
        self.neuron = neuron
        self.name = name
        self.id = len(Global.populations_)

        Global.populations_.append(self)
        self.generator = generator.Population(self)

    @property
    def size(self):
        """
        number of neurons in the population.
        """
        size = 1

        for i in xrange(len(self.geometry)):
            size *= self.geometry[i]

        return size

    @property
    def width(self):
        """
        width of the population.
        """
        return self.geometry[0]

    @property
    def height(self):
        """
        height of the population.
        """
        return self.geometry[1]    

    @property
    def depth(self):
        """
        depth of the population.
        """        
        return self.geometry[2]

    def rank_from_coordinates(self, w, h, d=0):
        """
        Returns rank corresponding to (w, h, d) coordinates. The depth d is 0 by default.
        """
        return d*self.width*self.height + h * self.width + w

    def coordinates_from_rank(self, rank):
        """
        Returns (w, h, d) coordinates corresponding to rank.
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
                h = rank - coord[1]*self.geometry[1]
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
