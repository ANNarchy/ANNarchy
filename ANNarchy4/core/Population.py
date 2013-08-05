"""
Population.py
"""

# core packages
import Global 

# others
from ANNarchy4 import generator

class Population(object):
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

    def size(self):
        """
        Returns number of neurons in the population.
        """
        size = 1

        for i in xrange(len(self.geometry)):
            size *= self.geometry[i]

        return size

    def width(self):
        """
        Returns width of the population.
        """
        return self.geometry[0]

    def height(self):
        """
        Returns height of the population.
        """
        return self.geometry[1]    

    def depth(self):
        """
        Returns depth of the population.
        """        
        return self.geometry[2]

    def rank_from_coordinates(self, width, height, depth=0):
        """
        Returns rank corresponding to (w, h, d) coordinates. The depth d is 0 by default.
        """
        return depth*self.geometry[0]*self.geometry[1] + self.geometry[0] * height + width

    def coordinates_from_rank(self, rank):
        """
        Returns (w, h, d) coordinates corresponding to rank.
        """
        coord = [0, 0, 0]

        if(self.geometry[2]==1):
            if(self.geometry[1]==1):
                coord[0] = rank
            else:
                coord[1] = rank / self.geometry[0]
                coord[0] = rank - coord[1]*self.geometry[1]
        else:
            coord[2] = rank / ( self.geometry[0] * self.geometry[1] )

            plane_rank = rank - coord[2] * ( self.geometry[0] * self.geometry[1] )
            coord[1] = plane_rank / self.geometry[0]
            coord[0] = plane_rank - coord[1]*self.geometry[1]

        return coord

    def normalized_coordinates_from_rank(self, rank, norm=1):
        """
        Returns (width, height, depth) coordinates normalized to norm corresponding to rank.
        """
        
        coord = [0, 0, 0]

        if(self.geometry[2]==1):
            if(self.geometry[1]==1):
                coord[0] = rank/(self.size()-norm)
            else:
                width = rank / self.geometry[0]
                height = rank - coord[1]*self.geometry[1]
                coord[0] = width / (self.width()-norm)
                coord[1] = height / (self.height()-norm)
        else:
            depth = rank / ( self.geometry[0] * self.geometry[1] )
            #coord in plane
            pRank = rank - coord[2] * ( self.geometry[0] * self.geometry[1] )
            height = pRank / self.geometry[0]
            width = pRank - coord[1]*self.geometry[1]

            coord[0] = width / (self.width()-norm)
            coord[1] = height / (self.height()-norm)
            coord[2] = depth / (self.depth()-norm)

        return coord

    def set(self, **keyValueArgs):
        """
        update neuron variable/parameter definition
        """
        self.neuron.set(keyValueArgs)