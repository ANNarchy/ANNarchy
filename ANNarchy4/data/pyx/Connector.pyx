# cython: embedsignature=True

from ANNarchy4.core.Random import *
import ANNarchy4.core.Projection as PyProjection

from libcpp.vector cimport vector 
from libc.stdlib cimport malloc
from libc.math cimport exp, abs

cdef class PyxConnector:
    """
    Base class for all connection patterns implemented as python extension.
    """
    cdef proj_type
    
    def __cinit__(self, proj_type):
        """
        Constructor
        
        Parameter:
        
        * proj_type:    unique ID of the projection (base projection = 0) 
        """
        self.proj_type = proj_type
        
    cdef create_local_proj(self, proj_type, pre_id, post_id, proj, target):
        return eval('LocalProjection'+str(proj_type)+'(proj_type, pre_id, post_id, proj, target)')
        
cdef class One2One(PyxConnector):
    """
    
    """
    cdef postSize

    def __cinit__(self, proj_type):
        PyxConnector.__init__(proj_type)
        
    cdef genRanks(self):
        cdef int i
        cdef vector[int] tmp
        cdef vector[vector[int]] ranks

        ranks.clear()
        tmp.clear()
        tmp.push_back(0)

        for i in range(self.postSize):
            tmp[0] = i

            ranks.push_back(tmp)

        return ranks

    def connect(self, pre, post, target, weights, delays, parameters):
        """
        Create the connection informations and instantiate the c++ classes.
        
        Parameters:
        
            * *pre*: the presynaptic population (python Population instance)            
            * *post*: the postsynaptic population (python Population instance)            
            * *target*: string describing the connection type
            * *weights*: synaptic weights as an instance of RandomDistribution            
            * *delays*: synaptic delays as an instance of RandomDistribution                        
            * *parameters*: pattern specific parameters
        
        Specific parameters:
        
            * None for this pattern.
        """
        self.postSize = post.size

        if (pre.size != self.postSize):
            return None

        r = self.genRanks()

        Proj = []
        
        for p in xrange(self.postSize):
            local = self.create_local_proj(self.proj_type, pre.id, post.id, p, target)
            v = weights.get_value()
            d = delays.get_value()
            
            local.init(r[p], v, d)
            
            Proj.append(local)

        return Proj    

cdef class All2All(PyxConnector):
    """
    All2All projection between two populations. Each neuron in the postsynaptic 
    population is connected to all neurons of the presynaptic population.
    
    """
    cdef allowSelfConnections
    cdef preID
    cdef postID
    cdef preSize
    cdef postSize
    cdef vector[vector[int]] ranks

    def __cinit__(self, proj_type):
        """
        Constructor.
        
        Parameters:
    
        * proj_type:    unique ID of the projection (base projection = 0)
        """
        PyxConnector.__init__(proj_type)

    def connect(self, pre, post, target, weights, delays, parameters):
        """
        Create the connection informations and instantiate the c++ classes.
        
        Parameters:
        
            * *pre*: the presynaptic population (python Population instance)            
            * *post*: the postsynaptic population (python Population instance)            
            * *target*: string describing the connection type
            * *weights*: synaptic weights as an instance of RandomDistribution            
            * *delays*: synaptic delays as an instance of RandomDistribution                        
            * *parameters*: pattern specific parameters
        
        Specific parameters:
        
            * *allow_self_connections*: if self-connections are allowed or not (default = False) 
        """
        self.preID = pre.id
        self.postID = post.id        
        self.preSize = pre.size
        self.postSize = post.size

        if 'allowSelfConnections' in parameters.keys():
            self.allowSelfConnections = parameters['allowSelfConnections']
        else:
            self.allowSelfConnections = False

        self.genRanks()
        
        Proj = []

        for p in xrange(self.postSize):
            local = self.create_local_proj(self.proj_type, pre.id, post.id, p, target)
            if not self.allowSelfConnections and (pre==post):
                v = weights.get_values(self.preSize-1)
                d = delays.get_values(self.preSize-1)
            else:
                v = weights.get_values(self.preSize)
                d = delays.get_values(self.preSize)
            
            local.init(self.ranks[p], v, d)
            Proj.append(local)            

        return Proj
    
    cdef genRanks(self):
        cdef int i
        cdef int j
        cdef vector[int] tmp
        
        self.ranks = vector[vector[int]]()
        
        if (self.preID == self.postID) and not self.allowSelfConnections:
            for i in range(self.postSize):
                tmp.clear()
                j = 0
    
                while j < self.preSize:
                    if( i != j):
                        tmp.push_back(j)
                    j = j+1

                self.ranks.push_back(tmp)
        else:
            tmp.clear()
            for i in xrange(self.preSize):
                tmp.push_back(i)

            for j in xrange(self.postSize):
                self.ranks.push_back(tmp)

cdef class DoG(PyxConnector):
    """
    Difference-of-gaussians projection between to populations.
    
    Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around 
    the neuron with the same rank and width weights following a difference-of-gaussians distribution.
    """
    cdef preSize
    cdef postSize
    cdef pre
    cdef post
    cdef amp_pos
    cdef amp_neg
    cdef sigma_pos
    cdef sigma_neg
    cdef limit
    cdef delay_dist

    def __cinit__(self, proj_type):
        """
        Constructor.
        
        Parameters:
    
        * proj_type:    unique ID of the projection (base projection = 0)
        """
        PyxConnector.__init__(proj_type)
      
    def connect(self, pre, post, target, weights, delays, parameters):
        """
        Create the connection informations and instantiate the c++ classes.
        
        Parameters:
        
            * *pre*: the presynaptic population (python Population instance)            
            * *post*: the postsynaptic population (python Population instance)            
            * *target*: string describing the connection type
            * *weights*: synaptic weights as an instance of RandomDistribution            
            * *delays*: synaptic weights as an instance of RandomDistribution                        
            * *parameters*: pattern specific parameters
        
        Specific parameters:
        
            * *amp_pos*: the maximal value of the positive gaussian distribution for the connection weights.
            * *sigma_pos*: the standard deviation of the positive gaussian distribution (normalized by the number of neurons in each dimension of the presynaptic population) 
            * *amp_pos*: the maximal value of the negative gaussian distribution for the connection weights.
            * *sigma_pos*: the standard deviation of the negative gaussian distribution (normalized by the number of neurons)
            * *limit*: percentage of ``amp_pos - amp_neg`` below which the connection is not created (default = 0.01)
            * *allow_self_connections*: if self-connections are allowed or not (default = False) 
        """        
        self.preSize = pre.size
        self.postSize = post.size
        self.pre = pre
        self.post = post

        self.sigma_pos = parameters['sigma_pos']
        self.sigma_neg = parameters['sigma_neg']
        self.amp_pos = parameters['amp_pos']
        self.amp_neg = parameters['amp_neg']
        
        self.delay_dist = delays
        
        if 'limit' in parameters.keys():
            self.limit = parameters['limit']
        else:
            self.limit = 0.01

        if (self.preSize != self.postSize):
            return None

        Proj = []

        for p in xrange(self.postSize):
            local = self.create_local_proj(self.proj_type, pre.id, post.id, p, target)
            
            r, v, d = self.genRanksAndValues(p)
            
            local.init(r, v, d)
            Proj.append(local)

        return Proj

    cdef compDist(self, pre, post):
        cdef float res = 0.0
        cdef int i = 0

        for i in range(len(pre)):
            res = res + (pre[i]-post[i])*(pre[i]-post[i]);

        return res

    cdef genRanksAndValues(self, postRank):
        cdef int j
        cdef float dist, value
        cdef vector[int] normPre, normPost
        cdef vector[int] ranks
        cdef vector[int] delay
        cdef vector[float] values
        selfConnection = False

        ranks.clear()
        values.clear()
        delay.clear()

        normPost = self.post.normalized_coordinates_from_rank(postRank)

        for j in range(self.preSize):
            if (not selfConnection or (self.pre != self.post)):
                normPre = self.pre.normalized_coordinates_from_rank(j)
                dist = self.compDist(normPre, normPost)
                value = self.amp_pos*exp(-dist/2.0/self.sigma_pos/self.sigma_pos) - self.amp_neg*exp(-dist/2.0/self.sigma_neg/self.sigma_neg)
                if (abs(value) > self.limit*abs(self.amp_pos-self.amp_neg)):
                    ranks.push_back(j)
                    values.push_back(value)
                    delay.self.delay_dist.get_value()
                    
        return ranks, values, delay
