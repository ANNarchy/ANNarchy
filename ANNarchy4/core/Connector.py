"""
    
    Connector.py
    
    This file is part of ANNarchy.
    
    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
"""
from Random import Constant
from Dendrite import Dendrite
import numpy as np

class Connector(object):
    """
    The connector class manages all information and operations related to connection patterns.
    """
    def __init__(self, weights, delays=0, **parameters):
        """
        Initialize a connection object.

        Parameters:
                
        * *weights*: synaptic weights for all synapses of one projection. Could be either a RandomDistribution object or an array with the corresponding amount of weight values.
        * *delays*: synaptic delay for all synapses of one projection. Could be either a RandomDistribution object or an array with the corresponding amount of delay values.
        * *parameters*: any key-value pairs, except the previous ones, given to this function are interpreted as parameters for the connection pattern.
        """
        if isinstance(weights, float) or isinstance(weights, int):
            self.weights = Constant( float(weights) )
        else:
            self.weights = weights
        
        if isinstance(delays, float) or isinstance(delays, int):
            self.delays = Constant( int(delays) )
        else:
            self.delays = delays
            
        self.parameters = parameters
        self.cy_instance = None
        self.proj = None    # set externally

    def cpp_call(self):
        """
        Returns the instantiation call for cpp only mode.
        """
        return None

class One2One(Connector):
    """
    One2One connector.
    """
    
    def __init__(self, weights, delays=0, **parameters):
        """
        Initialize an One2One connection object.

        Parameters:
        
            * *pre*: the presynaptic population (python Population instance)            
            * *post*: the postsynaptic population (python Population instance)            
            * *target*: string describing the connection type
            * *weights*: synaptic weights for all synapses of one projection. Could be either a RandomDistribution object or an array with the corresponding amount of weight values.            
            * *delays*: synaptic delay for all synapses of one projection. Could be either a RandomDistribution object or an array with the corresponding amount of delay values.                        
            * *parameters*: any key-value pairs, except the previous ones, given to this function are interpreted as parameters for the connection pattern.

        
        Specific parameters:
        
            * None for this pattern.
        """        
        super(self.__class__, self).__init__(weights, delays, **parameters)

    def connect(self):
        """
        Build up the defined connection pattern for all postsynaptic neurons.
        """
        cython_module = __import__('ANNarchyCython')
        proj_type = self.proj.generator.proj_class['ID']
        
        self.cy_instance = getattr(cython_module, 'One2One')(proj_type)
            
        target = self.proj.post.generator.targets.index(self.proj.target)
        tmp = self.cy_instance.connect(self.proj.pre,
                                          self.proj.post,
                                          target,
                                          self.weights,
                                          self.delays,
                                          self.parameters
                                          )
        
        #self.proj.pre.cyInstance.set_max_delay(int(self.delays.max()))

        dendrites = []
        post_ranks = []
        for i in xrange(len(tmp)):
            dendrites.append(Dendrite(self.proj, tmp[i].post_rank, tmp[i]))
            post_ranks.append(tmp[i].post_rank)

        return dendrites, post_ranks
    
    def cpp_call(self):
        return '&(One2OneConnector(new ' + self.weights._gen_cpp() +'))'

class All2All(Connector):
    """
    All2All projection between two populations. Each neuron in the postsynaptic 
    population is connected to all neurons of the presynaptic population.
    
    """
    def __init__(self, weights, delays=0, **parameters):
        """
        Initialize an One2One connection object.

        Parameters:
        
            * *pre*: the presynaptic population (python Population instance)            
            * *post*: the postsynaptic population (python Population instance)            
            * *target*: string describing the connection type
            * *weights*: synaptic weights for all synapses of one projection. Could be either a RandomDistribution object or an array with the corresponding amount of weight values.            
            * *delays*: synaptic delay for all synapses of one projection. Could be either a RandomDistribution object or an array with the corresponding amount of delay values.                        
            * *parameters*: any key-value pairs, except the previous ones, given to this function are interpreted as parameters for the connection pattern.

        
        Specific parameters:
        
            * *allow_self_connections*: if the ranks of two neurons are equal and within the same population, this parameter determines wether a connection is build up or not.
        """        
        super(self.__class__, self).__init__(weights, delays, **parameters)
    
    def connect(self):
        """
        Build up the defined connection pattern for all postsynaptic neurons.
        """
        cython_module = __import__('ANNarchyCython')
        proj_type = self.proj.generator.proj_class['ID']
        
        self.cy_instance = getattr(cython_module, 'All2All')(proj_type)
            
        target = self.proj.post.generator.targets.index(self.proj.target)
        tmp = self.cy_instance.connect(self.proj.pre,
                                          self.proj.post,
                                          target,
                                          self.weights,
                                          self.delays,
                                          self.parameters
                                          )

        #self.proj.pre.cyInstance.set_max_delay(int(self.delays.max()))
        
        dendrites = []
        post_ranks = []
        for i in xrange(len(tmp)):
            dendrites.append(Dendrite(self.proj, tmp[i].post_rank, tmp[i]))
            post_ranks.append(tmp[i].post_rank)

        return dendrites, post_ranks

    def cpp_call(self):
        return '&(All2AllConnector(new ' + self.weights._gen_cpp() +'))'

class Gaussian(Connector):
    """
    gaussians projection between to populations.
    
    Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around 
    the neuron with the same rank and width weights following a gaussians distribution.    
    """
    def __init__(self, weights, delays=0, **parameters):
        """
        Initialize an One2One connection object.

        Parameters:
        
            * *pre*: the presynaptic population (python Population instance)            
            * *post*: the postsynaptic population (python Population instance)            
            * *target*: string describing the connection type
            * *weights*: synaptic weights for all synapses of one projection. Could be either a RandomDistribution object or an array with the corresponding amount of weight values.            
            * *delays*: synaptic delay for all synapses of one projection. Could be either a RandomDistribution object or an array with the corresponding amount of delay values.                        
            * *parameters*: any key-value pairs, except the previous ones, given to this function are interpreted as parameters for the connection pattern.

        Specific parameters:
        
            * *amp*: the maximal value of the gaussian distribution for the connection weights.
            * *sigma*: the standard deviation of the gaussian distribution (normalized by the number of neurons in each dimension of the presynaptic population) 
            * *limit*: percentage of amplitude below which the connection is not created (default = 0.01)
            * *allow_self_connections*: if self-connections are allowed or not (default = False) 
        """        
        super(self.__class__, self).__init__(weights, delays, **parameters)
        
    def connect(self):
        """
        Build up the defined connection pattern for all postsynaptic neurons.
        """
        cython_module = __import__('ANNarchyCython')
        proj_type = self.proj.generator.proj_class['ID']
        
        self.cy_instance = getattr(cython_module, 'Gaussian')(proj_type)
            
        target = self.proj.post.generator.targets.index(self.proj.target)
        tmp = self.cy_instance.connect(self.proj.pre,
                                          self.proj.post,
                                          target,
                                          self.weights,
                                          self.delays,
                                          self.parameters
                                          )

        #self.proj.pre.cyInstance.set_max_delay(int(self.delays.max()))

        dendrites = []
        post_ranks = []
        for i in xrange(len(tmp)):
            dendrites.append(Dendrite(self.proj, tmp[i].post_rank, tmp[i]))
            post_ranks.append(tmp[i].post_rank)

        return dendrites, post_ranks        

class DoG(Connector):
    """
    Difference-of-gaussians projection between to populations.
    
    Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around 
    the neuron with the same rank and width weights following a difference-of-gaussians distribution.
    """    
    def __init__(self, weights, delays=0, **parameters):
        """
        Initialize an One2One connection object.

        Parameters:
        
            * *pre*: the presynaptic population (python Population instance)            
            * *post*: the postsynaptic population (python Population instance)            
            * *target*: string describing the connection type
            * *weights*: synaptic weights for all synapses of one projection. Could be either a RandomDistribution object or an array with the corresponding amount of weight values.            
            * *delays*: synaptic delay for all synapses of one projection. Could be either a RandomDistribution object or an array with the corresponding amount of delay values.                        
            * *parameters*: any key-value pairs, except the previous ones, given to this function are interpreted as parameters for the connection pattern.

        
        Specific parameters:
        
            * *amp_pos*: the maximal value of the positive gaussian distribution for the connection weights.
            * *sigma_pos*: the standard deviation of the positive gaussian distribution (normalized by the number of neurons in each dimension of the presynaptic population) 
            * *amp_pos*: the maximal value of the negative gaussian distribution for the connection weights.
            * *sigma_pos*: the standard deviation of the negative gaussian distribution (normalized by the number of neurons)
            * *limit*: percentage of ``amp_pos - amp_neg`` below which the connection is not created (default = 0.01)
            * *allow_self_connections*: if self-connections are allowed or not (default = False) 
        """
        super(self.__class__, self).__init__(weights, delays, **parameters)
        
    def connect(self):
        """
        Build up the defined connection pattern for all postsynaptic neurons.
        """
        cython_module = __import__('ANNarchyCython')
        proj_type = self.proj.generator.proj_class['ID']
        
        self.cy_instance = getattr(cython_module, 'DoG')(proj_type)
            
        target = self.proj.post.generator.targets.index(self.proj.target)
        tmp = self.cy_instance.connect(self.proj.pre,
                                          self.proj.post,
                                          target,
                                          self.weights,
                                          self.delays,
                                          self.parameters
                                          )
        
        #self.proj.pre.cyInstance.set_max_delay(int(self.delays.max()))

        dendrites = []
        post_ranks = []
        for i in xrange(len(tmp)):
            dendrites.append(Dendrite(self.proj, tmp[i].post_rank, tmp[i]))
            post_ranks.append(tmp[i].post_rank)

        return dendrites, post_ranks        
