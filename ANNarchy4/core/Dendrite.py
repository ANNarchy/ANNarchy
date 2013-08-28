"""
Dendrite.py
"""
from Variable import Descriptor, Attribute
import Global

import numpy as np
import traceback

class Dendrite(Descriptor):
    """
    A dendrite encapsulates all synapses of one neuron.
    
    *Hint*: this class will be created from projection class.
    
    Parameter:
    
    * *proj*: projection instance
    * *post_rank*: rank of the postsynaptic neuron
    * *cython_instance*: instance of the cythonized dendrite class.
    """
    def __init__(self, proj, post_rank, cython_instance):
        self.cyInstance = cython_instance
        self.post_rank = post_rank
        self.proj = proj
        self.pre = proj.pre
        
        #
        # base variables
        self.value = Attribute('value')
        self.rank = Attribute('rank')
        self.delay = Attribute('delay')
        self.dt = Attribute('dt')
        self.tau = Attribute('tau')

        #
        # synapse variables                
        for value in self.proj.generator.parsed_synapse_variables:
            if value['name'] in Global._pre_def_synapse:
                continue
             
            cmd = 'self.'+value['name']+' = Attribute(\''+value['name']+'\')'   
            exec(cmd)
    
    @property
    def variables(self):
        """
        Returns a list of all variable names.
        """
        ret_var=[] 
        
        for var in self.proj.generator.parsed_synapse_variables:
            if not var['type'] == 'parameter':
                ret_var.append(var['name'])
        
        return ret_var

    @property
    def parameter(self):
        """
        Returns a list of all parameter names.
        """
        ret_par=[] 
        
        for var in self.proj.generator.parsed_synapse_variables:
            if var['type'] == 'parameter':
                ret_par.append(var['name'])
        
        return ret_par
    
    @property
    def size(self):
        """
        Number of synapses.
        """
        return self.cyInstance.get_size()
        
    @property
    def target(self):
        """
        Connection type id.
        """
        return self.cyInstance.get_target()
        
    def get_variable(self, variable):
        """
        Returns the value of the given variable for all synapses in the dendrite, as a NumPy array having the same geometry as the presynaptic population.
        
        Parameters:
        
            * *variable*:    a string representing the variable's name.
        """
        if hasattr(self, variable):
            var = eval('self.'+variable)

            m = np.zeros(self.pre.width * self.pre.height)
            m[self.rank[:]] = var[:]

            return self._reshape_vector(m)
        else:
            print 'Error: variable',variable,'does not exist in this dendrite.'
            print traceback.print_stack()

    def get_parameter(self, parameter):
        """
        Returns the value of the given parameter, which is common for all synapses in the dendrite.
        
        Parameters:
        
            * *parameter*:    a string representing the parameter's name.
        """
        if hasattr(self, parameter):
            return eval('self.'+parameter)
        else:
            print 'Error: parameter',parameter,'does not exist in this dendrite.'
            print traceback.print_stack()

    def add_synapse(self, rank, value, delay=0):
        """
        Adds a synapse to the dendrite.
        
        Parameters:
        
            * *rank*:     rank of the presynaptic neuron
            * *value*:    synaptic weight
            * *delay*:    additional delay of the synapse (as default a delay of 1ms is assumed)
        """
        self.cyInstance.add_synapse(rank, value, delay)
    
    def remove_synapse(self, rank):
        """
        Removes the synapse from the dendrite.
        
        Parameters:
        
            * *rank*:     rank of the presynaptic neuron
        """
        self.cyInstance.remove_synapse(rank)

    def _reshape_vector(self, vector):
        """
        Transfers a list or a 1D np.array (indiced with ranks) into the correct 1D, 2D, 3D np.array
        """
        vec = np.array(vector) # if list transform to vec
        try:
            if self.pre.dimension == 1:
                return vec
            elif self.pre.dimension == 2:
                return vec.reshape(self.pre.height, self.pre.width)
            elif self.pre.dimension == 3:
                return vec.reshape(self.pre.depth, self.pre.height, self.pre.width)
        except ValueError:
            print 'Mismatch between pop: (',self.pre.width,',',self.pre.height,',',self.pre.depth,')'
            print 'and data vector (',type(vector),': (',vec.size,')'
