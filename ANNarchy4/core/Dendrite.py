"""
Dendrite.py
"""
from Variable import Descriptor, Attribute
import Global
from ANNarchy4.core.Random import RandomDistribution

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

        # synapse variables           
        for value in self.variables + self.parameters:
            setattr(self, value, Attribute(value))   
    
    def set( self, value ):
        """
        Update dendrite variable/parameter.
        
        Parameter:
        
            * *value*: value need to be update
            
                .. code-block:: python
                
                    set( 'tau' : 20, 'value'= np.random.rand(8,8) } )
        """
        for val_key in value.keys():
            if hasattr(self.cyInstance, val_key):
                # Check the type of the data!!
                if isinstance(value[val_key], RandomDistribution):
                    val = value[val_key].getValues(self.size) 
                else: 
                    val = value[val_key]           
                # Set the value
                setattr(self.cyInstance, val_key, val)
            else:
                print "Error: dendrite has no parameter/variable called", val_key+"."    
                
    def get(self, value):
        """
        Get current variable/parameter value
        
        Parameter:
        
            * *value*: value name as string
        """
        if value in self.variables:
            return self.get_variable(value)
        elif value in self.parameters:
            return self.get_parameter(value)
        else:
            print "Error: dendrite has no parameter/variable called", value+"."     
               
    @property
    def variables(self):
        """
        Returns a list of all variable names.
        """
        ret_var = Global._pre_def_synapse_var
        
        for var in self.proj._parsed_variables():
            if not var['type'] == 'parameter' and not var['name'] in ret_var:
                ret_var.append(var['name'])
        
        return ret_var

    @property
    def parameters(self):
        """
        Returns a list of all parameter names.
        """
        ret_par = Global._pre_def_synapse_par 
        
        for var in self.proj._parsed_variables():
            if var['type'] == 'parameter' and not var['name'] in ret_par:
                ret_par.append(var['name'])
        
        return ret_par
    
    @property
    def size(self):
        """
        Number of synapses.
        """
        return self.cyInstance.size
    
    def __len__(self):
        """
        Number of synapses.
        """
        return self.cyInstance.size
        
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
        if hasattr(self.cyInstance, variable):
            return getattr(self.cyInstance, variable)
        else:
            print 'Error: variable', variable, 'does not exist in this dendrite.'
            print traceback.print_stack()

    def get_parameter(self, parameter):
        """
        Returns the value of the given parameter, which is common for all synapses in the dendrite.
        
        Parameters:
        
            * *parameter*:    a string representing the parameter's name.
        """
        if hasattr(self.cyInstance, parameter):
            return getattr(self.cyInstance, parameter)
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

