"""

    Global.py
    
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
from __future__ import print_function

import sys, os
from datetime import datetime
from math import ceil

from PyQt4.QtCore import QCoreApplication

# instances
_populations = []       # created populations
_projections = []       # created projections

# predefined variables / parameters
_pre_def_synapse = ['value', 'rank', 'delay', 'psp']
_pre_def_synapse_var = ['value', 'rank', 'delay', 'psp']
_pre_def_synapse_par = []

_pre_def_neuron = ['rank', 'rate']

_cy_instance = None
_visualizer = None

# path to annarchy working directory
annarchy_dir = os.getcwd() + '/annarchy'

_compiled = False   #I know it's evil

# discretization timestamp
config = dict(
   { 
    'dt' : 1.0,
    'verbose': False,
    'show_time': False,
    'suppress_warnings': False,
    'float_prec': 'single',
    'num_threads': None #default by os
   }
)

def setup(**keyValueArgs):
    """
    The setup function is used to configure ANNarchy4 simulation environment. It takes various optional arguments: 

    Parameter:
    
    * *dt*:         discretization constant
    * *verbose*:    shows details about compilation process on console (by default False). Additional some information of the network construction will be shown.
    * *suppress_warnings*:  if set True warnings (e. g. from mathematical parser) are suppressed.
    * *show_time*:  if set True, initialization times are shown. ATTENTION: verbose should be set to True additionally.
    * *float_prec*: determine the used floating point precision. By default ANNarchy4 uses single floating point precision for computation. 
    
    **Note**: use this function before any other functions of ANNarchy4.
    """
    for key in keyValueArgs:

        if key in config.keys():
            config[key] = keyValueArgs[key]

def compile(clean=False, debug_build=False, cpp_stand_alone=False, profile_enabled=False):
    """
    Compile all classes and setup the network

    Parameters:
    
    * *clean*: boolean to specifying if the library should be recompiled entirely or only the changes since last compilation (default: False).
    * *debug_build*: creates a debug version of ANNarchy, which logs the creation of objects and some other data (default: False).

    .. hint: these parameters are also available but should only used if performance issues exists

        * *cpp_stand_alone*: creates a cpp library solely. It's possible to run the simulation, but no interaction possibilities exist. These argument should be always False.
        * *profile_enabled*: creates a profilable version of ANNarchy, which logs several computation timings (default: False).
    
    """
    generator.compile(clean, _populations, _projections, cpp_standalone, debug_build, profile_enabled)
    
def render():
    if _visualizer:
        _visualizer.render_data()
        
def reset(states=False, connections=False):
    """
    Reinitialises the network, runs each object's reset() method (resetting them to 0).

    Parameter:

    * *states*: if set to True then it will reinitialise the neuron state variables.
    * *connections*: if set to True then it will reinitialise the connection variables.
    """
    if states:
        for pop in _populations:
            pop.reset()
            
    if connections:
        print('currently not implemented')
        
def get_population(name):
    """
    Returns population corresponding to *name*.
    
    Parameter:
    
    * *name*: population name

    Returns:
    
    the requested population if existing otherwise None is returned.
    """
    for pop in _populations:
        if pop.name == name:
            return pop
        
    print("Error: no population",name,"found.")
    return None

def get_projection(pre, post, target):
    """
    Returns projection corresponding to the arguments.
    
    Parameters:
    
    * *pre*: presynaptic population
    * *post*: postsynaptic population
    * *target*: connection type
    
    Returns:
    
    the requested projection if existing otherwise None is returned.
    """
    for proj in _projections:
        
        if proj.post == post:
            if proj.pre == pre:
                if proj.target == target:
                    return proj
    
    print("Error: no projection",pre.name,"->",post.name,"with target ",target,"found.")
    return None

def simulate(duration):
    """
    Runs the network for the given duration.
    """
    nb_steps = ceil(duration / config['dt'])
    import ANNarchyCython
    
    #
    # check if user defined a certain number of threads.
    if config['num_threads'] != None:
        ANNarchyCython.pyNetwork().set_num_threads(config['num_threads'])
    
    for i in xrange(int(nb_steps)):
        QCoreApplication.processEvents() # handle user events
        ANNarchyCython.pyNetwork().Run(1)
    
def current_time():
    """
    Returns current simulation time in ms.
    
    **Note**: computed as number of simulation steps times dt
    """
    import ANNarchyCython
    ANNarchyCython.pyNetwork().get_time() * config['dt']

def current_step():
    """
    Returns current simulation step.
    """
    import ANNarchyCython
    ANNarchyCython.pyNetwork().get_time()

def set_current_time(time):
    """
    Set current simulation time in ms.
    
    **Note**: computed as number of simulation steps times dt
    """
    import ANNarchyCython
    ANNarchyCython.pyNetwork().set_time(int( time / config['dt']))
    
def set_current_step(time):
    """
    set current simulation step.
    """
    import ANNarchyCython
    ANNarchyCython.pyNetwork().set_time( time )
        
def record(to_record):
    """
    Record variables of one or more populations. For more detailed information please refer to the Population.record method.
    
    Parameter:
    
    * *to_record*: a set of dictionaries containing population objects and variable names. Optionally you may append an as_1D key to get a different format.
    
    Example:
    
        .. code-block:: python
        
            to_record = [
                { 'pop': Input, 'var': 'rate' }, 
                { 'pop': Input, 'var': 'mp', 'as_1D': True }        
            ]
            
        By default the variable data is always handled in population geometry shape. In this example, we force the storage of ``mp`` as one dimensional array.

    """
    for data_set in to_record:
        data_set['pop'].start_record(data_set['var'])

def get_record(to_record):
    """
    Retrieve recorded variables of one or more populations. For more detailed information please refer to the Population.get_record method.
  
    Parameter:
    
    * *to_record*: a set of dictionaries containing population objects and variable names. Optionally you may append an as_1D key to get a different format. For more details check Population.record().
    
    Returns:
    
    * A dictionary containing all recorded values. The dictionary is empty if no recorded data is available.
    
    Example:
    
        .. code-block:: python
        
            ...
        
    """    
    data = {}
    
    for data_set in to_record:
        if data_set['pop'].name in data:
            if 'as_1D' in data_set.keys():
                data[ data_set['pop'].name ].update( { data_set['var']: data_set['pop'].get_record(data_set['var'], data_set['as_1D']) } )
            else:
                data[ data_set['pop'].name ].update( { data_set['var']: data_set['pop'].get_record(data_set['var']) } )
        else:
            if 'as_1D' in data_set.keys():
                data.update( { data_set['pop'].name: { data_set['var']: data_set['pop'].get_record(data_set['var'], data_set['as_1D']) } } )
            else:
                data.update( { data_set['pop'].name: { data_set['var']: data_set['pop'].get_record(data_set['var']) } } )
    
    return data

def current_time():
    """
    Returns current simulation time in ms.
    
    **Note**: computed as number of simulation steps times dt
    """
    import ANNarchyCython
    return ANNarchyCython.pyNetwork().get_time() * config['dt']

def current_step():
    """
    Returns current simulation step.
    """
    import ANNarchyCython    
    return ANNarchyCython.pyNetwork().get_time()    

def set_current_time(time):
    """
    Set current simulation time in ms.
    
    **Note**: computed as number of simulation steps times dt
    """
    import ANNarchyCython
    ANNarchyCython.pyNetwork().set_time(int( time / config['dt']))

def set_current_step(time):
    """
    set current simulation step.
    """
    import ANNarchyCython    
    ANNarchyCython.pyNetwork().set_time( time )    

def _print(*var_text):
    """
    Prints a message to standard out.
    """    
    text = ''
    for var in var_text:
        text += str(var) + ' '
        
    if sys.version_info[:2] >= (2, 6) and sys.version_info[:2] < (3, 0):
        p = print        
        p(text)
    else:
        print(text)

def _debug(*var_text):
    """
    Prints a message to standard out, if verbose mode set True.
    """    
    if not config['verbose']:
        return
    
    text = ''
    for var in var_text:
        text += str(var) + ' '
        
    if sys.version_info[:2] >= (2, 6) and sys.version_info[:2] < (3, 0):
        p = print        
        p(text)
    else:
        print(text)
        
def _warning(*var_text):
    """
    Prints a warning message to standard out.
    """
    text = 'WARNING: '
    for var in var_text:
        text += str(var) + ' '

    if sys.version_info[:2] >= (2, 6) and sys.version_info[:2] < (3, 0):
        p = print        
        p(text)
    else:
        print(text)
        
def _error(*var_text):
    """
    Prints an error message to standard out.
    """
    text = 'ERROR: '
    for var in var_text:
        text += str(var) + ' '
    
    if sys.version_info[:2] >= (2, 6) and sys.version_info[:2] < (3, 0):
        p = print        
        p(text)
    else:
        print(text)

