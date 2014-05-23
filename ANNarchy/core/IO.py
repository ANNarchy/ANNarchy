"""

    IO.py
    
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
from ANNarchy.core import Global 
import os
import pickle

def load_parameter(in_file):
    """
    Load parameter set from xml file.
    
    Parameters:
    
    * *in_file*: either single or collection of strings. if the location of the xml file differs from the base directory, you need to provide relative or absolute path.
    """
    try:
        from lxml import etree 
    except:
        print 'lxml is not installed. Unable to save in xml format.'
        return
    par = {}
    damaged_pars = []   # for printout
    
    files = []
    if isinstance(in_file,str):
        files.append(in_file)
    else:
        files = in_file
    
    for file in files:
        try:
            doc = etree.parse(file)
            
        except IOError:
            print('Error: file \'',file,'\' not found.')
            continue
        
        matches = doc.findall('parameter')
        
        for parameter in matches:
            childs = parameter.getchildren()
    
            #TODO: allways correct ???
            if len(childs) != 2:
                print('Error: to much tags in parameter')
    
            name=None
            value=None
            for child in childs:
    
                if child.tag == 'name':
                    name = child.text
                elif child.tag == 'value':
                    value = child.text
                    
                    if value == None:
                        print('Error: no value defined for',name)
                        damaged_pars.append(name)
                        value = 0
                    else:
                        try:
                            value = int(value)
                        except ValueError:
                            try:
                                value = float(value)
                            except ValueError:
                                value = value
                        
                else:
                    print('Error: unexpected xml-tag', child.tag)
            
            if name == None:
                print('Error: no name in parameter set.')
            elif value == None:
                print('Error: no value in parameter set.')
                damaged_pars.append(name)
            elif name in par.keys():
                print("Error: parameter",name,"already exists.")
                damaged_pars.append(name)
            else:
                par[name] = value
        
        #if len(damaged_pars) > 0:
        #    print 'damaged parameters:'
        #    for d_par in damaged_pars:
        #        print '-',d_par
             
    return par
    
def save(in_file, pure_data=True, variables=True, connections=True):
    """
    Save the current network state to file.
    
    Parameter:
    
    * *in_file*: filename, may contain relative or absolute path. Allowed file extensions: '.mat' and '.data'
    
        .. warning:: Only the '.data' files are loadable by ANNarchy. 
        
    * *pure_data*: if True only the network state will be saved. If False additionaly all neuron and synapse definitions will be saved (by default True).
    
    * *variables*: if True population data will be saved (by default True)
    
    * *connections*: if True projection data will be saved (by default True)
    
    Example:
    
        .. code-block:: python
        
            save('results/init.data')
            
            save('1000_trials.mat')
    
    """    
    # Check if the repertory exist
    (path, filename) = os.path.split(in_file) 
    
    if not path == '':
        if not os.path.isdir(path):
            print('creating folder', path)
            os.mkdir(path)
    
    extension = os.path.splitext(filename)[1]
    
    #
    #
    if pure_data:
        data = _net_description(variables, connections)
    else:
        #
        # steps_
        #
        # pickle neuron, synapse
        #
        # pickle proj, population
        #
        # data = above + _description_data(variables, connections)
        print('Complete save currently not implemented')
        return
    
    if extension == '.mat':
        Global._debug("Save in matlab format.")
        import scipy.io as sio
        sio.savemat(in_file, data)
        
    elif extension == '.data':
        Global._debug("Save in python pickle format.")
        
        #
        # save in Pythons pickle format
        with open(in_file, mode = 'w') as w_file:
            try:
                pickle.dump(data, w_file, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Error while saving in Python pickle format.')
                print(e)
                return
    else:
        Global._error("invalid file format.")
        return

def load(in_file, pure_data=True):
    """
    Load the current network state.
    
    Parameter:
    
    * *in_file*: the complete filename, allowed extensions are: '.data' for python pickle format.
    * *pure_data*: if True only the network state will be loaded assumes that the network is build up. If False the stored neuron and synapse definitions will be used to build up a network (by default True).
    * *variables*: if True population data will be saved (by default True)
    * *connections*: if True projection data will be saved (by default True)
    
    Example:
    
        .. code-block:: python
        
            load('results/init.data')
            
    """    
    (path, filename) = os.path.split(in_file)
    extension = os.path.splitext(filename)[1]
    
    with open(in_file, mode = 'r') as r_file:
        try:
            net_desc = {}
            
            if extension == '.mat':
                Global._error("currently not supported to load network data from matlab files.")
                return
        
            elif extension == '.data':         
                net_desc = pickle.load(r_file)
    
            if pure_data:
                _load_pop_data(net_desc)
                
                _load_proj_data(net_desc)
            else:
                #
                # steps_
                #
                # unpickle neuron, synapse
                #
                # unpickle proj, population
                #
                # compile()
                #
                # _load_only_data(net_desc)
                print('Load network from scratch is not implemented yet.')
                return
    
        except Exception as e:
            print('Error while loading in Python pickle format.')
            print(e)
            return
  
def _net_description(variables, connections):
    """
    Returns a dictionary containing the requested network data.
    
    Parameter:
    
        * *variables*: if *True* the population data will be saved
        * *projection*: if *True* the projection data will be saved
    """
    network_desc = {}   
    
    if variables:
        for pop in Global._populations:
            pop_desc = {}
            pop_desc['name'] = pop.name
            pop_desc['geometry'] = pop.geometry
            # Attributes
            attributes = pop.attributes
            # Save all attributes           
            for var in attributes:
                try:
                    pop_desc[var] = getattr(pop.cyInstance, '_get_'+var)()
                except:
                    Global._error('Can not save the attribute ' + var + 'in the projection.')              
            network_desc[pop.name] = pop_desc 

    if connections:
        for proj in Global._projections:
    
            proj_desc = {}
            proj_desc['post_ranks'] = proj._post_ranks

            synapse_count = []
            dendrites = []  
            
            for d in proj.post_ranks:
                dendrite_desc = {}
                # Number of synapses in the dendrite
                synapse_count.append(proj[d].size())
                # Postsynaptic rank
                dendrite_desc['post_rank'] = d
                # Attributes
                attributes = proj.attributes
                if not 'value' in attributes:
                    attributes.append('value')
                if not 'rank' in attributes:
                    attributes.append('rank')
                if not 'delay' in attributes:
                    attributes.append('delay')
                # Save all attributes           
                for var in attributes:
                    try:
                        dendrite_desc[var] = getattr(proj.cyInstance, '_get_'+var)(d) 
                    except:
                        Global._error('Can not save the attribute ' + var + 'in the projection.')               
                dendrites.append(dendrite_desc)
            
            proj_desc['dendrites'] = dendrites
            proj_desc['number_of_synapses'] = synapse_count
            network_desc[proj.name] = proj_desc 

    return network_desc
            
def _load_pop_data(net_desc):
    """
    Update populations with the stored data set. 
    """
    # Over all populations
    for pop in Global._populations:        
        # check if the population is contained in save file
        if pop.name in net_desc.keys():
            pop_desc = net_desc[pop.name]
            for var in pop_desc.keys():
                if var in pop.attributes:
                    try:
                        getattr(pop.cyInstance, '_set_'+var)(pop_desc[var]) 
                    except:
                        Global._error('Can not load the variable ' + var * ' in the population ' + pop.name)
                        return
    
def _load_proj_data(net_desc):
    """
    Update projections with the stored data set. 
    """
    for proj in Global._projections:
        if proj.name in net_desc.keys():            
            proj_desc = net_desc[proj.name]
            if not proj_desc['post_ranks'] == proj._post_ranks:
                Global._error('The projection has not the same number of postsynaptic neurons.')
                return
            for dendrite in proj_desc['dendrites']:
                rk = dendrite['post_rank']
                for var in dendrite.keys():
                    if not var in ['post_rank', 'rank', 'delay']:
                        try:
                            getattr(proj.cyInstance, '_set_' + var)(rk, dendrite[var])
                        except Exception, e:
                            print e
                            Global._error('Can not set attribute ' + var + ' in the projection.')
                            return


                    
