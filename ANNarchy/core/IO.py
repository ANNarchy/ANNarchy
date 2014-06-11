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
import cPickle

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
    
def save(filename, populations=True, projections=True):#, pure_data=True):
    """
    Save the current network state to a file.

    * If the extension is '.mat', the data will be saved as a Matlab 7.2 file. Scipy must be installed.

    * If the extension ends with '.gz', the data will be pickled into a binary file and compressed using gzip.

    * Otherwise, the data will be pickled into a simple text file.
    
    Parameter:
    
    * *filename*: filename, may contain relative or absolute path.
    
        .. warning:: The '.mat' data will not be loadable by ANNarchy, it is only for external analysis purpose. 
    
    * *populations*: if True, population data will be saved (by default True)
    
    * *projections*: if True, projection data will be saved (by default True)
    
    Example:
    
        .. code-block:: python
        
            save('results/init.data')
        
            save('results/init.txt.gz')
            
            save('1000_trials.mat')
    
    """    
    # Check if the repertory exist
    (path, fname) = os.path.split(filename) 
    
    if not path == '':
        if not os.path.isdir(path):
            print('creating folder', path)
            os.mkdir(path)
    
    extension = os.path.splitext(fname)[1]
    
    data = _net_description(populations, projections)
    
    if extension == '.mat':
        Global._debug("Save in Matlab format.")
        import scipy.io as sio
        sio.savemat(filename, data)
        
    elif extension == '.gz':
        Global._debug("Save in gunzipped binary format.")
        try:
            import gzip
        except:
            Global._error('gzip is not installed.')
            return
        with gzip.open(filename, mode = 'wb') as w_file:
            try:
                cPickle.dump(data, w_file, protocol=cPickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Error while saving in gzipped binary format.')
                print(e)
                return
        
    else:
        Global._debug("Save in text format.")
        # save in Pythons pickle format
        with open(filename, mode = 'w') as w_file:
            try:
                cPickle.dump(data, w_file, protocol=cPickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Error while saving in text format.')
                print(e)
                return
        return

def load(filename, populations=True, projections=True):#, pure_data=True): TODO
    """
    Load the current network state.

    Warning: Matlab data can not be loaded.
    
    Parameter:
    
    * *in_file*: the complete filename.
    
    * *populations*: if True, population data will be loaded (by default True)
    
    * *projections*: if True, projection data will be loaded (by default True)
    
    Example:
    
        .. code-block:: python
        
            load('results/init.data')
            
    """   

    def _load(r_file):
        try:
            net_desc = cPickle.load(r_file)
    
            if populations:
                _load_pop_data(net_desc)  
            if projections:              
                _load_proj_data(net_desc)
    
        except Exception as e:
            print('Error while loading in Python pickle format.')
            print(e)
            return

    (path, fname) = os.path.split(filename)
    extension = os.path.splitext(fname)[1]

    if extension == '.mat':
        Global._error('Unable to load Matlab format.')
        return
    elif extension == '.gz':
        try:
            import gzip
        except:
            Global._error('gzip is not installed.')
            return
        with gzip.open(filename, mode = 'rb') as r_file:
            _load(r_file)
    else:
        with open(filename, mode = 'r') as r_file:
            _load(r_file)

  
def _net_description(populations, projections):
    """
    Returns a dictionary containing the requested network data.
    
    Parameter:
    
        * *populations*: if *True* the population data will be saved
        * *projections*: if *True* the projection data will be saved
    """
    network_desc = {}   
    
    if populations:
        for pop in Global._populations:             
            network_desc[pop.name] = pop._data() 

    if projections:
        for proj in Global._projections:  
            network_desc[proj.name] = proj._data() 


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
            if not 'attributes' in pop_desc.keys():
                _error('Saved with a too old version of ANNarchy.')
                return
            for var in pop_desc['attributes']:
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


                    
