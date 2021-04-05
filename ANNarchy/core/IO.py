#===============================================================================
#
#     IO.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
from ANNarchy.core import Global
import os
try:
    import cPickle as pickle # Python2
except:
    import pickle # Python3
import numpy as np


def load_parameters(filename, global_only=True, verbose=False, net_id=0):
    """
    Loads the global parameters of a network (flag ``population`` for neurons, ``projection`` for synapses) from a JSON file.

    It is advised to generate the JSON file first with ``save_parameters()`` and later edit it manually.

    A strong restriction is that population/projection names cannot change between saving and loading.
    By default, they take names such as ``pop0`` or ``proj2``, we advise setting explicitly a name in their constructor for readability.

    If you add a parameter name to the JSON file but it does not exist in te neuron/synapse, it will be silently skipped.
    Enable ``verbose=True`` to see which parameters are effectively changed.

    If you set ``global_only`` to True, you will be able to set values for non-global parameters (e.g. synapse-specific), but a single value will be loaded for all.
    The JSON file cannot contain arrays.

    If you want to save/load the value of variables after a simulation, please refer to ``save()`` or ``load()``.

    :param filename: path to the JSON file.
    :param global_only: True if only global parameters (flags ``population`` and ``projection``) should be loaded, the other values are ignored. (default: True)
    :param verbose: True if the old and new values of the parameters should be printed (default: False).
    :param net_id: ID of the network (default: 0, the global network).
    :return: a dictionary of additional parameters not related to populations or projections (keyword ``network`` in the JSON file).

    """
    import json
    with open(filename, 'r') as rfile:
        desc = json.load(rfile)

    if verbose:
        Global._print('Loading parameters from file', filename)
        Global._print('-'*40)

    # Populations
    try:
        populations = desc['populations']
    except:
        populations = {}
        if verbose:
            Global._print('load_parameters(): no population parameters.')
    for name, parameters in populations.items():
        # Get the population
        for pop in Global._network[net_id]['populations']:
            if pop.name == name:
                population = pop
                break
        else:
            Global._warning('The population', name, 'defined in the file', filename, 'does not exist in the current network.')

        if verbose:
            Global._print('Population', name)

        # Set the parameters
        for name, val in parameters.items():
            # Check that the variable indeed exists
            if not name in population.parameters:
                Global._print('  ', name, 'is not a global parameter of', population.name, ', skipping.')
                continue
            if global_only and not name in population.neuron_type.description['global']:
                Global._print('  ', name, 'is not a global parameter of', population.name, ', skipping.')
                continue

            if verbose:
                Global._print('  ', name, ':', population.get(name), '->', val)

            population.set({name: float(val)})

    # Projections
    try:
        projections = desc['projections']
    except:
        projections = {}
        if verbose:
            Global._print('load_parameters(): no projection parameters.')
    for name, parameters in projections.items():
        # Get the projection
        for proj in Global._network[net_id]['projections']:
            if proj.name == name:
                projection = proj
                break
        else:
            Global._warning('The projection', name, 'defined in the file', filename, 'does not exist in the current network.')

        if verbose:
            Global._print('Projection', name)

        # Set the parameters
        for name, val in parameters.items():
            # Check that the variable indeed exists
            if not name in projection.parameters:
                Global._print('  ', name, 'is not a global parameter of', population.name, ', skipping.')
                continue
            if global_only and not name in projection.synapse_type.description['global']:
                Global._print('  ', name, 'is not a global parameter of', population.name, ', skipping.')
                continue

            if verbose:
                Global._print('  ', name, ':', projection.get(name), '->', val)

            projection.set({name: float(val)})

    # Constants
    try:
        constants = desc['constants']
    except:
        constants = {}
        if verbose:
            Global._print('load_parameters(): no constants.')
    for name, value in constants.items():
        if name in Global.list_constants(): # modify it
            Global.get_constant(name).value = value
        else: # create it
            _ = Global.Constant(name, value)

    # Global user-defined parameters
    try:
        network_parameters = {}
        for name, val in desc['network'].items():
            network_parameters[name] = float(val)
    except:
        network_parameters = {}

    return network_parameters

def save_parameters(filename, net_id=0):
    """
    Saves the global parameters of a network (flag ``population`` for neurons, ``projection`` for synapses) to a JSON file.

    :param filename: path to the JSON file.
    :param net_id: ID of the network (default: 0, the global network).
    """
    import json

    # Get the netowrk description
    network = Global._network[net_id]

    # Dictionary of parameters
    description = {
        'populations' : {},
        'projections' : {},
        'network' : {},
        'constants' : {},
    }

    # Constants
    for constant in Global._objects['constants']:
        description['constants'][constant.name] = constant.value

    # Populations
    for pop in network['populations']:

        # Get the neuron description
        neuron = pop.neuron_type

        pop_description = {}

        for param in neuron.description['global']:
            pop_description[param] = pop.init[param]

        description['populations'][pop.name] = pop_description

    # Projections
    for proj in network['projections']:

        # Get the synapse description
        synapse = proj.synapse_type

        proj_description = {}

        for param in synapse.description['global']:
            proj_description[param] = proj.init[param]

        description['projections'][proj.name] = proj_description

    # Save the description in a json file
    with open(filename, 'w') as wfile:
        json.dump(description, wfile, indent=4)


# Backwards compatibility with XML
def load_parameter(in_file):
    Global._warning('load_parameter() is deprecated. Use load_parameters() and JSON files instead.')
    return _load_parameters_from_xml(in_file)

def _load_parameters_from_xml(in_file):
    """
    Load parameter set from xml file.

    If the location of the xml file differs from the base directory, you need to provide relative or absolute path.

    :param in_file: either single or collection of strings.
    """
    try:
        from lxml import etree
    except:
        Global._print('lxml is not installed. Unable to load in xml format.')
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
            Global._print('Error: file \'', file, '\' not found.')
            continue

        matches = doc.findall('parameter')

        for parameter in matches:
            childs = parameter.getchildren()

            #TODO: allways correct ???
            if len(childs) != 2:
                Global._print('Error: to much tags in parameter')

            name=None
            value=None
            for child in childs:

                if child.tag == 'name':
                    name = child.text
                elif child.tag == 'value':
                    value = child.text

                    if value is None:
                        Global._print('Error: no value defined for',name)
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
                    Global._print('Error: unexpected xml-tag', child.tag)

            if name is None:
                Global._print('Error: no name in parameter set.')
            elif value is None:
                Global._print('Error: no value in parameter set.')
                damaged_pars.append(name)
            elif name in par.keys():
                Global._print("Error: parameter",name,"already exists.")
                damaged_pars.append(name)
            else:
                par[name] = value

    return par

def _save_data(filename, data):
    """
    Internal routine to save data in a file.

    """
    # Check if the repertory exist
    (path, fname) = os.path.split(filename)

    if not path == '':
        if not os.path.isdir(path):
            Global._print('Creating folder', path)
            os.mkdir(path)

    extension = os.path.splitext(fname)[1]

    if extension == '.mat':
        Global._print("Saving network in Matlab format...")
        try:
            import scipy.io as sio
            sio.savemat(filename, data)
        except Exception as e:
            Global._error('Error while saving in Matlab format.')
            Global._print(e)
            return

    elif extension == '.gz':
        Global._print("Saving network in gunzipped binary format...")
        try:
            import gzip
        except:
            Global._error('gzip is not installed.')
            return
        with gzip.open(filename, mode = 'wb') as w_file:
            try:
                pickle.dump(data, w_file, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                Global._print('Error while saving in gzipped binary format.')
                Global._print(e)
                return

    elif extension == '.npz':
        Global._print("Saving network in Numpy format...")
        np.savez_compressed(filename, allow_pickle=True, **data )

    else:
        Global._print("Saving network in text format...")
        # save in Pythons pickle format
        with open(filename, mode = 'wb') as w_file:
            try:
                pickle.dump(data, w_file, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                Global._print('Error while saving in text format.')
                Global._print(e)
                return
        return

def save(filename, populations=True, projections=True, net_id=0):#, pure_data=True):
    """
    Save the current network state (parameters and variables) to a file.

    * If the extension is '.npz', the data will be saved and compressed using `np.savez_compressed` (recommended).

    * If the extension is '.mat', the data will be saved as a Matlab 7.2 file. Scipy must be installed.

    * If the extension ends with '.gz', the data will be pickled into a binary file and compressed using gzip.

    * Otherwise, the data will be pickled into a simple binary text file using cPickle.

    **Warning:** The '.mat' data will not be loadable by ANNarchy, it is only for external analysis purpose.

    Example:

    ```python
    save('results/init.npz')

    save('results/init.data')

    save('results/init.txt.gz')

    save('1000_trials.mat')
    ```

    :param filename: filename, may contain relative or absolute path.
    :param populations: if True, population data will be saved (by default True)
    :param projections: if True, projection data will be saved (by default True)

    """
    data = _net_description(populations, projections, net_id)
    _save_data(filename, data)

def _load_data(filename):
    """
    Internally loads data contained in a given file.

    :param filename: path to the file.
    :return: A dictionary with the connectivity and synaptic variables if the file ``filename`` is available otherwise None is returned.
    """
    (_, fname) = os.path.split(filename)
    extension = os.path.splitext(fname)[1]

    if extension == '.mat':
        Global._error('Unable to load Matlab format.')
        return None

    elif extension == '.gz':
        try:
            import gzip
        except:
            Global._error('gzip is not installed.')
            return None
        try:
            with gzip.open(filename, mode = 'rb') as r_file:
                desc = pickle.load(r_file)
            return desc
        except Exception as e:
            Global._print('Unable to read the file ' + filename)
            Global._print(e)
            return None

    elif extension == '.npz':
        try:
            data = np.load(filename, allow_pickle=True)
            desc = {}
            for attribute in data.files:
                # We need to distinguish two cases: 1) full network save
                # or 2) single pop/proj. The first case leads to a dictionary
                # of several objects. The latter to a dictionary containing all
                # values.
                if data[attribute].dtype == np.dtype('O'):
                    # attribute is a collection of multiple objects
                    desc[attribute] = data[attribute].item(0)
                else:
                    # attribute is a scalar/array
                    desc[attribute] = data[attribute]

            return desc
        except Exception as e:
            Global._print('Unable to read the file ' + filename)
            Global._print(e)
            return None

    else:
        try:
            with open(filename, mode = 'rb') as r_file:
                desc = pickle.load(r_file)
            return desc
        except Exception as e:
            Global._print('Unable to read the file ' + filename)
            Global._print(e)
            return None

def _load_connectivity_data(filename):
    """
    Internally loads data contained in a given file.

    :param filename: path to the file.
    :return: A dictionary with the connectivity and synaptic variables if the file ``filename`` is available otherwise None is returned.
    """
    (_, fname) = os.path.split(filename)
    extension = os.path.splitext(fname)[1]

    if extension == '.mat':
        Global._error('Unable to load Matlab format.')
        return None

    elif extension == '.gz':
        try:
            import gzip
        except:
            Global._error('gzip is not installed.')
            return None
        try:
            with gzip.open(filename, mode = 'rb') as r_file:
                desc = pickle.load(r_file)
            return desc
        except Exception as e:
            Global._print('Unable to read the file ' + filename)
            Global._print(e)
            return None

    elif extension == '.npz':
        try:
            data = np.load(filename, allow_pickle=True)
            desc = {}
            for attribute in data.files:
                # We need to distinguish two cases: 1) full network save
                # or 2) single pop/proj. The first case leads to a dictionary
                # of several objects. The latter to a dictionary containing all
                # values.
                desc[attribute] = data[attribute]

            return desc
        except Exception as e:
            Global._print('Unable to read the file ' + filename)
            Global._print(e)
            return None

    else:
        try:
            with open(filename, mode = 'rb') as r_file:
                desc = pickle.load(r_file)
            return desc
        except Exception as e:
            Global._print('Unable to read the file ' + filename)
            Global._print(e)
            return None

def load(filename, populations=True, projections=True, net_id=0):
    """
    Loads a saved state of the network.

    **Warning:** Matlab data can not be loaded.

    Example:

    ```python
    load('results/network.npz')
    ```


    :param filename: the filename with relative or absolute path.
    :param populations: if True, population data will be loaded (by default True)
    :param projections: if True, projection data will be loaded (by default True)
    """

    desc = _load_data(filename)
    if desc is None:
        return

    if 'time_step' in desc.keys():
        Global.set_current_step(desc['time_step'], net_id)

    if populations:
        # Over all populations
        for pop in Global._network[net_id]['populations']:
            # check if the population is contained in save file
            if pop.name in desc.keys():
                pop._load_pop_data(desc[pop.name])

    if projections:
        for proj in Global._network[net_id]['projections'] :
            if proj.name in desc.keys():
                proj._load_proj_data(desc[proj.name])


def _net_description(populations, projections, net_id=0):
    """
    Returns a dictionary containing the requested network data.

    :param populations: if True, the population data will be saved.
    :param projections: if True, the projection data will be saved.
    """
    network_desc = {}
    network_desc['time_step'] = Global.get_current_step(net_id)
    network_desc['net_id'] = net_id

    pop_names = []
    proj_names = []

    if populations:
        for pop in Global._network[net_id]['populations']:
            network_desc[pop.name] = pop._data()
            pop_names.append(pop.name)

    if projections:
        for proj in Global._network[net_id]['projections']:
            if not proj._saveable:
                continue
            network_desc[proj.name] = proj._data()
            proj_names.append(proj.name)

    network_desc['obj_names'] = {
        'populations': pop_names,
        'projections': proj_names,
    }

    return network_desc
