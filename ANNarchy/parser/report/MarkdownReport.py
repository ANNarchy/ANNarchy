import ANNarchy
from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.ConfigManagement import get_global_config
from ANNarchy.intern.GlobalObjects import GlobalObjectManager
from ANNarchy.intern import Messages
from ANNarchy.core.Neuron import Neuron
from ANNarchy.core.Synapse import Synapse
from ANNarchy.core.PopulationView import PopulationView
import ANNarchy.parser.report.LatexParser as LatexParser
from ANNarchy.parser.AnalyseNeuron import analyse_neuron
from ANNarchy.parser.AnalyseSynapse import analyse_synapse
from ..Extraction import *

import numpy as np
import os

##################################
### Main method
##################################

def report_markdown(filename:str="./report.tex", standalone:bool=True, gather_subprojections:bool=False, title:str=None, author:str=None, date:str=None, net_id:int=0):
    """ Generates a .md file describing the network.

    *Parameters:*

    * **filename**: name of the .tex file where the report will be written (default: "./report.tex")
    * **standalone**: tells if the generated file should be directly compilable or only includable (ignored)
    * **gather_subprojections**: if a projection between two populations has been implemented as a multiple of projections between sub-populations, this flag allows to group them in the summary (default: False).
    * **net_id**: id of the network to be used for reporting (default: 0, everything that was declared)
    * **title**: title of the document (default: "Network description")
    * **author**: author of the document (default: "ANNarchy (Artificial Neural Networks architect)")
    * **date**: date of the document (default: empty)
    """

    # stdout
    Messages._print('Generating report in', filename)

    # Header
    if title == None:
        title = "Network description"
    if author == None:
        author = "ANNarchy (Artificial Neural Networks architect)"
    if date == None:
        date = ""
    header = """---
title: %(title)s
author: %(author)s
date: %(date)s
---
""" % {'title': title, 'author': author, 'date': date}

    # Structure
    structure = _generate_summary(net_id)

    # Neurons
    neuron_models = _generate_neuron_models(net_id)

    # Synapses
    synapse_models = _generate_synapse_models(net_id)

    # Parameters
    parameters = _generate_parameters(net_id, gather_subprojections)

    # Possibly create the directory if it does not exist
    path_name = os.path.dirname(filename)
    if not path_name in ["", "."]:
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        
    with open(filename, 'w') as wfile:
        wfile.write(header)
        wfile.write(structure)
        wfile.write(neuron_models)
        wfile.write(synapse_models)
        wfile.write(parameters)


def _generate_summary(net_id):

    txt = """
# Structure of the network
"""

    # General information
    backend = 'default'
    if get_global_config('paradigm') == 'cuda':
        backend = "CUDA"
    elif get_global_config('paradigm') == "openmp" and get_global_config('num_threads') > 1:
        backend = "OpenMP"
    txt +="""
* ANNarchy %(version)s using the %(backend)s backend.
* Numerical step size: %(dt)s ms.
""" % {'version': ANNarchy.__release__, 'backend': backend, 'dt': get_global_config('dt')}

    # Populations
    if NetworkManager().number_populations(net_id=net_id) > 0:
        headers = ["Population", "Size", "Neuron type"]
        populations = []
        for pop in NetworkManager().get_populations(net_id=net_id):
            # Find a name for the neuron
            neuron_name = "Neuron " + str(pop.neuron_type._rk_neurons_type) if pop.neuron_type.name in Neuron._default_names.values() \
                else pop.neuron_type.name

            populations.append([
                pop.name, 
                pop.geometry if len(pop.geometry)>1 else pop.size, 
                neuron_name])

        txt += """
## Populations

"""
        txt += _make_table(headers, populations)


    # Projections
    if NetworkManager().number_projections(net_id=net_id) > 0 :
        headers = ["Source", "Destination", "Target", "Synapse type", "Pattern"]
        projections = []
        for proj in NetworkManager().get_projections(net_id=net_id):
            # Find a name for the synapse
            synapse_name = "Synapse " + str(proj.synapse_type._rk_synapses_type) if proj.synapse_type.name in Synapse._default_names.values() \
                else proj.synapse_type.name

            projections.append([
                proj.pre.name, 
                proj.post.name, 
                LatexParser._format_list(proj.target, ' / '),
                synapse_name,
                proj.connector_description
                ])

        txt += """
## Projections

"""
        txt += _make_table(headers, projections)

    # Monitors
    if NetworkManager().number_monitors(net_id=net_id) > 0:
        headers = ["Object", "Variables", "Period"]
        monitors = []
        for monitor in NetworkManager().get_monitors(net_id=net_id):
            monitors.append([
                monitor.object.name + (" (subset)" if isinstance(monitor.object, PopulationView) else ""), 
                LatexParser._format_list(monitor.variables, ', '),
                monitor.period
                ])

        txt += """
## Monitors

"""
        txt += _make_table(headers, monitors)

    # Functions
    if GlobalObjectManager().number_functions() > 0 :
        txt += """## Functions

"""
        for _, func in GlobalObjectManager().get_functions():
            txt += LatexParser._process_functions(func, begin="$$", end="$$\n\n")

    return txt


# Neuron template
neuron_tpl = """
## %(name)s

%(description)s

**Parameters:**

%(parameters)s

**Equations:**

%(eqs)s
"""
def _generate_neuron_models(net_id):
    txt = """
# Neuron models
"""
    for idx, neuron in enumerate(GlobalObjectManager().get_neuron_types()):

        # Name
        if neuron.name in Neuron._default_names.values(): # name not set
            neuron_name = "Neuron " + str(neuron._rk_neurons_type)
        else:
            neuron_name = neuron.name

        # Description
        description = neuron.short_description
        if description == None:
            description = "Spiking neuron." if neuron.type == 'spike' else 'Rate-coded neuron'

        # Parameters
        parameters = extract_parameters(neuron.parameters, neuron.extra_values)
        parameters_list = [
            ["$" + LatexParser._latexify_name(param['name'], []) + "$", param['init'], 
                _adapt_locality_neuron(param['locality']), param['ctype']] 
                    for param in parameters]

        parameters_headers = ["Name", "Default value", "Locality", "Type"]
        parameters_table = _make_table(parameters_headers, parameters_list)

        if len(parameters) == 0:
            parameters_table = "$$\\varnothing$$"

        # Generate the code for the equations
        variables, spike_condition, spike_reset = LatexParser._process_neuron_equations(neuron)
        
        eqs = _process_variables(variables, neuron=True)

        # Spiking neurons
        if neuron.type == 'spike':

            reset_txt = "* Emit a spike a time $t$.\n"
            for r in spike_reset:
                reset_txt += "* $" + r + "$\n"

            eqs += """
**Spike emission:**

if $%(condition)s$ :

%(reset)s
""" % {'condition': spike_condition, 'reset': reset_txt}


        # Possible function
        if not neuron.functions == None:
            eqs += """
**Functions**

%(functions)s
""" % {'functions': LatexParser._process_functions(neuron.functions, begin="$$", end="$$\n\n")}

        # Finalize the template
        txt += neuron_tpl % {   'name': neuron_name, 
                                'description': description, 
                                'parameters': parameters_table,
                                'eqs': eqs}

    return txt

def _adapt_locality_neuron(l):
    d = {
        'local': "per neuron",
        'semiglobal': "per population",
        'global': "per population"
    }
    return d[l]

# Synapse template
synapse_tpl = """
## %(name)s

%(description)s

**Parameters:**

%(parameters)s

**Equations:**

%(eqs)s
%(psp)s
"""
def _generate_synapse_models(net_id):
    txt = """
# Synapse models
"""

    for idx, synapse in enumerate(GlobalObjectManager().get_synapse_types()):

        # Do not document default synapses
        if synapse.name == "-":
            continue

        # Find a name for the synapse
        synapse_name = "Synapse " + str(synapse._rk_synapses_type) if synapse.name in Synapse._default_names.values() else synapse.name

        # Description
        description = synapse.short_description
        if description == None:
            description = "Spiking synapse." if synapse.type == 'spike' else 'Rate-coded synapse'

        # Parameters
        parameters = extract_parameters(synapse.parameters, synapse.extra_values)
        parameters_list = [
            ["$" + LatexParser._latexify_name(param['name'], []) + "$", param['init'], _adapt_locality_synapse(param['locality']), param['ctype']] 
                for param in parameters]

        parameters_headers = ["Name", "Default value", "Locality", "Type"]
        parameters_table = _make_table(parameters_headers, parameters_list)

        if len(parameters) == 0:
            parameters_table = "$$\\varnothing$$"

        # Generate the code for the equations
        psp, variables, pre_desc, post_desc = LatexParser._process_synapse_equations(synapse)

        eqs = _process_variables(variables, neuron = False)

        # PSP
        if synapse.type == "rate":
            psp = """
**Weighted sum:**

$$%(transmission)s$$
"""  % {'transmission': psp}
        elif synapse.type == "spike" and psp != "":
            psp = """
**Continuous transmission:**

$$%(transmission)s$$
"""  % {'transmission': psp}
        else:
            psp = ""

        # Pre- and post-events
        if synapse.type == "spike":
            if len(pre_desc) > 0:
                eqs += """
**Pre-synaptic event at $t_\\text{pre} + d$:**
"""
                for pre in pre_desc:
                    eqs += "$$"+pre+"$$\n"
            if len(post_desc) > 0:
                eqs += """
**Post-synaptic event at $t_\\text{post}$:**
"""
                for post in post_desc:
                    eqs += "$$"+post+"$$\n"

        # Finalize the template
        txt += synapse_tpl % {  'name': synapse_name, 
                                'description': description, 
                                'psp': psp,
                                'parameters': parameters_table,
                                'eqs': eqs}

    return txt

def _adapt_locality_synapse(l):
    d = {
        'local': "per synapse",
        'semiglobal': "per post-synaptic neuron",
        'global': "per projection"
    }
    return d[l]


def _generate_parameters(net_id, gather_subprojections):
    txt = """
# Parameters
"""

    # Constants
    if GlobalObjectManager().number_constants() > 0:
        txt += """
## Constants

"""
        constants_list = [
            ["$" + LatexParser._latexify_name(constant.name, []) + "$",  constant.value]
                for constant in GlobalObjectManager().get_constants()]

        constants_headers = ["Name", "Value"]
        txt += _make_table(constants_headers, constants_list)

    # Population parameters
    txt += """
## Population parameters

"""
    parameters_list = []
    for rk, pop in enumerate(NetworkManager().get_populations(net_id=net_id)):    
        neuron_name = "Neuron " + str(pop.neuron_type._rk_neurons_type) if pop.neuron_type.name in Neuron._default_names.values() \
            else pop.neuron_type.name

        for idx, param in enumerate(pop.parameters):
            val = pop.init[param]
            if isinstance(val, (list, np.ndarray)):
                val = "$[" + str(np.array(val).min()) + ", " + str(np.array(val).max()) + "]$"
            parameters_list.append(
                [   LatexParser.pop_name(pop.name) if idx==0 else "", 
                    neuron_name if idx==0 else "", 
                    "$" + LatexParser._latexify_name(param, []) + "$", 
                    val ] )

    population_headers = ["Population", "Neuron type", "Name", "Value"]
    txt += _make_table(population_headers, parameters_list)

    # Projection parameters
    txt += """
## Projection parameters

"""
    if gather_subprojections:
        projections = []
        for proj in NetworkManager().get_projections(net_id=net_id):
            for existing_proj in projections:
                if proj.pre.name == existing_proj.pre.name and proj.post.name == existing_proj.post.name \
                    and proj.target == existing_proj.target : # TODO
                    break
            else:
                projections.append(proj)
    else:
        projections = NetworkManager().get_projections(net_id=net_id)

    parameters_list = []
    for rk, proj in enumerate(projections):
        for idx, param in enumerate(proj.parameters):
            if param == 'w':
                continue
            if idx == 0:
                proj_name = "%(pre)s  $\\rightarrow$ %(post)s with target %(target)s" % {
                    'pre': LatexParser.pop_name(proj.pre.name), 
                    'post': LatexParser.pop_name(proj.post.name), 
                    'target': LatexParser._format_list(proj.target, ' / ')}
            else:
                proj_name = ""
            
            synapse_name = "Synapse " + str(proj.synapse_type._rk_synapses_type) if proj.synapse_type.name in Synapse._default_names.values() \
                else proj.synapse_type.name
            
            val = proj.init[param]
            if isinstance(val, (list, np.ndarray)):
                val = "$[" + str(np.array(val).min()) + ", " + str(np.array(val).max()) + "]$"
            parameters_list.append(
                [   proj_name, 
                    synapse_name if idx == 0 else "",
                    "$" + LatexParser._latexify_name(param, []) + "$", 
                    val ] )

    projection_headers = ["Projection", "Synapse type", "Name", "Value"]
    txt += _make_table(projection_headers, parameters_list)

    return txt



def _process_variables(variables, neuron=True):
    eqs = ""
    for var in variables:
        # Min value
        if 'min' in var['bounds'].keys():
            min_val = ", minimum: " + str(var['bounds']['min'])
        else:
            min_val =""
        # Max value
        if 'max' in var['bounds'].keys():
            max_val = ", maximum: " + str(var['bounds']['max'])
        else:
            max_val =""
        # Method
        if var['ode']:
            method = ", " + find_method(var) + " numerical method"
        else:
            method = ""

        eqs += """
* Variable %(name)s : %(locality)s, initial value: %(init)s%(min)s%(max)s%(method)s

$$
%(code)s
$$
""" % { 'name': "$" + LatexParser._latexify_name(var['name'], []) + "$", 
        'code': var['latex'],
        'locality': _adapt_locality_neuron(var['locality']) if neuron else _adapt_locality_synapse(var['locality']),
        'init': var['init'],
        'min': min_val,
        'max': max_val,
        'method': method}

    return eqs

def _make_table(header, data):
    "Creates a markdown table from the data, with headers."

    nb_col = len(header)
    nb_data = len(data)

    # Compute the maximum size of each column
    max_size = [len(header[c]) + 4 for c in range(nb_col)]
    for c in range(nb_col):
        for e in range(nb_data):
            max_size[c] = max(max_size[c], len(str(data[e][c])))

    # Create the table
    table= "| "
    for c in range(nb_col):
        table += "**" + header[c] + "**" + " "*(max_size[c] - len(header[c]) - 4) + " | "
    table += "\n| "
    for c in range(nb_col):
        table += "-"*max_size[c] + " | "
    table += "\n"
    for e in range(nb_data):
        table += "| "
        for c in range(nb_col):
            table += str(data[e][c]) + " "*(max_size[c] - len(str(data[e][c]))) + " | "    
        table += "\n"    
    table += "\n"


    return table

