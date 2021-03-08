import ANNarchy.core.Global as Global
import ANNarchy.parser.report.LatexParser as LatexParser
from ANNarchy.core.Neuron import Neuron
from ANNarchy.core.Synapse import Synapse

import numpy as np
import os

##################################
### Templates
##################################

header = """
%  LaTeX file for generating the Model Description Table in Fig. 5 of
%
%  Nordlie E, Gewaltig M-O, Plesser HE (2009)
%  Towards Reproducible Descriptions of Neuronal Network Models.
%  PLoS Comput Biol 5(8): e1000456.
%
%  Paper URL : http://dx.doi.org/10.1371/journal.pcbi.1000456
%  Figure URL: http://dx.doi.org/10.1371/journal.pcbi.1000456.g005
%
%  This file is released under a
%
%   Creative Commons Attribution, non-commercial, share-alike licence
%   http://creativecommons.org/licenses/by-nc-sa/3.0/de/deed.en
%
%  with the following specifications:
%
%  1. When publishing tables generated from this LaTeX file and modified
%     versions of it, you must cite the paper by Nordlie et al given above.
%
%  2. The non-commercial clause applies only to the distribution of THIS FILE
%     and LaTeX source code files derived from it. You may commercially publish
%     documents generated using this file and derivatived versions of this file.
%
%  Contact: Hans Ekkehard Plesser, UMB (hans.ekkehard.plesser at umb.no)
"""

preamble = """
\\documentclass{article}
\\usepackage[margin=1in]{geometry}
\\usepackage{tabularx}
\\usepackage{multirow}
\\usepackage{colortbl}

\\usepackage[fleqn]{amsmath}
\\setlength{\\mathindent}{0em}
%%\\usepackage{mathpazo}
\\usepackage{breqn}

\\usepackage[scaled=.95]{helvet}
\\renewcommand\\familydefault{\\sfdefault}

\\renewcommand\\arraystretch{1.2}
\\pagestyle{empty}

\\newcommand{\hdr}[3]{
    \\multicolumn{#1}{|l|}{
        \\color{white}\\cellcolor[gray]{0.0}
        \\textbf{\makebox[0pt]{#2}\\hspace{0.5\\linewidth}\\makebox[0pt][c]{#3}}
    }
}

\\begin{document}
"""

summary_template="""
\\noindent
\\begin{tabularx}{\\linewidth}{|l|X|}\\hline
\\hdr{2}{A}{Model Summary}\\\\ \\hline
\\textbf{Populations}     & %(population_names)s \\\\ \\hline
\\textbf{Topology}        & --- \\\\ \\hline
\\textbf{Connectivity}    & %(connectivity)s \\\\ \\hline
\\textbf{Neuron models}   & %(neuron_models)s \\\\ \\hline
\\textbf{Channel models}  & --- \\\\ \\hline
\\textbf{Synapse models}  & --- \\\\ \\hline
\\textbf{Plasticity}      & %(synapse_models)s\\\\ \\hline
\\textbf{Input}           & --- \\\\ \\hline
\\textbf{Measurements}    & --- \\\\ \\hline
\\end{tabularx}

\\vspace{2ex}
"""

populations_template = """
\\noindent
\\begin{tabularx}{\\linewidth}{|l|l|X|}\\hline
\\hdr{3}{B}{Populations}\\\\ \\hline
    \\textbf{Name}   & \\textbf{Elements} & \\textbf{Size} \\\\ \\hline
%(populations_description)s
\\end{tabularx}

\\vspace{2ex}
"""

connectivity_template = """
\\noindent
\\begin{tabularx}{\\linewidth}{|l|l|l|X|X|}\\hline
\\hdr{5}{C}{Connectivity}\\\\ \\hline
\\textbf{Source} & \\textbf{Destination} & \\textbf{Target} & \\textbf{Synapse} & \\textbf{Pattern} \\\\ \\hline
%(projections_description)s
\\end{tabularx}

\\vspace{2ex}
"""


parameters_template = """
\\noindent
\\begin{tabularx}{\\linewidth}{|X|}\\hline
\hdr{1}{F}{Parameters}\\\\ \\hline
\\\\ \\hline
\\end{tabularx}
\\vspace{2ex}
"""

constants_template = """
\\noindent
\\begin{tabularx}{\\linewidth}{|p{0.25\\linewidth}|p{0.25\\linewidth}|X|}\\hline
\\textbf{Constants} &\\textbf{Name} & \\textbf{Value}   \\\\ \\hline
%(parameters)s
\\end{tabularx}

\\vspace{2ex}
"""

functions_template = """
\\noindent
\\begin{tabularx}{\\linewidth}{|p{0.25\\linewidth}|X|}\\hline
\\textbf{Functions} &  
%(parameters)s
\\\\ \\hline
\\end{tabularx}

\\vspace{2ex}
"""

popparameters_template = """
\\noindent
\\begin{tabularx}{\\linewidth}{|p{0.25\\linewidth}|p{0.25\\linewidth}|X|}\\hline
\\textbf{Population} & \\textbf{Parameter} & \\textbf{Value}   \\\\ \\hline
%(parameters)s
\\end{tabularx}

\\vspace{2ex}
"""

projparameters_template = """
\\noindent
\\begin{tabularx}{\\linewidth}{|p{0.25\\linewidth}|p{0.25\\linewidth}|X|}\\hline
\\textbf{Projection} & \\textbf{Parameter} & \\textbf{Value}   \\\\ \\hline
%(parameters)s
\\end{tabularx}

\\vspace{2ex}
"""

footer = """
\\noindent\\begin{tabularx}{\\linewidth}{|l|X|}\\hline
\\hdr{2}{G}{Input}\\\\ \\hline
\\textbf{Type} & \\textbf{Description} \\\\ \\hline
---
\\\\ \\hline
\\end{tabularx}

\\vspace{2ex}

\\noindent\\begin{tabularx}{\\linewidth}{|X|}\\hline
\\hdr{1}{H}{Measurements}\\\\ \\hline
---
\\\\ \\hline
\\end{tabularx}

\\end{document}
"""

##################################
### Main method
##################################

def report_latex(filename="./report.tex", standalone=True, gather_subprojections=False, net_id=0):
    """ Generates a .tex file describing the network according to:

    Nordlie E, Gewaltig M-O, Plesser HE (2009). Towards Reproducible Descriptions of Neuronal Network Models. PLoS Comput Biol 5(8): e1000456.

    **Parameters:**

    * *filename*: name of the .tex file where the report will be written (default: "./report.tex")
    * *standalone*: tells if the generated file should be directly compilable or only includable (default: True)
    * *gather_subprojections*: if a projection between two populations has been implemented as a multiple of projections between sub-populations, this flag allows to group them in the summary (default: False).
    * *net_id*: id of the network to be used for reporting (default: 0, everything that was declared)
    """

    # stdout
    Global._print('Generating report in', filename)

    # Generate the summary
    summary = _generate_summary(net_id)
    # Generate the populations
    populations = _generate_populations(net_id)
    # Generate the projections
    projections = _generate_projections(net_id, gather_subprojections)
    # Generate the neuron models
    neuron_models = _generate_neuron_models(net_id)
    # Generate the synapse models
    synapse_models = _generate_synapse_models(net_id)
    # Generate the constants
    constants = _generate_constants(net_id)
    # Generate the functions
    functions = _generate_functions(net_id)
    # Generate the population parameters
    pop_parameters = _generate_population_parameters(net_id)
    # Generate the population parameters
    proj_parameters = _generate_projection_parameters(net_id, gather_subprojections)

    # Possibly create the directory if it does not exist
    path_name = os.path.dirname(filename)
    if not path_name in ["", "."]:
        if not os.path.exists(path_name):
            os.makedirs(path_name)

    with open(filename, 'w') as wfile:
        if standalone:
            wfile.write(header)
            wfile.write(preamble)
        wfile.write(summary)
        wfile.write(populations)
        wfile.write(projections)
        wfile.write(neuron_models)
        wfile.write(synapse_models)
        wfile.write(parameters_template)
        wfile.write(constants)
        wfile.write(functions)
        wfile.write(pop_parameters)
        wfile.write(proj_parameters)
        if standalone:
            wfile.write(footer)

##################################
### Process major fields
##################################

def _generate_summary(net_id):
    "part A"

    population_names = str(len(Global._network[net_id]['populations'])) + ': '
    connectivity = ""
    neuron_models = ""
    synapse_models = ""

    # List the names of all populations
    for pop in Global._network[net_id]['populations']:
        # population name
        population_names += LatexParser.pop_name(pop.name) + ", "
    population_names = population_names[:-2] # suppress the last ,

    # List all neuron types
    neuron_model_names = []
    for neur in Global._objects['neurons']:
        neuron_model_names.append(neur.name)
    for neur in list(set(neuron_model_names)):
        neuron_models += neur + ', '
    neuron_models = neuron_models[:-2] # suppress the last ,

    list_connectivity = []
    list_synapse_models = []
    for proj in Global._network[net_id]['projections']:
        list_connectivity.append(proj.connector_name)
        if not proj.synapse_type.name in list(Synapse._default_names.values()) + ['-']:
            list_synapse_models.append(proj.synapse_type.name)
    for con in list(set(list_connectivity)):
        connectivity += con + ', '
    for syn in list(set(list_synapse_models)):
        synapse_models += syn + ', '
    connectivity = connectivity[:-2]
    synapse_models = synapse_models[:-2] # suppress the last ,


    # Write the summary
    txt = summary_template  % {
        'population_names' : population_names,
        'connectivity' : connectivity,
        'neuron_models' : neuron_models,
        'synapse_models' : synapse_models
    }
    return txt

def _generate_populations(net_id):
    def format_size(pop):
        size = str(pop.size)
        if pop.dimension >1:
            size += ' ('
            for d in range(pop.dimension):
                size += str(pop.geometry[d]) + '*'
            size = size.rsplit('*', 1)[0] + ')'
        return size

    txt = ""
    pop_tpl = """
    %(pop_name)s             & %(neuron_type)s        & $N_{\\text{%(pop_name)s}}$ = %(size)s  \\\\ \\hline
"""
    for pop in Global._network[net_id]['populations']:
        # Find a name for the neuron
        if pop.neuron_type.name in Neuron._default_names.values(): # name not set
            neuron_name = "Neuron " + str(pop.neuron_type._rk_neurons_type)
        else:
            neuron_name = pop.neuron_type.name

        txt += pop_tpl % {
            'pop_name': LatexParser.pop_name(pop.name), 
            'neuron_type': neuron_name, 
            'size': format_size(pop)}

    return populations_template % {'populations_description': txt}

def _generate_constants(net_id):
    cst_tpl = """
    & $%(param)s$        & %(value)s  \\\\ \\hline
"""
    parameters = ""
    if len(Global._objects['constants']) == 0:
        return ""

    for constant in Global._objects['constants']:
        parameters += cst_tpl % {'param': LatexParser._latexify_name(constant.name, []), 'value': constant.value}

    txt = constants_template % {'parameters': parameters}

    return txt


def _generate_functions(net_id):

    functions = ""
    if len(Global._objects['functions']) == 0:
        return functions

    for name, func in Global._objects['functions']:
        functions += LatexParser._process_functions(func) + "\n"

    return functions_template % {'parameters': functions, 'firstfunction': "\hdr{1}{G}{Functions}\\\\ \\hline"}

def _generate_population_parameters(net_id):
    txt = ""
    pop_tpl = """
    %(name)s             & $%(param)s$        & %(value)s  \\\\ \\hline
"""
    for rk, pop in enumerate(Global._network[net_id]['populations']):
        parameters = ""
        for idx, param in enumerate(pop.parameters):
            val = pop.init[param]
            if isinstance(val, (list, np.ndarray)):
                val = "$[" + str(np.array(val).min()) + ", " + str(np.array(val).max()) + "]$"
            parameters += pop_tpl % {'name': LatexParser.pop_name(pop.name) if idx==0 else "", 'param': LatexParser._latexify_name(param, []), 'value': val}

        txt += popparameters_template % {'parameters': parameters, 'firstpopulation': "\hdr{3}{H}{Population parameters}\\\\ \\hline" if rk==0 else ""}

    return txt

def _generate_projections(net_id, gather_subprojections):
    txt = ""
    proj_tpl = """
    %(pre)s & %(post)s & %(target)s & %(synapse)s &
    %(description)s \\\\ \\hline
"""
    if gather_subprojections:
        projections = []
        for proj in Global._network[net_id]['projections']:
            for existing_proj in projections:
                if proj.pre.name == existing_proj.pre.name and proj.post.name == existing_proj.post.name and proj.target == existing_proj.target : # TODO
                    break
            else:
                projections.append(proj)
    else:
        projections = Global._network[net_id]['projections']

    for proj in projections:
        # Find a name for the synapse
        if proj.synapse_type.name in Synapse._default_names.values(): # name not set
            synapse_name = "Synapse " + str(proj.synapse_type._rk_synapses_type)
        else:
            synapse_name = proj.synapse_type.name

        txt += proj_tpl % { 'pre': LatexParser.pop_name(proj.pre.name), 
                            'post': LatexParser.pop_name(proj.post.name), 
                            'target': LatexParser._format_list(proj.target, ' / '),
                            'synapse': synapse_name,
                            'description': proj.connector_description}

    return connectivity_template % {'projections_description': txt}

def _generate_projection_parameters(net_id, gather_subprojections):
    txt = ""
    proj_tpl = """
    %(name)s & $%(param)s$        & %(value)s  \\\\ \\hline
"""
    if gather_subprojections:
        projections = []
        for proj in Global._network[net_id]['projections']:
            for existing_proj in projections:
                if proj.pre.name == existing_proj.pre.name and proj.post.name == existing_proj.post.name and proj.target == existing_proj.target : # TODO
                    break
            else:
                projections.append(proj)
    else:
        projections = Global._network[net_id]['projections']

    first = True
    for rk, proj in enumerate(projections):
        parameters = ""
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
            val = proj.init[param]
            
            if isinstance(val, (list, np.ndarray)):
                val = "$[" + str(np.min(val)) + ", " + str(np.max(val)) + "]$"
            parameters += proj_tpl % {'name': proj_name, 'param': LatexParser._latexify_name(param, []), 'value': val}

        if parameters != "":
            txt += projparameters_template % {'parameters': parameters, 'firstprojection': "\hdr{3}{H}{Projection parameters}\\\\ \\hline" if first else ""}
            first = False

    return txt

def _generate_neuron_models(net_id):
    neurons = ""

    firstneuron = "\\hdr{2}{D}{Neuron Models}\\\\ \\hline"

    neuron_tpl = """
\\noindent
\\begin{tabularx}{\\linewidth}{|p{0.15\\linewidth}|X|}\\hline
%(firstneuron)s
\\textbf{Name} & %(name)s \\\\ \\hline
\\textbf{Type} & %(description)s\\\\ \\hline
\\textbf{%(equation_type)s} &
%(variables)s
%(spike)s
%(functions)s
\\end{tabularx}
\\vspace{2ex}
"""
    for idx, neuron in enumerate(Global._objects['neurons']):

        # Name
        if neuron.name in Neuron._default_names.values(): # name not set
            neuron_name = "Neuron " + str(neuron._rk_neurons_type)
        else:
            neuron_name = neuron.name

        # Generate the code for the equations
        variables, spike_condition, spike_reset = LatexParser._process_neuron_equations(neuron)

        eqs = ""
        for var in variables:
            eqs += """
\\begin{dmath*}
%(eq)s
\\end{dmath*}
""" % {'eq': var['latex']}

        variables_eqs = """
%(eqs)s
\\\\ \\hline
""" % {'eqs': eqs}

        # Spiking neurons have an extra field for the spike condition
        spike_extra = ""
        if neuron.type == 'spike':
            spike_code = "If $" + spike_condition + "$ or $t \leq t^* + t_{\\text{refractory}}$:"

            # Reset
            spike_code += """
            \\begin{enumerate}
                \\item Emit a spike at time $t^*$"""

            for var in spike_reset:
                spike_code += """
            \\item $""" + var + "$"

            spike_code += """
        \\end{enumerate}"""


            spike_extra = """
\\textbf{Spiking} &
%(spike)s
\\\\ \\hline
""" % {'spike': spike_code}

        # Possible function
        functions = ""
        if not neuron.functions == None:
            functions = """
\\textbf{Functions} &
%(functions)s
\\\\ \\hline
""" % {'functions': LatexParser._process_functions(neuron.functions)}

        # Build the dictionary
        desc = {
            'name': neuron_name,
            'description': neuron.short_description,
            'firstneuron': firstneuron if idx ==0 else "",
            'variables': variables_eqs,
            'spike': spike_extra,
            'functions': functions,
            'equation_type': "Subthreshold dynamics" if neuron.type == 'spike' else 'Equations'
        }

        # Generate the code depending on the neuron position
        neurons += neuron_tpl % desc

    return neurons

def _generate_synapse_models(net_id):
    firstsynapse = ""
    synapses = ""

    firstsynapse = "\\hdr{2}{E}{Synapse Models}\\\\ \\hline"

    synapse_tpl = """
\\noindent
\\begin{tabularx}{\\linewidth}{|p{0.15\\linewidth}|X|}\\hline
%(firstsynapse)s
\\textbf{Name} & %(name)s \\\\ \\hline
\\textbf{Type} & %(description)s\\\\ \\hline
%(psp)s
%(variables)s
%(preevent)s
%(postevent)s
%(functions)s
\\end{tabularx}
\\vspace{2ex}
"""
    for idx, synapse in enumerate(Global._objects['synapses']):
        # Do not document default synapses
        if synapse.name == "-":
            continue

        # Find a name for the synapse
        if synapse.name in Synapse._default_names.values(): # name not set
            synapse_name = "Synapse " + str(synapse._rk_synapses_type)
        else:
            synapse_name = synapse.name

        # Generate the code for the equations
        psp, variables, pre_desc, post_desc = LatexParser._process_synapse_equations(synapse)

        eqs = ""
        for var in variables:
            eqs += """
\\begin{dmath*}
%(eq)s
\\end{dmath*}
""" % {'eq': var['latex']}

        variables_eqs = """
%(eqs)s
\\\\ \\hline
""" % {'eqs': eqs}

        # Synaptic variables
        variables = """
\\textbf{Equations} & %(variables)s  
\\\\ \\hline""" % {'variables':eqs} if eqs != "" else ""

        # PSP
        if psp != "":
            psp_code = """
\\textbf{PSP} & \\begin{dmath*}
%(psp)s
\\end{dmath*}
\\\\ \\hline""" % {'psp': psp}
        else:
            psp_code = ""

        # Spiking neurons have extra fields for the event-driven
        if synapse.type == 'spike':
            if len(pre_desc) > 0:
                txt_pre = ""
                for l in pre_desc:
                    txt_pre += """
\\begin{dmath*}
%(eq)s
\\end{dmath*}
""" % {'eq': l}
                preevent = """
\\textbf{Pre-synaptic event} &
%(preevent)s
\\\\ \\hline
""" % {'preevent': txt_pre}
            else:
                preevent = ""

            if len(post_desc) > 0:
                txt_post = ""
                for l in post_desc:
                    txt_post += """
\\begin{dmath*}
%(eq)s
\\end{dmath*}
""" % {'eq': l}
                postevent = """
\\textbf{Post-synaptic event} &
%(postevent)s
\\\\ \\hline
""" % {'postevent': txt_post}
            else:
                postevent = ""
        else:
            preevent = ""
            postevent = ""

        # Possible functions
        functions = ""
        if not synapse.functions == None:
            functions = """
\\textbf{Functions} &
%(functions)s
\\\\ \\hline
""" % {'functions': LatexParser._process_functions(synapse.functions)}

        # Build the dictionary
        desc = {
            'name': synapse_name,
            'description': synapse.short_description,
            'firstsynapse': firstsynapse if idx == 0 else "",
            'variables': variables,
            'psp': psp_code,
            'preevent': preevent,
            'postevent': postevent,
            'functions': functions
        }

        # Generate the code
        synapses += synapse_tpl % desc

    return synapses

