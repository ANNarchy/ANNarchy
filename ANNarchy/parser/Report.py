from ANNarchy.core.Global import _network, _objects, _warning, _error, _print
from ANNarchy.core.Random import RandomDistribution
from .Extraction import *
from ANNarchy.core.Synapse import Synapse

from sympy import *
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor, auto_number
import re
import numpy as np



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

popparameters_template = """
\\noindent
\\begin{tabularx}{\\linewidth}{|p{0.15\\linewidth}|p{0.15\\linewidth}|X|}\\hline
%(firstpopulation)s
\\textbf{Population} & \\textbf{Parameter} & \\textbf{Value}   \\\\ \\hline
%(parameters)s
\\end{tabularx}

\\vspace{2ex}
"""

projparameters_template = """
\\noindent
\\begin{tabularx}{\\linewidth}{|p{0.25\\linewidth}|p{0.15\\linewidth}|X|}\\hline
%(firstprojection)s
\\textbf{Projection} & \\textbf{Parameter} & \\textbf{Value}   \\\\ \\hline
%(parameters)s
\\end{tabularx}

\\vspace{2ex}
"""

footer = """
\\noindent\\begin{tabularx}{\\linewidth}{|l|X|}\\hline
\\hdr{2}{H}{Input}\\\\ \\hline
\\textbf{Type} & \\textbf{Description} \\\\ \\hline
---
\\\\ \\hline
\\end{tabularx}

\\vspace{2ex}

\\noindent\\begin{tabularx}{\\linewidth}{|X|}\\hline
\\hdr{1}{I}{Measurements}\\\\ \\hline
---
\\\\ \\hline
\\end{tabularx}

\\end{document}
"""

##################################
### Main method
##################################

def report(filename="./report.tex", standalone=True, gather_subprojections=False, net_id=0):
    """ Generates a .tex file describing the network according to:

    Nordlie E, Gewaltig M-O, Plesser HE (2009). Towards Reproducible Descriptions of Neuronal Network Models. PLoS Comput Biol 5(8): e1000456.

    **Parameters:**

    * *filename*: name of the .tex file where the report will be written (default: "./report.tex")
    * *standalone*: tells if the generated file should be directly compilable or only includable (default: True)
    * *gather_subprojections*: if a projection between two populations has been implemented as a multiple of projections between sub-populations, this flag allows to group them in the summary (default: False).
    * *net_id*: id of the network to be used for reporting (default: 0, everything that was declared)
    """

    # stdout
    _print('Generating report in', filename)

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
    # Generate the population parameters
    pop_parameters = _generate_population_parameters(net_id)
    # Generate the population parameters
    proj_parameters = _generate_projection_parameters(net_id, gather_subprojections)

    with open(filename, 'w') as wfile:
        if standalone:
            wfile.write(header)
            wfile.write(preamble)
        wfile.write(summary)
        wfile.write(populations)
        wfile.write(projections)
        wfile.write(neuron_models)
        wfile.write(synapse_models)
        wfile.write(pop_parameters)
        wfile.write(proj_parameters)
        if standalone:
            wfile.write(footer)

##################################
### Process major fields
##################################

def _generate_summary(net_id):
    "part A"

    population_names = str(len(_network[net_id]['populations'])) + ': '
    connectivity = ""
    neuron_models = ""
    synapse_models = ""

    # List the names of all populations
    for pop in _network[net_id]['populations']:
        # population name
        population_names += pop_name(pop.name) + ", "
    population_names = population_names[:-2] # suppress the last ,

    # List all neuron types
    neuron_model_names = []
    for neur in _objects['neurons']:
        neuron_model_names.append(neur.name)
    for neur in list(set(neuron_model_names)):
        neuron_models += neur + ', '
    neuron_models = neuron_models[:-2] # suppress the last ,

    list_connectivity = []
    list_synapse_models = []
    for proj in _network[net_id]['projections']:
        list_connectivity.append(proj.connector_name)
        if not proj.synapse_type.name in list(Synapse._default_names.values()) + ['Static weight']:
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
    for pop in _network[net_id]['populations']:
        txt += pop_tpl % {'pop_name': pop_name(pop.name), 'neuron_type': pop.neuron_type.name, 'size': format_size(pop)}

    return populations_template % {'populations_description': txt}

def _generate_population_parameters(net_id):
    txt = ""
    pop_tpl = """
    %(name)s             & $%(param)s$        & %(value)s  \\\\ \\hline
"""
    for rk, pop in enumerate(_network[net_id]['populations']):
        parameters = ""
        for idx, param in enumerate(pop.parameters):
            val = pop.init[param]
            if isinstance(val, (list, np.ndarray)):
                val = "$[" + str(min(val)) + ", " + str(max(val)) + "]$"
            parameters += pop_tpl % {'name': pop_name(pop.name) if idx==0 else "", 'param': _latexify_name(param, []), 'value': val}

        txt += popparameters_template % {'parameters': parameters, 'firstpopulation': "\hdr{3}{F}{Population parameters}\\\\ \\hline" if rk==0 else ""}

    return txt

def _generate_projections(net_id, gather_subprojections):
    txt = ""
    proj_tpl = """
    %(pre)s & %(post)s & %(target)s & %(synapse)s &
    %(description)s \\\\ \\hline
"""
    if gather_subprojections:
        projections = []
        for proj in _network[net_id]['projections']:
            for existing_proj in projections:
                if proj.pre.name == existing_proj.pre.name and proj.post.name == existing_proj.post.name and proj.target == existing_proj.target : # TODO
                    break
            else:
                projections.append(proj)
    else:
        projections = _network[net_id]['projections']

    for proj in projections:
        if not proj.synapse_type.name in Synapse._default_names.values():
            name = proj.synapse_type.name
        else:
            name = "-"
        txt += proj_tpl % {'pre': pop_name(proj.pre.name), 'post': pop_name(proj.post.name), 'target': proj.target,
                            'synapse': name,
                            'description': proj.connector_description}

    return connectivity_template % {'projections_description': txt}



def _generate_projection_parameters(net_id, gather_subprojections):
    txt = ""
    proj_tpl = """
    %(name)s & $%(param)s$        & %(value)s  \\\\ \\hline
"""
    if gather_subprojections:
        projections = []
        for proj in _network[net_id]['projections']:
            for existing_proj in projections:
                if proj.pre.name == existing_proj.pre.name and proj.post.name == existing_proj.post.name and proj.target == existing_proj.target : # TODO
                    break
            else:
                projections.append(proj)
    else:
        projections = _network[net_id]['projections']

    first = True
    for rk, proj in enumerate(projections):
        parameters = ""
        for idx, param in enumerate(proj.parameters):
            if param == 'w':
                continue
            if idx == 0:
                proj_name = "%(pre)s  $\\rightarrow$ %(post)s with target %(target)s" % {'pre': pop_name(proj.pre.name), 'post': pop_name(proj.post.name), 'target': proj.target}
            else:
                proj_name = ""
            val = proj.init[param]
            if isinstance(val, (list, np.ndarray)):
                val = "$[" + str(min(val)) + ", " + str(max(val)) + "]$"
            parameters += proj_tpl % {'name': proj_name, 'param': _latexify_name(param, []), 'value': val}

        if parameters != "":
            txt += projparameters_template % {'parameters': parameters, 'firstprojection': "\hdr{3}{G}{Projection parameters}\\\\ \\hline" if first else ""}
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
\\\\ \\hline
\\end{tabularx}
\\vspace{2ex}
"""
    for idx, neuron in enumerate(_objects['neurons']):
        # Generate the code for the equations
        eqs, spike_txt = _process_neuron_equations(neuron)

        # Spiking neurons have an extra field for the spike condition
        if neuron.type == 'spike':
            spike_extra = """
\\\\ \\hline
\\textbf{Spiking} &
%(spike)s
"""
            eqs += spike_extra % {'spike': spike_txt}


        # Build the dictionary
        desc = {
            'name': neuron.name,
            'description': neuron.short_description,
            'firstneuron': firstneuron if idx ==0 else "",
            'variables': eqs,
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
\\end{tabularx}
\\vspace{2ex}
"""
    for idx, synapse in enumerate(_objects['synapses']):
        # Generate the code for the equations
        psp, eqs, pre_desc, post_desc = _process_synapse_equations(synapse)

        # Synaptic variables
        variables = "\\textbf{Equations} & %(variables)s  \\\\ \\hline" % {'variables':eqs} if eqs != "" else ""

        # PSP
        if psp != "":
            psp_code = """
\\textbf{PSP} & %(psp)s\\\\ \\hline""" % {'psp': psp}
        else:
            psp_code = ""

        # Spiking neurons have extra fields for the event-driven
        if synapse.type == 'spike':
            if pre_desc != "":
                preevent = """
\\textbf{Pre-synaptic event} &
%(preevent)s
\\\\ \\hline
""" % {'preevent': pre_desc}
            else:
                preevent = ""
            if post_desc != "":
                postevent = """
\\textbf{Post-synaptic event} &
%(postevent)s
\\\\ \\hline
""" % {'postevent': post_desc}
            else:
                postevent = ""
        else:
            preevent = ""
            postevent = ""

        # Build the dictionary
        desc = {
            'name': synapse.name,
            'description': synapse.short_description,
            'firstsynapse': firstsynapse if idx == 0 else "",
            'variables': variables,
            'psp': psp_code,
            'preevent': preevent,
            'postevent': postevent
        }

        # Generate the code
        synapses += synapse_tpl % desc

    return synapses

##################################
### Process individual equations
##################################

def _process_random(val):
    "Transforms a connector attribute (weights, delays) into a string representation"
    if isinstance(val, RandomDistribution):
        return val.latex()
    else:
        return str(val)

# Really crappy...
# When target has a number (ff1), sympy thinks the 1 is a number
# the target is replaced by a text to avoid this
target_replacements = [
    'firsttarget',
    'secondtarget',
    'thirdtarget',
    'fourthtarget',
    'fifthtarget',
    'sixthtarget',
    'seventhtarget',
    'eighthtarget',
    'ninthtarget',
    'tenthtarget',
]

def _process_neuron_equations(neuron):
    code = ""

    # Extract parameters and variables
    parameters = extract_parameters(neuron.parameters, neuron.extra_values)
    variables = extract_variables(neuron.equations)
    variable_names = [var['name'] for var in variables]
    attributes, local_var, global_var = get_attributes(parameters, variables)

    # Create a dictionary for parsing
    local_dict = {
        'g_target': Symbol('g_{\\text{target}}'),
        't_pre': Symbol('t_{\\text{pre}}'),
        't_post': Symbol('t_{\\text{pos}}'),
        'Uniform': Function('\mathcal{U}'),
        'Normal': Function('\mathcal{N}'),
    }

    for att in attributes:
        local_dict[att] = Symbol(_latexify_name(att, variable_names))

    tex_dict = {}
    for key, val in local_dict.items():
        tex_dict[val] = str(val)

    for var in variables:
        # Retrieve the equation
        eq = var['eq']
        eq = eq.replace(' ', '') # supress spaces

        # Extract sum(target)
        targets = []
        target_list = re.findall('(?P<pre>[^\w.])sum\(\s*([^()]+)\s*\)', eq)
        for l, t in target_list:
            if t.strip() == '':
                continue
            targets.append((t.strip(), target_replacements[len(targets)]))
        for target, repl in targets:
            eq = eq.replace('sum('+target+')', repl)

        # Parse the equation
        ode = re.findall(r'([^\w]*)d([\w]+)/dt', eq)
        if len(ode) > 0:
            name = ode[0][1]
            eq = eq.replace('d'+name+'/dt', '_grad_'+name)
            grad_symbol = Symbol('\\frac{d'+_latexify_name(name, variable_names)+'}{dt}')
            local_dict['_grad_'+name] = grad_symbol
            tex_dict[grad_symbol] = '\\frac{d'+_latexify_name(name, variable_names)+'}{dt}'

        var_code = _analyse_equation(var['eq'], eq, local_dict, tex_dict)

        # Replace the targets
        for target, repl in targets:
            var_code = var_code.replace(repl, '\\sum_{\\text{'+target+'}} \\text{psp}(t)')


        # Add the code
        code += """\\begin{dmath*}
%(eq)s
\\end{dmath*}
""" % {'eq': var_code}

    if not neuron.spike: # rate-code, no spike
        return code, ""

    # Additional code for spiking neurons
    spike_code = "If $" + _analyse_part(neuron.spike, local_dict, tex_dict) + "$ or $t \leq t^* + t_{\\text{refractory}}$:"

    # Reset
    spike_code += """
    \\begin{enumerate}
        \\item Emit a spike at time $t^*$"""

    reset_vars = extract_spike_variable(neuron.description)['spike_reset']
    for var in reset_vars:
        eq = var['eq']
        spike_code += """
        \\item $""" + _analyse_equation(var['eq'], eq, local_dict, tex_dict) + "$"

    spike_code += """
    \\end{enumerate}"""

    return code, spike_code


def _process_synapse_equations(synapse):
    psp = ""
    code = ""
    pre_event = ""
    post_event = ""

    # Extract parameters and variables
    parameters = extract_parameters(synapse.parameters)
    variables = extract_variables(synapse.equations)
    variable_names = [var['name'] for var in variables]
    attributes, local_var, global_var = get_attributes(parameters, variables)

    # Create a dictionary for parsing
    local_dict = {
        'w': Symbol('w(t)'),
        'g_target': Symbol('g_{\\text{target}(t)}'),
        't_pre': Symbol('t_{\\text{pre}}'),
        't_post': Symbol('t_{\\text{pos}}'),
        'Uniform': Function('\mathcal{U}'),
        'Normal': Function('\mathcal{N}'),
    }

    for att in attributes:
        local_dict[att] = Symbol(_latexify_name(att, variable_names))

    tex_dict = {}
    for key, val in local_dict.items():
        tex_dict[val] = str(val)


    # PSP
    if synapse.psp:
        psp, untouched_var, dependencies = extract_prepost('psp', synapse.psp.strip(), synapse.description)
        for dep in dependencies['post']:
            local_dict['_post_'+dep] = Symbol("{" + dep + "^{\\text{post}}}(t)")
        for dep in dependencies['pre']:
            local_dict['_pre_'+dep] = Symbol("{" + dep + "^{\\text{pre}}}(t)")
        psp = "$" + _analyse_part(psp, local_dict, tex_dict) + "$"
    else:
        if synapse.type == 'rate':
            psp = "$w(t) \cdot r^{\\text{pre}}(t)$"
        else:
            psp = ""


    # Variables
    for var in variables:
        # Retrieve the equation
        eq = var['eq']

        # pre/post variables
        targets=[]
        eq, untouched_var, dependencies = extract_prepost(var['name'], eq, synapse.description)
        for dep in dependencies['post']:
            if dep.startswith('sum('):
                target = re.findall(r'sum\(([\w]+)\)', dep)[0]
                targets.append(target)
                local_dict['_post_sum_'+target] = Symbol('PostSum'+target)
            else:
                local_dict['_post_'+dep] = Symbol("{" + dep + "^{\\text{post}}}(t)")
        for dep in dependencies['pre']:
            if dep.startswith('sum('):
                target = re.findall(r'sum\(([\w]+)\)', dep)[0]
                targets.append(target)
                local_dict['_pre_sum_'+target] = Symbol('PreSum'+target)
            else:
                local_dict['_pre_'+dep] = Symbol("{" + dep + "^{\\text{pre}}}(t)")

        # Parse the equation
        eq = eq.replace(' ', '') # supress spaces
        ode = re.findall(r'([^\w]*)d([\w]+)/dt', eq)
        if len(ode) > 0:
            name = ode[0][1]
            eq = eq.replace('d'+name+'/dt', '_grad_'+name)
            grad_symbol = Symbol('\\frac{d'+_latexify_name(name, variable_names)+'}{dt}')
            local_dict['_grad_'+name] = grad_symbol
            tex_dict[grad_symbol] = '\\frac{d'+_latexify_name(name, variable_names)+'}{dt}'

        # Analyse
        var_code = _analyse_equation(var['eq'], eq, local_dict, tex_dict)

        # replace targets
        for target in targets:
            var_code = var_code.replace('PostSum'+target, "(\\sum_{\\text{" + target + "}} \\text{psp}(t))^{\\text{post}}")
            var_code = var_code.replace('PreSum'+target,  "(\\sum_{\\text{" + target + "}} \\text{psp}(t))^{\\text{pre}}")

        # Add the code
        code += """\\begin{dmath*}
%(eq)s
\\end{dmath*}
""" % {'eq': var_code}

    # Pre-event
    if synapse.type == 'spike' and synapse.description:
        for var in extract_pre_spike_variable(synapse.description):
            eq = var['eq']
            # pre/post variables
            eq, untouched_var, dependencies = extract_prepost(var['name'], eq, synapse.description)
            for dep in dependencies['post']:
                local_dict['_post_'+dep] = Symbol("{" + dep + "^{\\text{post}}}(t)")
            for dep in dependencies['pre']:
                local_dict['_pre_'+dep] = Symbol("{" + dep + "^{\\text{pre}}}(t)")

            var_code = _analyse_equation(var['eq'], eq, local_dict, tex_dict)
            pre_event += """\\begin{dmath*}
%(eq)s
\\end{dmath*}
""" % {'eq': var_code}

        for var in extract_post_spike_variable(synapse.description):
            eq = var['eq']
            # pre/post variables
            eq, untouched_var, dependencies = extract_prepost(var['name'], eq, synapse.description)
            for dep in dependencies['post']:
                local_dict['_post_'+dep] = Symbol("{" + dep + "^{\\text{post}}}(t)")
            for dep in dependencies['pre']:
                local_dict['_pre_'+dep] = Symbol("{" + dep + "^{\\text{pre}}}(t)")

            var_code = _analyse_equation(var['eq'], eq, local_dict, tex_dict)
            post_event += """\\begin{dmath*}
%(eq)s
\\end{dmath*}
""" % {'eq': var_code}


    return psp, code, pre_event, post_event

# Splits an equation into two parts, caring for the increments
def _analyse_equation(orig, eq, local_dict, tex_dict):

    left = eq.split('=')[0]
    if left[-1] in ['+', '-', '*', '/']:
        op = left[-1]
        try:
            left = _analyse_part(left[:-1], local_dict, tex_dict)
        except Exception as e:
            _print(e)
            _warning('can not transform the left side of ' + orig +' to LaTeX, you have to do it by hand...')
            left = left[:-1]
        operator = " = " + left +  " " + op + (" (" if op != '+' else '')
    else:
        try:
            left = _analyse_part(left, local_dict, tex_dict)
        except Exception as e:
            _print(e)
            _warning('can not transform the left side of ' + orig +' to LaTeX, you have to do it by hand...')
        operator = " = "
    try:
        right = _analyse_part(eq.split('=')[1], local_dict, tex_dict)
    except Exception as e:
        _print(e)
        _warning('can not transform the right side of ' + orig +' to LaTeX, you have to do it by hand...')
        right = eq.split('=')[1]

    return left + operator + right + (" )" if operator.endswith('(') else "")


# Analyses and transform to latex a single part of an equation
def _analyse_part(expr, local_dict, tex_dict):

    def regular_expr(expr):
        analysed = parse_expr(expr,
            local_dict = local_dict,
            transformations = (standard_transformations + (convert_xor,))
            )
        return latex(analysed, symbol_names = tex_dict, mul_symbol="dot")

    def _condition(condition):
        condition = condition.replace('and', ' & ')
        condition = condition.replace('or', ' | ')
        return regular_expr(condition)

    def _extract_conditional(condition):
        if_statement = condition[0]
        then_statement = condition[1]
        else_statement = condition[2]
        # IF condition
        if_code = _condition(if_statement)
        # THEN
        if isinstance(then_statement, list): # nested conditional
            then_code =  _extract_conditional(then_statement)
        else:
            then_code = regular_expr(then_statement)
        # ELSE
        if isinstance(else_statement, list): # nested conditional
            else_code =  _extract_conditional(else_statement)
        else:
            else_code = regular_expr(else_statement)
        return "\\begin{cases}" + then_code + "\qquad \\text{if} \quad " + if_code + "\\\\ "+ else_code +" \qquad \\text{otherwise.} \end{cases}"

    # Extract if/then/else
    if 'else:' in expr:
        ite_code = extract_ite(expr)
        return _extract_conditional(ite_code)

    # return the transformed equation
    return regular_expr(expr)

def extract_ite(eq):
    def transform(code):
        " Transforms the code into a list of lines."
        res = []
        items = []
        for arg in code.split(':'):
            items.append( arg.strip())
        for i in range(len(items)):
            if items[i].startswith('if '):
                res.append( items[i].strip() )
            elif items[i].endswith('else'):
                res.append(items[i].split('else')[0].strip() )
                res.append('else' )
            else: # the last then
                res.append( items[i].strip() )
        return res


    def parse(lines):
        " Recursive analysis of if-else statements"
        result = []
        while lines:
            if lines[0].startswith('if'):
                block = [lines.pop(0).split('if')[1], parse(lines)]
                if lines[0].startswith('else'):
                    lines.pop(0)
                    block.append(parse(lines))
                result.append(block)
            elif not lines[0].startswith(('else')):
                result.append(lines.pop(0))
            else:
                break
        return result[0]

    # Process the equation
    multilined = transform(eq)
    condition = parse(multilined)
    return condition

# Latexify names
greek = ['alpha', 'beta', 'gamma', 'epsilon', 'eta', 'kappa', 'delta', 'lambda', 'mu', 'nu', 'zeta', 'sigma', 'phi', 'psi', 'rho', 'omega', 'xi', 'tau',
         'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Phi', 'Psi', 'Omega'
]

def _latexify_name(name, local):
    parts = name.split('_')
    if len(parts) == 1:
        if len(name) == 1:
            equiv = name
        elif name in greek:
            equiv = '\\' + name
        else:
            equiv = '{\\text{' + name + '}}'
        if name in local:
            equiv = '{' + equiv + '}(t)'
        return equiv
    elif len(parts) == 2:
        equiv = ""
        for p in parts:
            if len(p) == 1:
                equiv += '' + p + '_'
            elif p in greek:
                equiv += '\\' + p + '_'
            else:
                equiv += '{\\text{' + p + '}}' + '_'
        equiv = equiv[:-1]
        if name in local:
            equiv = '{' + equiv + '}(t)'
        return equiv
    else:
        equiv = '{\\text{' + name + '}}'
        equiv = equiv.replace('_', '\_')
        if name in local:
            equiv = equiv + '(t)'
        return equiv

def pop_name(name):
    return name.replace('_', '\_')
