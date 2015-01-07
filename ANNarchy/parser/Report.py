from ANNarchy.core.Global import _populations, _projections, _neurons, _synapses, _warning, _error, _print
from ANNarchy.core.Random import RandomDistribution
from .Extraction import *
from .SingleAnalysis import pattern_omp

from sympy import *
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor, auto_number
import re

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
\documentclass{article}
\usepackage[margin=1in]{geometry} 
\usepackage{tabularx}  
\usepackage{multirow}  
\usepackage{colortbl} 

\usepackage[fleqn]{amsmath} 
\setlength{\mathindent}{0em}
\usepackage{mathpazo}
\usepackage[scaled=.95]{helvet}
\\renewcommand\\familydefault{\sfdefault}

\\renewcommand\\arraystretch{1.2}  
\pagestyle{empty}

\\newcommand{\hdr}[3]{
    \multicolumn{#1}{|l|}{
        \color{white}\cellcolor[gray]{0.0}
        \\textbf{\makebox[0pt]{#2}\hspace{0.5\linewidth}\makebox[0pt][c]{#3}}
    }
}

\\begin{document}
"""

summary_template="""
\\noindent
\\begin{tabularx}{\linewidth}{|l|X|}\hline
\hdr{2}{A}{Model Summary}\\\\ \\hline
\\textbf{Populations}     & %(population_names)s \\\\ \\hline
\\textbf{Topology}        & --- \\\\ \\hline
\\textbf{Connectivity}    & %(connectivity)s \\\\ \\hline
\\textbf{Neuron models}   & %(neuron_models)s \\\\ \\hline
\\textbf{Channel models}  & --- \\\\ \\hline
\\textbf{Synapse models}  & --- \\\\ \\hline
\\textbf{Plasticity}      & %(synapse_models)s\\\\ \\hline
\\textbf{Input}           & --- \\\\ \\hline
\\textbf{Measurements}    & --- \\\\ \\hline
\end{tabularx}

\\vspace{2ex}
"""

populations_template = """
\\noindent
\\begin{tabularx}{\linewidth}{|l|l|X|}\hline
\hdr{3}{B}{Populations}\\\\ \\hline
    \\textbf{Name}   & \\textbf{Elements} & \\textbf{Size} \\\\ \\hline
%(populations_description)s
\end{tabularx}

\\vspace{2ex}
"""

connectivity_template = """
\\noindent
\\begin{tabularx}{\linewidth}{|l|l|l|X|}\hline
\hdr{4}{C}{Connectivity}\\\\ \\hline
\\textbf{Source} & \\textbf{Destination} & \\textbf{Target} & \\textbf{Pattern} \\\\ \\hline
%(projections_description)s
\end{tabularx}

\\vspace{2ex}
"""

neuron_models_template = """
\\noindent
\\begin{tabularx}{\linewidth}{|p{0.15\linewidth}|X|}\hline
\hdr{2}{D}{Neuron Models}\\\\ \\hline
%(first_neuron)s
\end{tabularx}
\\vspace{2ex}

%(neurons)s

"""

synapse_models_template = """
\\noindent
\\begin{tabularx}{\linewidth}{|p{0.15\linewidth}|X|}\hline
\hdr{2}{E}{Synapse Models}\\\\ \\hline
%(first_synapse)s
\end{tabularx}
\\vspace{2ex}

%(synapses)s

"""

footer = """
\\noindent\\begin{tabularx}{\linewidth}{|l|X|}\hline
\hdr{2}{X}{Input}\\\\ \\hline
\\textbf{Type} & \\textbf{Description} \\\\ \\hline
---
\\\\ \\hline
\end{tabularx}

\\vspace{2ex}

\\noindent\\begin{tabularx}{\linewidth}{|X|}\hline
\hdr{1}{Y}{Measurements}\\\\ \\hline
---
\\\\ \\hline
\end{tabularx}

\end{document}
"""


def report(filename="./report.tex"):
    """ Generates a .tex file describing the network according to: 
    Nordlie E, Gewaltig M-O, Plesser HE (2009). Towards Reproducible Descriptions of Neuronal Network Models. PLoS Comput Biol 5(8): e1000456.

    **Parameters:**

    * *filename*: name of the .tex file where the report will be written (default: "./report.tex")
    """
    def generate_summary():
        "part A"

        population_names = str(len(_populations)) + ': ' 
        connectivity = ""
        neuron_models = ""
        synapse_models = ""

        for pop in _populations:
            # population name
            population_names += pop.name + ", "
        for neur in _neurons:
            neuron_models += neur.name + ', '
        population_names = population_names[:-2] # suppress the last ,
        neuron_models = neuron_models[:-2] # suppress the last ,

        list_connectivity = []
        list_synapse_models = []
        for proj in _projections:
            list_connectivity.append(proj.connector_name)
            if not proj.synapse.name in ['Spiking synapse', 'Rate-coded synapse']:
                list_synapse_models.append(proj.synapse.name)
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

    def generate_populations():
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
    %(pop_name)s             & %(neuron_type)s        & $N_\\text{%(pop_name)s}$ = %(size)s  \\\\ \\hline
"""
        for pop in _populations:
            txt += pop_tpl % {'pop_name': pop.name, 'neuron_type': pop.neuron_type.name, 'size': format_size(pop)}

        return populations_template % {'populations_description': txt}

    def generate_projections():
        txt = ""
        proj_tpl = """
    %(pre)s & %(post)s & %(target)s &
    %(description)s \\\\ \\hline
"""        
        for proj in _projections:
            txt += proj_tpl % {'pre': proj.pre.name, 'post': proj.post.name, 'target': proj.target,
                                'description': proj.connector_description}

        return connectivity_template % {'projections_description': txt}

    def generate_neuron_models():
        firstneuron = ""
        neurons = ""

        firstneuron_tpl = """
\\textbf{Name} & %(name)s \\\\ \\hline
\\textbf{Type} & %(description)s\\\\ \\hline
\\textbf{%(equation_type)s} &
%(variables)s 
\\\\ \\hline
"""

        neuron_tpl = """
\\noindent
\\begin{tabularx}{\linewidth}{|p{0.15\linewidth}|X|}\hline
\\textbf{Name} & %(name)s \\\\ \\hline
\\textbf{Type} & %(description)s\\\\ \\hline
\\textbf{%(equation_type)s} &
%(variables)s 
\\\\ \\hline
\end{tabularx}
\\vspace{2ex}
"""
        for idx, neuron in enumerate(_neurons):
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
                'variables': eqs,
                'equation_type': "Subthreshold dynamics" if neuron.type == 'spike' else 'Equations'
            }

            # Generate the code depending on the neuron position
            if idx == 0:
                firstneuron = firstneuron_tpl % desc
            else:
                neurons += neuron_tpl % desc

        return neuron_models_template % {'first_neuron': firstneuron,'neurons': neurons}

    def generate_synapse_models():
        firstsynapse = ""
        synapses = ""

        firstsynapse_tpl = """
\\textbf{Name} & %(name)s \\\\ \\hline
\\textbf{Type} & %(description)s\\\\ \\hline
\\textbf{PSP} & %(psp)s\\\\ \\hline
\\textbf{%(equation_type)s} &
%(variables)s 
\\\\ \\hline
"""

        synapse_tpl = """
\\noindent
\\begin{tabularx}{\linewidth}{|p{0.15\linewidth}|X|}\hline
\\textbf{Name} & %(name)s \\\\ \\hline
\\textbf{Type} & %(description)s\\\\ \\hline
\\textbf{PSP} & %(psp)s\\\\ \\hline
\\textbf{%(equation_type)s} &
%(variables)s 
\\\\ \\hline
\end{tabularx}
\\vspace{2ex}
"""
        for idx, synapse in enumerate(_synapses):
            # Generate the code for the equations
            psp, eqs, spike_txt = _process_synapse_equations(synapse)

            # Spiking neurons have extra fields for the event-driven
            if synapse.type == 'spike':
                spike_extra = """
\\\\ \\hline
\\textbf{Spiking} & 
%(spike)s
"""
                #eqs += spike_extra % {'spike': spike_txt}


            # Build the dictionary
            desc = {
                'name': synapse.name,
                'description': synapse.short_description,
                'variables': eqs,
                'psp': psp,
                'equation_type': 'Equations'
            }

            # Generate the code depending on the neuron position
            if idx == 0:
                firstsynapse = firstsynapse_tpl % desc
            else:
                synapses += synapse_tpl % desc

        return synapse_models_template % {'first_synapse': firstsynapse,'synapses': synapses}

    # stdout
    _print('Generating report in', filename)

    # Generate the summary
    summary = generate_summary()
    # Generate the populations
    populations = generate_populations()
    # Generate the populations
    projections = generate_projections()
    # Generate the neuron models
    neuron_models = generate_neuron_models()
    # Generate the synapse models
    synapse_models = generate_synapse_models()

    with open(filename, 'w') as wfile:
        wfile.write(header)
        wfile.write(preamble)
        wfile.write(summary)
        wfile.write(populations)
        wfile.write(projections)
        wfile.write(neuron_models)
        wfile.write(synapse_models)
        wfile.write(footer)

def _process_random(val):
    "Transforms a connector attribute (weights, delays) into a string representation"
    if isinstance(val, RandomDistribution):
        return val.latex()
    else:
        return str(val)

def _process_neuron_equations(neuron):
    code = ""

    # Extract parameters and variables
    parameters = extract_parameters(neuron.parameters)
    variables = extract_variables(neuron.equations)
    variable_names = [var['name'] for var in variables]
    attributes, local_var, global_var = get_attributes(parameters, variables)

    # Create a dictionary for parsing
    local_dict = {
        'g_target': Symbol('g_\\text{target}'),
        't_pre': Symbol('t_\\text{pre}'),
        't_post': Symbol('t_\\text{pos}'),
        'Uniform': Function('\mathcal{U}'),
        'Normal': Function('\mathcal{N}'),
    }

    for att in attributes:
        local_dict[att] = Symbol(_latexify_name(att, variable_names))

    tex_dict = {}
    for key, val in local_dict.iteritems():
        tex_dict[val] = str(val)

    for var in variables:
        # Retrieve the equation
        eq = var['eq']
        # Parse the equation
        eq = eq.replace(' ', '') # supress spaces
        ode = re.findall(r'([^\w]*)d([\w]+)/dt', eq)
        if len(ode) > 0:
            name = ode[0][1]
            eq = eq.replace('d'+name+'/dt', '_grad_'+name)
            grad_symbol = Symbol('\\frac{d'+_latexify_name(name, variable_names)+'}{dt}')
            local_dict['_grad_'+name] = grad_symbol
            tex_dict[grad_symbol] = '\\frac{d'+_latexify_name(name, variable_names)+'}{dt}'

        left, right = eq.split('=')
        try:
            latex_left = _analyse_part(left, local_dict, tex_dict)
        except:
            _warning('can not transform the left side of ' + var['eq']+' to LaTeX, you have to it by hand...')
            latex_left = var['eq'].split('=')[0]
        try:
            latex_right = _analyse_part(right, local_dict, tex_dict)
        except:
            _warning('can not transform the right side of ' + var['eq']+' to LaTeX, you have to it by hand...')
            latex_right = var['eq'].split('=')[1]

        var_code = latex_left + ' = ' + latex_right

        # Add the code
        code += """\\[
%(eq)s
\\]
""" % {'eq': var_code}

    if not neuron.spike: # rate-code, no spike
        return code, ""

    # Additional code for spiking neurons
    spike_code = "If $" + _analyse_part(neuron.spike, local_dict, tex_dict) + "$:"

    # Reset
    spike_code += """
    \\begin{enumerate}
        \item Emit a spike"""
    
    reset_vars = extract_spike_variable(neuron.description, pattern_omp)['spike_reset']
    for var in reset_vars:
        eq = var['eq']

        left = eq.split('=')[0]
        if left[-1] in ['+', '-', '*', '/']:
            op = left[-1]
            left = _analyse_part(left[:-1], local_dict, tex_dict)
            operator = " = " + left +  " " + op + " "
        else:
            left = _analyse_part(left, local_dict, tex_dict)
            operator = " = "
        right = _analyse_part(eq.split('=')[1], local_dict, tex_dict)
        spike_code += """
        \item $""" + left + " " + operator + " " + right + "$"

        if 'unless_refractory' in var['constraint']:
            spike_code += " (not during the refractory period)."

    spike_code += """
    \end{enumerate}"""

    return code, spike_code


def _process_synapse_equations(synapse):
    psp = ""
    code = ""
    spiking = ""

    # Extract parameters and variables
    parameters = extract_parameters(synapse.parameters)
    variables = extract_variables(synapse.equations)
    variable_names = [var['name'] for var in variables]
    attributes, local_var, global_var = get_attributes(parameters, variables)

    # Create a dictionary for parsing
    local_dict = {
        'g_target': Symbol('g_\\text{target}'),
        't_pre': Symbol('t_\\text{pre}'),
        't_post': Symbol('t_\\text{pos}'),
        'Uniform': Function('\mathcal{U}'),
        'Normal': Function('\mathcal{N}'),
    }

    for att in attributes:
        local_dict[att] = Symbol(_latexify_name(att, variable_names))

    tex_dict = {}
    for key, val in local_dict.iteritems():
        tex_dict[val] = str(val)


    # PSP
    if synapse.psp:
        psp = synapse.psp.strip()
    else:
        if synapse.type == 'rate':
            psp = "$w(t) \cdot \\text{pre}.r(t)$"
        else:
            psp = "$g_\\text{target}(t) = g_\\text{target}(t) + w(t)$"


    # Variables
    for var in variables:
        # Retrieve the equation
        eq = var['eq']
        # pre/post variables
        eq, untouched_var, dependencies = extract_prepost(var['name'], eq, synapse.description, pattern_omp)
        print eq, untouched_var, dependencies
        # Parse the equation
        eq = eq.replace(' ', '') # supress spaces
        ode = re.findall(r'([^\w]*)d([\w]+)/dt', eq)
        if len(ode) > 0:
            name = ode[0][1]
            eq = eq.replace('d'+name+'/dt', '_grad_'+name)
            grad_symbol = Symbol('\\frac{d'+_latexify_name(name, variable_names)+'}{dt}')
            local_dict['_grad_'+name] = grad_symbol
            tex_dict[grad_symbol] = '\\frac{d'+_latexify_name(name, variable_names)+'}{dt}'

        left, right = eq.split('=')
        try:
            latex_left = _analyse_part(left, local_dict, tex_dict)
        except:
            _warning('can not transform the left side of ' + var['eq']+' to LaTeX, you have to it by hand...')
            latex_left = var['eq'].split('=')[0]
        try:
            latex_right = _analyse_part(right, local_dict, tex_dict)
        except:
            _warning('can not transform the right side of ' + var['eq']+' to LaTeX, you have to it by hand...')
            latex_right = var['eq'].split('=')[1]

        var_code = latex_left + ' = ' + latex_right

        # Add the code
        code += """\\[
%(eq)s
\\]
""" % {'eq': var_code}

    return psp, code, spiking



# Analyses and transform to latex a single part of an equation
def _analyse_part(expr, local_dict, tex_dict):
    def regular_expr(expr):
        analysed = parse_expr(expr,
            local_dict = local_dict,
            transformations = (standard_transformations + (convert_xor,)) 
            )
        return latex(analysed, symbol_names = tex_dict, mul_symbol="dot")

    # Extract if/then/else
    if 'else:' in expr:
        condition = re.findall(r'if(.*?):', expr)[0]
        then = re.findall(':(.*?)else:', expr)[0]
        else_st = expr.split('else:')[1]
        return "\\begin{cases}" + regular_expr(then) + "\qquad \\text{if} \quad " + regular_expr(condition) + "\\\\ "+ regular_expr(else_st) +" \qquad \\text{otherwise.} \end{cases}"
    
    # return the transformed equation
    return regular_expr(expr)

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
            equiv = '\\text{' + name + '}'
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
                equiv += '\\text{' + p + '}' + '_'
        equiv = equiv[:-1]
        if name in local:
            equiv = '{' + equiv + '}(t)'
        return equiv
    else:
        equiv = '\\text{' + name + '}'
        equiv = equiv.replace('_', '\_')
        if name in local:
            equiv = equiv + '(t)'
        return equiv
