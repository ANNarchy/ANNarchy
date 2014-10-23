from ANNarchy.core.Global import _populations, _projections

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
    E             & Iaf neuron        & $N_{\\text{E}} = 4N_{\\text{I}}$  \\\\ \\hline
    I             & Iaf neuron        & $N_{\\text{I}}$ \\\\ \\hline
    E$_{\\text{ext}}$ & Poisson generator & $C_E(N_{\\text{E}}+N_{\\text{I}})$ \\\\ \\hline
\end{tabularx}

\\vspace{2ex}
"""

connectivity_template = """
\\noindent
\\begin{tabularx}{\linewidth}{|l|l|l|X|}\hline
\hdr{4}{C}{Connectivity}\\\\ \\hline
\\textbf{Name} & \\textbf{Source} & \\textbf{Target} & \\textbf{Pattern} \\\\ \\hline
    EE & E & E & 
    Random convergent $C_{\\text{E}}\\rightarrow 1$, weight $J$, delay $D$ \\\\ \\hline
    IE & E & I & 
    Random convergent $C_{\\text{E}}\\rightarrow 1$, weight $J$, delay $D$ \\\\ \\hline
    EI & I & E & 
    Random convergent $C_{\\text{I}}\\rightarrow 1$, weight $-gJ$, delay $D$ \\\\ \\hline
    II & I & I & 
    Random convergent $C_{\\text{I}}\\rightarrow 1$, weight $-gJ$, delay $D$ \\\\ \\hline
    Ext& E$_{\\text{ext}}$ & E $\cup$ I & 
    Non-overlapping $C_{\\text{E}}\\rightarrow 1$, weight $J$, delay $D$ \\\\ \\hline
\end{tabularx}

\\vspace{2ex}
"""

neuron_models_template = """
\\noindent
\\begin{tabularx}{\linewidth}{|p{0.15\linewidth}|X|}\hline
\hdr{2}{D}{Neuron and Synapse Model}\\\\ \\hline
\\textbf{Name} & Iaf neuron \\\\ \\hline
\\textbf{Type} & Leaky integrate-and-fire, $\delta$-current input\\\\ \\hline
\\raisebox{-4.5ex}{\parbox{\linewidth}{\\textbf{Subthreshold dynamics}}} &
\\\\ \\hline
\multirow{3}{*}{\\textbf{Spiking}} & 
\\\\ \\hline
\end{tabularx}

\\vspace{2ex}
"""

footer = """
\\noindent\\begin{tabularx}{\linewidth}{|l|X|}\hline
\hdr{2}{E}{Input}\\\\ \\hline
\\textbf{Type} & \\textbf{Description} \\\\ \\hline
---
\\\\ \\hline
\end{tabularx}

\\vspace{2ex}

\\noindent\\begin{tabularx}{\linewidth}{|X|}\hline
\hdr{1}{F}{Measurements}\\\\ \\hline
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

        list_neuron_models = []
        for pop in _populations:
            # population name
            population_names += pop.name + ", "
            # neuron models
            list_neuron_models.append(pop.neuron_type.name)
        for neur in list(set(list_neuron_models)):
            neuron_models += neur + ', '
        population_names = population_names[:-2] # suppress the last ,
        neuron_models = neuron_models[:-2] # suppress the last ,

        list_connectivity = []
        list_synapse_models = []
        for proj in _projections:
            list_connectivity.append(proj.connector_description)
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

    # Analyse only the neurons/synapses which are different

    # Generate the summary
    summary = generate_summary()

    with open(filename, 'w') as wfile:
        wfile.write(header)
        wfile.write(preamble)
        wfile.write(summary)
        wfile.write(footer)