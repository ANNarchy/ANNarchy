from ANNarchy.core.Global import _populations, _projections

template = """
% LaTeX file for generating the Model Description Table in Fig. 5 of
%
%  Nordlie E, Gewaltig M-O, Plesser HE (2009) 
%  Towards Reproducible Descriptions of Neuronal Network Models.
%  PLoS Comput Biol 5(8): e1000456. 
%
%  Paper URL : http://dx.doi.org/10.1371/journal.pcbi.1000456
%  Figure URL: http://dx.doi.org/10.1371/journal.pcbi.1000456.g005
%
% This file is released under a
%
%   Creative Commons Attribution, non-commercial, share-alike licence
%   http://creativecommons.org/licenses/by-nc-sa/3.0/de/deed.en
%
% with the following specifications:
%
%  1. When publishing tables generated from this LaTeX file and modified
%     versions of it, you must cite the paper by Nordlie et al given above.
%
%  2. The non-commercial clause applies only to the distribution of THIS FILE
%     and LaTeX source code files derived from it. You may commercially publish
%     documents generated using this file and derivatived versions of this file.
%
% Contact: Hans Ekkehard Plesser, UMB (hans.ekkehard.plesser at umb.no)

\documentclass{article}

\usepackage[margin=1in]{geometry} % get enough space on page

\usepackage{tabularx}  % automatically adjusts column width in tables
\usepackage{multirow}  % allows entries spanning several rows  
\usepackage{colortbl}  % allows coloring tables

\usepackage[fleqn]{amsmath}   % displayed equations flush left
\setlength{\mathindent}{0em}

% use Helvetica for text, Pazo math fonts
\usepackage{mathpazo}
\usepackage[scaled=.95]{helvet}
\\renewcommand\\familydefault{\sfdefault}

\\renewcommand\\arraystretch{1.2}  % slightly more space in tables

\pagestyle{empty}  % no header of footer

% \hdr{ncols}{label}{title}
%
% Typeset header bar across table with ncols columns 
% with label at left margin and centered title
%
\\newcommand{\hdr}[3]{%
  \multicolumn{#1}{|l|}{%
    \color{white}\cellcolor[gray]{0.0}%
    \\textbf{\makebox[0pt]{#2}\hspace{0.5\linewidth}\makebox[0pt][c]{#3}}%
  }%
}


\\begin{document}

% - A ------------------------------------------------------------------------------

\\noindent
\\begin{tabularx}{\linewidth}{|l|X|}\hline
%
\hdr{2}{A}{Model Summary}\\\\ \\hline
\\textbf{Populations}     & Three: excitatory, inhibitory, external input \\\\ \\hline
\\textbf{Topology}        & --- \\\\ \\hline
\\textbf{Connectivity}    & Random convergent connections \\\\ \\hline
\\textbf{Neuron model}    & Leaky integrate-and-fire, fixed voltage
                           threshold, fixed absolute refractory time (voltage clamp) \\\\ \\hline
\\textbf{Channel models}  & --- \\\\ \\hline
\\textbf{Synapse model}   & $\delta$-current inputs (discontinuous
                           voltage jumps) \\\\ \\hline
\\textbf{Plasticity}      & ---\\\\ \\hline
\\textbf{Input}           & Independent fixed-rate Poisson spike trains to all
                           neurons \\\\ \\hline
\\textbf{Measurements}    & Spike activity \\\\ \\hline
\end{tabularx}

\\vspace{2ex}

% - B -----------------------------------------------------------------------------

\\noindent\\begin{tabularx}{\linewidth}{|l|l|X|}\hline
\hdr{3}{B}{Populations}\\\\ \\hline
  \\textbf{Name}   & \\textbf{Elements} & \\textbf{Size} \\\\ \\hline
    E             & Iaf neuron        & $N_{\\text{E}} = 4N_{\\text{I}}$  \\\\ \\hline
    I             & Iaf neuron        & $N_{\\text{I}}$ \\\\ \\hline
    E$_{\\text{ext}}$ & Poisson generator & $C_E(N_{\\text{E}}+N_{\\text{I}})$ \\\\ \\hline
\end{tabularx}

\\vspace{2ex}

% - C ------------------------------------------------------------------------------

\\noindent\\begin{tabularx}{\linewidth}{|l|l|l|X|}\hline
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

% - D ------------------------------------------------------------------------------

\\noindent\\begin{tabularx}{\linewidth}{|p{0.15\linewidth}|X|}\hline
\hdr{2}{D}{Neuron and Synapse Model}\\\\ \\hline
\\textbf{Name} & Iaf neuron \\\\ \\hline
\\textbf{Type} & Leaky integrate-and-fire, $\delta$-current input\\\\ \\hline
\\raisebox{-4.5ex}{\parbox{\linewidth}{\\textbf{Subthreshold dynamics}}} &
\\rule{1em}{0em}\\vspace*{-3.5ex}
    \\begin{equation*}
      \\begin{array}{r@{\;=\;}lll}
      \\tau \dot{V}(t) & -V(t) + R I(t) &\\text{if} & t > t^*+\\tau_{\\text{rp}} \\
      V(t) & V_{\\text{r}} & \\text{else} \\[2ex]
      I(t) & \multicolumn{3}{l}{\\frac{\\tau}{R} \sum_{\\tilde{t}} w
        \delta(t-(\\tilde{t}+\Delta))}
      \end{array}
    \end{equation*} 
\\vspace*{-2.5ex}\\rule{1em}{0em}
 \\\\ \\hline
\multirow{3}{*}{\\textbf{Spiking}} & 
   If $V(t-)<\\theta \wedge V(t+)\geq \theta$
\\vspace*{-1ex}
\\begin{enumerate}\setlength{\itemsep}{-0.5ex}
\item set $t^* = t$
\item emit spike with time-stamp $t^*$
\end{enumerate}
\\vspace*{-4ex}\\rule{1em}{0em}
 \\\\ \\hline
\end{tabularx}

\\vspace{2ex}

% - E ------------------------------------------------------------------------------

\\noindent\\begin{tabularx}{\linewidth}{|l|X|}\hline
\hdr{2}{E}{Input}\\\\ \\hline
\\textbf{Type} & \\textbf{Description} \\\\ \\hline
{Poisson generators} & Fixed rate $\\nu_{\\text{ext}}$, $C_{\\text{E}}$
generators per neuron, each generator projects to one neuron\\\\ \\hline
\end{tabularx}

\\vspace{2ex}

% - F -----------------------------------------------------------------------------

\\noindent\\begin{tabularx}{\linewidth}{|X|}\hline
\hdr{1}{F}{Measurements}\\\\ \\hline
Spike activity as raster plots, rates and ``global frequencies'', no details given
\\\\ \\hline
\end{tabularx}

% ---------------------------------------------------------------------------------

\end{document}
"""


def report(filename="./report.tex"):
    """ Generates a .tex file describing the network according to: 
Nordlie E, Gewaltig M-O, Plesser HE (2009). Towards Reproducible Descriptions of Neuronal Network Models. PLoS Comput Biol 5(8): e1000456.

    **Parameters:**

    * *filename*: name of the .tex file where the report will be written (default: "./report.tex")
    """

    # Print populations names and geometry

    # Write the file to disk
    txt = template      
    with open(filename, 'w') as wfile:
        wfile.write(txt)