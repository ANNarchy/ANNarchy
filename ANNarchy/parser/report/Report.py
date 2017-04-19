from ANNarchy.core.Global import _network, _objects, _warning, _error, _print

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

    if filename.endswith('.tex'):
        from .LatexReport import report_latex
        report_latex(filename, standalone, gather_subprojections, net_id)

    elif filename.endswith('.md'):
        from .MarkdownReport import report_markdown
        report_markdown(filename, standalone, gather_subprojections, net_id)

    else:
        _error('report(): the filename must end with .text or .md.')