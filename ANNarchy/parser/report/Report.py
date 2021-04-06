from ANNarchy.core.Global import _network, _objects, _warning, _error, _print

##################################
### Main method
##################################

def report(filename="./report.tex", standalone=True, gather_subprojections=False, title=None, author=None, date=None, net_id=0):
    """ 
    Generates a report describing the network.

    If the filename ends with ``.tex``, the TeX file is generated according to:

    Nordlie E, Gewaltig M-O, Plesser HE (2009). Towards Reproducible Descriptions of Neuronal Network Models. PLoS Comput Biol 5(8): e1000456.

    If the filename ends with ``.md``, a (more complete) Markdown file is generated, which can be converted to pdf or html by ``pandoc``::

        pandoc report.md  -sSN -V geometry:margin=1in -o report.pdf
        pandoc report.md  -sSN -o report.html

    :param filename: name of the file where the report will be written (default: "./report.tex")
    :param standalone: tells if the generated TeX file should be directly compilable or only includable (default: True). Ignored for Markdown.
    :param gather_subprojections: if a projection between two populations has been implemented as a multiple of projections between sub-populations, this flag allows to group them in the summary (default: False).
    :param title: title of the document (Markdown only)
    :param author: author of the document (Markdown only)
    :param date: date of the document (Markdown only)
    :param net_id: id of the network to be used for reporting (default: 0, everything that was declared)
    """

    if filename.endswith('.tex'):
        from .LatexReport import report_latex
        report_latex(filename, standalone, gather_subprojections, net_id)

    elif filename.endswith('.md'):
        from .MarkdownReport import report_markdown
        report_markdown(filename, standalone, gather_subprojections, title, author, date, net_id)

    else:
        _error('report(): the filename must end with .tex or .md.')