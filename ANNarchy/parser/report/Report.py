from ANNarchy.intern import Messages

##################################
### Main method
##################################

def report(network:"Network", filename:str="./report.tex", standalone:bool=True, gather_subprojections:bool=False, title:str=None, author:str=None, date:str=None):
    """ 
    Generates a report describing the network.

    If the filename ends with ``.tex``, the TeX file is generated according to:

    > Nordlie E, Gewaltig M-O, Plesser HE (2009). Towards Reproducible Descriptions of Neuronal Network Models. PLoS Comput Biol 5(8): e1000456.

    If the filename ends with ``.md``, a (more complete) Markdown file is generated, which can be converted to pdf or html by ``pandoc``::

        pandoc report.md -o report.pdf
        pandoc report.md -o report.html

    :param network: Network instance.
    :param filename: name of the file where the report will be written.
    :param standalone: tells if the generated TeX file should be directly compilable or only includable (default: True). Ignored for Markdown.
    :param gather_subprojections: if a projection between two populations has been implemented as a multiple of projections between sub-populations, this flag allows to group them in the summary.
    :param title: title of the document (Markdown only).
    :param author: author of the document (Markdown only).
    :param date: date of the document (Markdown only).
    """

    if filename.endswith('.tex'):
        from .LatexReport import report_latex
        report_latex(filename, standalone, gather_subprojections, network.id)

    elif filename.endswith('.md'):
        from .MarkdownReport import report_markdown
        report_markdown(filename, standalone, gather_subprojections, title, author, date, network.id)

    else:
        Messages._error('report(): the filename must end with .tex or .md.')