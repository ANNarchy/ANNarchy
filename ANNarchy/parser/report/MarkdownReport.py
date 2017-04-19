import ANNarchy.core.Global as Global
from ANNarchy.core.Synapse import Synapse
from .LatexReport import _latexify_name, _target_list

##################################
### Main method
##################################
header = """% Description of the network
% ANNarchy (Artificial Neural Networks architect)
%
"""



def report_markdown(filename="./report.tex", standalone=True, gather_subprojections=False, net_id=0):
    """ Generates a .md file describing the network.

    **Parameters:**

    * *filename*: name of the .tex file where the report will be written (default: "./report.tex")
    * *standalone*: tells if the generated file should be directly compilable or only includable (default: True)
    * *gather_subprojections*: if a projection between two populations has been implemented as a multiple of projections between sub-populations, this flag allows to group them in the summary (default: False).
    * *net_id*: id of the network to be used for reporting (default: 0, everything that was declared)
    """

    # stdout
    Global._print('Generating report in', filename)

    # Structure
    structure = _generate_summary(net_id)

    with open(filename, 'w') as wfile:
        wfile.write(header)
        wfile.write(structure)


def _generate_summary(net_id):

    txt = """
# Structure of the network

## Populations

"""

    # Populations
    headers = ["Population", "Size", "Neuron type"]
    populations = []
    for pop in Global._network[net_id]['populations']:
        populations.append([pop.name, pop.geometry if len(pop.geometry)>1 else pop.size, pop.neuron_type.name])

    txt += _make_table(headers, populations)


    # Projections
    headers = ["Source", "Destination", "Target", "Synapse", "Pattern"]
    projections = []
    for proj in Global._network[net_id]['projections']:
        projections.append([
            proj.pre.name, 
            proj.post.name, 
            _target_list(proj.target),
            proj.synapse_type.name if not proj.synapse_type.name in Synapse._default_names.values() else "-",
            proj.connector_description
            ])

    txt += """
## Projections

"""
    txt += _make_table(headers, projections)

    return txt


def _make_table(header, data):
    "Creates a markdown table from the data, with headers defined in headers."

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
