#
#    Create a complete list of pyx modules
#
py_extension = """include "Network.pyx"

%(pop_inc)s  

%(proj_inc)s  

%(profile)s
include "Connector.pyx"
"""