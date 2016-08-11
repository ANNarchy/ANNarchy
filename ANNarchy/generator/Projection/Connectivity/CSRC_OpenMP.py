######################################
### CSRC Connectivity matrix
######################################
connectivity_matrix = {
    'declare': """
""",
    'accessor': """
""",
    'init': """
""",
    'pyx_struct': """
""",
    'pyx_wrapper_args': "synapses",
    'pyx_wrapper_init': """
        cdef CSRC syn = synapses
""",
    'pyx_wrapper_accessor': """
"""
}

weight_matrix = {
    'declare': """
""",
    'accessor': """
""",
    'init': """
""",
    'pyx_struct': """
""",
    'pyx_wrapper_args': "",
    'pyx_wrapper_init': """
""",
    'pyx_wrapper_accessor': """
"""
}

