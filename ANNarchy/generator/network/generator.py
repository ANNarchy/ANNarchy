#
#    Create a complete list of pyx modules
#
py_extension = """include "Network.pyx"

%(pop_inc)s  

%(proj_inc)s  

%(profile)s
"""

core = [

]
rate_code = [
'RateDendrite.cpp','RateDendrite.h',
'RateProjection.cpp','RateProjection.h',
]

spike_code = """
"""