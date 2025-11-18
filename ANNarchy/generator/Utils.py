"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.PopulationView import PopulationView

from ANNarchy.intern.ConfigManagement import ConfigManager, _check_paradigm
from ANNarchy.intern import Messages

import re
import subprocess
import sys

def sort_odes(desc, locality='local'):
    equations = []
    is_ode = False
    for param in desc['variables']:
        if param['cpp'] == '':
            continue
        if param['method'] == 'event-driven':
            continue
        if param['name'] in desc[locality]:
            if param['switch']: # ODE
                if is_ode: # was already ODE
                    if len(equations) == 0:
                        equations.append(('ode', [param]))
                    else:
                        equations[-1][1].append(param)
                else: # new block
                    is_ode = True
                    equations.append(('ode', [param]))
            else: # non-ODE
                if is_ode:
                    is_ode = False
                    equations.append(('non-ode', [param]))
                else:
                    if len(equations) == 0:
                        equations.append(('non-ode', [param]))
                    else:
                        equations[-1][1].append(param)

    return equations

def generate_bound_code(param, obj):
    code = ""
    for bound, val in param['bounds'].items():
        if bound in ['min', 'max']:
            code += """if(%(var)s%(index)s %(operator)s %(val)s)
    %(var)s%(index)s = %(val)s;
""" % {
        'index': '%(local_index)s' if param['locality'] == 'local' else ('%(semiglobal_index)s' if param['locality'] == 'semiglobal' else'%(global_index)s'),
        'var' : param['name'], 'val' : val,
        'operator': '<' if bound=='min' else '>'
    }
    return code

def append_mask_multiplier(switch_code, var_name):
    """
    To apply SIMD vectorization on neural and continous synaptic updates the
    inner loop code should not contain if-else statements.

    Therefore, we add to switch statements of the ODE a multiplicant which
    represents the boolean condition.
    """
    return switch_code.replace(";", "* "+var_name+"%(local_index)s;")

def generate_non_ODE_block(variables, locality, obj, conductance_only, wrap_w, mask_variable, split_loop=False):
    " TODO: documentation "
    block_code = ""
    block_bounds = ""
    for param in variables:
        if conductance_only: # skip the variables which do not start with g_
            if not param['name'].startswith('g_'):
                continue

        # Add refractoriness
        if not param['name'].startswith('g_'):
            cpp_code = param['cpp'] if mask_variable is None else "if ("+mask_variable+"%(local_index)s) { "+param['cpp']+" }"
        else:
            cpp_code = param['cpp']

        bounds = generate_bound_code(param, obj)
        if wrap_w and param['name'] == "w":
            block_code += """
%(comment)s
if(%(wrap)s){
%(cpp)s
%(bounds)s
}
""" % { 'comment': '// ' + param['eq'],
        'cpp': cpp_code,
        'wrap': wrap_w,
        'bounds': bounds if not split_loop else ""}
            block_bounds += bounds if split_loop else ""

        else:
            block_code += """
%(comment)s
%(cpp)s
%(bounds)s
""" % { 'comment': '// ' + param['eq'],
        'cpp': cpp_code,
        'bounds': bounds if not split_loop else "" }
            block_bounds += bounds if split_loop else ""

    if not split_loop:
        return block_code
    else:
        return block_code, block_bounds


def generate_ODE_block(odes, locality, obj, conductance_only, wrap_w, mask_variable):
    code = ""

    # Count how many steps (midpoint has more than one step)
    nb_step = 0
    for param in odes:
        if isinstance(param['cpp'], list):
            nb_step = max(len(param['cpp']), nb_step)
        else:
            nb_step = max(1, nb_step)

    if len(odes) == 0:
        return ""

    # Iterate over all steps
    for step in range(nb_step):
        for param in odes:
            if conductance_only: # skip the variables which do not start with g_
                if not param['name'].startswith('g_'):
                    continue

            # Retrieve equation
            if isinstance(param['cpp'], list) and step < len(param['cpp']):
                eq = param['cpp'][step]
            elif isinstance(param['cpp'], str) and step == 0:
                eq = param['cpp']
            else:
                eq = ''
            # Generate code
            code += """
%(comment)s
%(cpp)s
""" % { 'comment': '// '+param['eq'],
        'cpp': eq }

    # Generate the switch code
    for param in odes:
        if conductance_only: # skip the variables which do not start with g_
            if not param['name'].startswith('g_'):
                continue

        bounds = generate_bound_code(param, obj)

        if not param['name'].startswith('g_'):
            switch = param['switch'] if mask_variable is None else append_mask_multiplier(param['switch'], mask_variable)
        else:
            switch = param['switch']

        if wrap_w and param['name'] == "w":
            code += """
%(comment)s
if(%(wrap)s){
%(switch)s
%(bounds)s
}
""" % { 'comment': '// '+param['eq'],
        'wrap': wrap_w,
        'bounds': bounds,
        'switch' : switch}
        else:
            code += """
%(comment)s
%(switch)s
%(bounds)s
""" % { 'comment': '// '+param['eq'],
        'bounds': bounds,
        'switch' : switch}


    return code

def generate_equation_code(desc, locality='local', obj='pop', conductance_only=False, wrap_w=None, mask_variable=None, padding=3):
    """
    Generate the C++ code for all equations part of *desc*.

    :param obj: either *obj* or *proj*
    :param locality: either *local*/*global* for populations or *local*/*semiglobal*/*global* for projections
    :param conductance_only: special case for populations considering only statements related to g_target variables
    :param wrap_w: if True, a condition check is added on the weight update
    :param with_refractory: special code generation for spiking models using refractoriness to ensure SIMD friendliness
    """
    # Separate ODEs from the pre- and post- equations
    odes = sort_odes(desc, locality)

    if odes == []: # No equations
        return ""

    # Generate code
    code = ""
    for type_block, block in odes:
        if type_block == 'ode':
            code += generate_ODE_block(block, locality, obj, conductance_only, wrap_w, mask_variable=mask_variable)
        elif type_block == 'non-ode':
            code += generate_non_ODE_block(block, locality, obj, conductance_only, wrap_w, mask_variable=mask_variable, split_loop=False)
        else:
            raise NotImplementedError

    # Add the padding to each line
    padded_code = tabify(code, padding)

    return padded_code

def determine_idx_type_for_projection(proj):
    """
    The suitable index type depends on the maximum number of neurons in
    pre-synaptic and post-synaptic layer.

    Notice (8th June 2021):

    It appears to a problem for the current Cython version to handle
    datatypes like "unsigned int". So I decided to replace the unsigned
    datatypes by an own definition. These definitions are placed in

    *ANNarchy/generator/Template/PyxTemplate.py*
    """
    # The user disabled this optimization.
    if ConfigManager().get('only_int_idx_type', proj.net_id):
        return "int", "int", "int", "int"

    # Currently only implemented for some cases,
    # the others default to "old" configuration
    if proj.synapse_type.type == "spike":
        return "int", "int", "int", "int"

    if _check_paradigm("cuda", proj.net_id):
        return "int", "int", "int", "int"

    if proj._storage_format != "lil" and ConfigManager().get('num_threads', proj.net_id)>1:
        return "int", "int", "int", "int"

    # max_size is related to the population sizes. As we use one type for
    # both dimension we need to determine the maximum
    pre_size = proj.pre.population.size if isinstance(proj.pre, PopulationView) else proj.pre.size
    post_size = proj.post.population.size if isinstance(proj.post, PopulationView) else proj.post.size
    max_size_one_dim = max(pre_size, post_size)
    max_size_both_dim = pre_size * post_size

    # For type decision we rely on the C++ boundaries which are decremented by 1
    # to allow usage of CSR-like formats without row overflow.
    if max_size_one_dim < 255:
        # 1 byte
        cpp_idx_type = "unsigned char"
        cython_idx_type= "_ann_uint8"

        if max_size_both_dim < 255:
            # can use the same type (should be seldom ...)
            cpp_size_type = "unsigned char"
            cython_size_type= "_ann_uint8"
        else:
            # next higher data type
            cpp_size_type = "unsigned short int"
            cython_size_type= "_ann_uint16"

    elif max_size_one_dim < 65534:
        # 2 byte
        cpp_idx_type = "unsigned short int"
        cython_idx_type= "_ann_uint16"

        if max_size_both_dim < 65534:
            cpp_size_type = "unsigned short int"
            cython_size_type= "_ann_uint16"
        else:
            cpp_size_type = "unsigned int"
            cython_size_type= "_ann_uint32"

    elif max_size_one_dim < 4294967294:
        # 4 byte
        cpp_idx_type = "unsigned int"
        cython_idx_type= "_ann_uint32"

        if max_size_both_dim < 4294967294:
            cpp_size_type = "unsigned int"
            cython_size_type= "_ann_uint32"
        else:
            cpp_size_type = "unsigned long int"
            cython_size_type= "_ann_uint64"

    else:
        # this is a hypothetical case I guess (HD: 4th June 2021)
        raise NotImplementedError("The matrix dimension exceeded the representable size ...")

    return cpp_idx_type, cython_idx_type, cpp_size_type, cython_size_type

def cpp_connector_available(connector_name, desired_format, storage_order, net_id):
    """
    Checks if a CPP implementation is available for the desired connection pattern
    (*connector_name*) and the target sparse matrix format (*desired_format*). Please
    note that not all formats are available for *pre_to_post* storage order.
    """
    # The user disabled this feature
    if not ConfigManager().get('use_cpp_connectors', net_id):
        return False

    cpp_patterns = {
        'st': {
            'post_to_pre': {
                "lil": ["Random", "Random Convergent", "All-to-All"],
                "csr": ["Random", "Random Convergent"],
                "coo": [],
                "hyb": [],
                "ell": [],
                "dense": ["Random"]
            },
            'pre_to_post': {
                "csr": ["Random", "Random Convergent"]
            }
        },
        'omp': {
            'post_to_pre': {
                "lil": ["Random"],
                "csr": [],
                "coo": [],
                "ell": []
            },
            'pre_to_post': {
                "csr": ["Random"]
            }
        },
        'cuda': {
            'post_to_pre': {
                "csr": ["Random", "Random Convergent"],
                "coo": [],
                "ellr": ["Random", "Random Convergent"],
                "dense": ["Random"]
            }
        }
    }

    if _check_paradigm("openmp", net_id):
        paradigm = "st" if ConfigManager().get('num_threads', net_id) == 1 else "omp"
    else:
        paradigm = "cuda"

    try:
        return connector_name in cpp_patterns[paradigm][storage_order][desired_format]

    except KeyError:
        # Fall back to Python construction
        return False

#####################################################################
#   Code formatting
#####################################################################
def indentLine(line, spaces=1):
    return (' ' * 4 * spaces) + line

def tabify(s, numSpaces):
    s = s.split('\n')
    s = map(lambda a, ns=numSpaces: indentLine(a, ns), s)
    s = '\n'.join(s)
    return s

def remove_trailing_spaces(code):
    """
    The generated code templates often contain empty lines, which are indented by tabify() or indentLine()
    afterwards which this introduces many white spaces which are annoying in some editors. The call of rstrip()
    on the complete string can not remove them. Therefore we implement this little helper function to call
    rstrip on each line.
    """
    stripped_lines = [line.rstrip() for line in code.split('\n')]

    stripped_code = ""
    for line in stripped_lines:
        stripped_code += line +'\n'

    return stripped_code

#####################################################################
#   Hardware-related
#####################################################################
def check_cuda_version(nvcc_executable):
    """
    Some features like atomic add for double values and power function are dependent on the CUDA version.
    """
    version_str = str(subprocess.check_output([nvcc_executable, "--version"]))
    try:
        version = float(version_str.split("\\")[-2].split(",")[1].split(" ")[2])
    except:
        try:
            version = float(version_str.split("\\")[-3].split(",")[1].split(" ")[2])
        except:
            Messages._error("Could not detect CUDA version: please check the CUDA installation or the configuration in annarchy.json")

    return version

def check_and_apply_pow_fix(eqs, cuda_version):
    """
    CUDA SDKs before 7.5 had an error if std=c++11 is enabled related
    to pow(double, int). Only pow(double, double) was detected as
    device function, the pow(double, int) will be detected as host
    function. (This was fixed within SDK 7.5)

    To support also earlier versions, we simply add a double type cast.
    """
    if eqs.strip() == "":
        # nothing to do
        return eqs

    if cuda_version > 7.0:
        # nothing to do, is working in higher SDKs
        return eqs

    # detect all pow statements
    pow_occur = re.findall(r"pow[\( [\S\s]*?\)*?, \d+\)]*?", eqs)
    for term in pow_occur:
        eqs = eqs.replace(term, term.replace(', ', ', (double)'))

    return eqs

def check_simd_instructions_type(simd_instr_set="avx", net_id:int=0):
    """
    Check the present CPUs if they offer a specific SIMD instruction set. We use 'lscpu' to detect
    the availabe instruction sets. In ANNarchy we support for now:

    - "avx":     256-bit register width
    - "avx512f": 512-bit register width

    Please note, that even though we detect the instruction set. Compiler flags must be set accordingly
    that it can work (i. e. either march=native detect the CPU correct or the -mavx/-mavx512f flags must be set)

    Parameters:

    * simd_instr_set: either "sse4_1", "avx" or "avx512f".

    Returns:

    True, if the specified flag was found, in any other cases it defaults to False.

    Remark (31th May 2021):

    This is a rather simple approach to detect the AVX capability of a CPU. If it fails, one can
    still hope for the auto-vectorization.
    """
    # Checks whether the user disabled the generation of SIMD code
    if ConfigManager().get('disable_SIMD_SpMV', net_id):
        return False

    # The hand-written codes are only validated
    # with g++ as target compiler
    if not sys.platform.startswith('linux'):
        return False

    try:
        # search for CPU flags
        lscpu_txt = (subprocess.check_output("lscpu | grep 'Flags' ", shell=True).strip()).decode()
        return simd_instr_set in lscpu_txt

    except:
        try:
            # lets try german
            lscpu_txt = (subprocess.check_output("lscpu | grep 'Markierungen' ", shell=True).strip()).decode()
            return simd_instr_set in lscpu_txt

        except:
            # give up and proceed without AVX
            return False

def get_highest_available_simd_instructions_type(net_id:int=0):
    """
    Check if SIMD operations are available. As higher order methods
    always contain the lower, we need to test in order SSE4, AVX, AVX512
    """
    simd_type = None

    if check_simd_instructions_type("sse4_1", net_id):
        simd_type = "sse"

    if check_simd_instructions_type("avx", net_id):
        simd_type = "avx"

    if check_simd_instructions_type("avx512f", net_id):
        simd_type = "avx512"

    return simd_type
