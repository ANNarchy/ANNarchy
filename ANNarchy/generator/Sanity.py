"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import re

from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.intern.SpecificProjection import SpecificProjection
from ANNarchy.intern.ConfigManagement import get_global_config, _check_paradigm
from ANNarchy.intern import Messages
from ANNarchy.models.Synapses import DefaultRateCodedSynapse

# No variable can have these names
reserved_variables = [
    't',
    'dt',
    't_pre',
    't_post',
    't_last',
    'last_spike',
    'rk_post',
    'rk_pre',
    'i',
    'j',
    'active',
    'refractory',
    'size',
]

def check_structure(populations, projections):
    """
    Checks the structure before compilation to display more useful error messages.
    """
    from ANNarchy.extensions.convolution.Transpose import Transpose

    # Check variable names
    _check_reserved_names(populations, projections)

    # Check that projections are created before compile
    for proj in projections:
        if isinstance(proj, Transpose):
            continue

        if not proj._connection_method:
            Messages._error('The projection between populations', proj.pre.id, 'and', proj.post.id, 'has not been connected.',
                            ' Call a connector method before compiling the network.')

    # Check if the storage formats are valid for the selected paradigm
    _check_storage_formats(projections)

    # Check that synapses access existing variables in the pre or post neurons
    _check_prepost(populations, projections)

    # Check locality of variable is respected
    _check_locality(populations, projections)

def check_experimental_features(populations, projections):
    """
    The idea behind this method, is to check if new experimental features are used. This
    should help also the user to be aware of changes.
    """
    # CPU-related formats
    if get_global_config('paradigm') == "openmp":
        if get_global_config("disable_SIMD_SpMV") == False:
            Messages._warning("Using hand-written SIMD kernel for continuous transmission is an experimental feature, we greatly appreciate bug reports.")

        for proj in projections:
            if proj._storage_format == "csr" and proj._storage_order == "pre_to_post":
                Messages._warning("Compressed sparse row (CSR) and pre_to_post ordering representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "bsr":
                Messages._warning("Blocked sparse row (BSR) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "coo":
                Messages._warning("Coordinate (COO) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "dia":
                Messages._warning("Diagonal (dia) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "ellr":
                Messages._warning("ELLPACK-R (ELLR) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "sell":
                Messages._warning("Sliced ELLPACK (SELL) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "ell":
                Messages._warning("ELLPACK (ELL) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "hyb":
                Messages._warning("Hybrid (ELL + COO) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "dense" and proj.synapse_type.type=="spike":
                Messages._warning("Dense representation is an experimental feature for spiking models, we greatly appreciate bug reports.")
                break

    # GPU-related formats
    elif get_global_config('paradigm') == "cuda":
        for pop in populations:
            if pop.neuron_type.description['type'] == "spike":
                Messages._warning('Spiking neurons on GPUs is an experimental feature. We greatly appreciate bug reports.')
                break

        for proj in projections:
            if proj._storage_format == "dense":
                Messages._warning("Dense representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "sell":
                Messages._warning("Sliced ELLPACK representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "ellr":
                Messages._warning("ELLPACK-R (ELLR) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "bsr":
                Messages._warning("Blocked sparse row (BSR) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "coo":
                Messages._warning("Coordinate (COO) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "hyb":
                Messages._warning("Hybrid (ELL + COO) representation is an experimental feature, we greatly appreciate bug reports.")
                break

    else:
        pass

def _check_reserved_names(populations, projections):
    """
    Checks no reserved variable names is redefined
    """
    # Check populations
    for pop in populations:
        # Reserved variable names
        for term in reserved_variables:
            if term in pop.attributes:
                Messages._print(pop.neuron_type.parameters)
                Messages._print(pop.neuron_type.equations)
                Messages._error(term + ' is a reserved variable name')

    # Check projections
    for proj in projections:
        # Reserved variable names
        for term in reserved_variables:
            if term in proj.attributes:
                Messages._print(proj.synapse_type.parameters)
                Messages._print(proj.synapse_type.equations)
                Messages._error(term + ' is a reserved variable name')

def _check_storage_formats(projections):
    """
    ANNarchy 4.7 introduced a set of sparse matrix formats. Some of them are not implemented for
    all paradigms or might not support specific optimizations.
    """
    for proj in projections:
        # Most of the sparse matrix formats are not trivially invertable and therefore we can not implement
        # spiking models with them
        if proj.synapse_type.type == "spike" and proj._storage_format in ["csr_vector", "csr_scalar", "ell", "ellr", "coo", "hyb", "dia"]:
            raise Messages.InvalidConfiguration("Using 'storage_format="+ proj._storage_format + "' is not allowed for spiking synapses.")

        # For some of the sparse matrix formats we don't implemented plasticity yet.
        if proj.synapse_type.type == "spike":
            if _check_paradigm("cuda"):
                if proj._storage_format == "csr" and proj._storage_order == "pre_to_post" and proj.synapse_type.description['plasticity']==True:
                    raise Messages.InvalidConfiguration("Plasticity rules for spiking models are not supported for CSRC and pre_to_post ordering.")

            # Continous transmission, e.g. gap junctions, should not be combined with pre-to-post
            if 'psp' in  proj.synapse_type.description.keys() and proj._storage_order=="pre_to_post":
                raise Messages.InvalidConfiguration("Using continuous transmission within a spiking synapse prevents the application of pre-to-post matrix ordering")

        # For some of the sparse matrix formats we don't implemented plasticity for rate-coded models yet.
        if proj.synapse_type.type == "rate":
            if _check_paradigm("openmp"):
                if proj._storage_format in ["coo", "hyb", "bsr", "sell", "dia"] and not isinstance(proj.synapse_type, DefaultRateCodedSynapse):
                    raise Messages.InvalidConfiguration("Using 'storage_format="+ proj._storage_format + "' is only allowed for default rate-coded synapses yet.")
            elif _check_paradigm("cuda"):
                if proj._storage_format in ["coo", "hyb", "bsr", "ell", "sell", "csr_vector", "csr_scalar"] and not isinstance(proj.synapse_type, DefaultRateCodedSynapse):
                    raise Messages.InvalidConfiguration("Using 'storage_format="+ proj._storage_format + "' is only allowed for default rate-coded synapses yet.")

        # Single weight optimization available?
        if proj._has_single_weight() and proj._storage_format in ["dense", "bsr"]:
            raise Messages.InvalidConfiguration("Using 'storage_format="+ proj._storage_format + "' is not allowed for single weight projections.")

        # In some cases we don't allow the usage of non-unifom delay
        if (proj.max_delay > 1 and proj.uniform_delay == -1):
            if _check_paradigm("cuda"):
                raise Messages.InvalidConfiguration("Using non-uniform delays is not available for CUDA devices.")

            else:
                if proj._storage_format == "ellr":
                    raise Messages.InvalidConfiguration("Using 'storage_format="+ proj._storage_format + "' is and non-uniform delays is not implemented.")

        if proj._storage_format == "dia":
            if _check_paradigm("cuda"):
                raise Messages.InvalidConfiguration('Using diagonal format is limited to CPUs yet.')

            if proj.pre.size < proj.post.size:
                raise Messages.InvalidConfiguration('Using diagonal format is not implemented for projections where the pre-synaptic layer is smaller than the post-synaptic one.')

            if isinstance(proj.post, PopulationView):
                raise Messages.InvalidConfiguration('Using diagonal format and post-synaptic PopulationViews is not available.')

        if not _check_paradigm("cuda") and (proj._storage_format in ["csr_scalar", "csr_vector"]):
            Messages._error("The CSR variants csr_scalar/csr_vector are only intended for GPUs.")

        if _check_paradigm("cuda") and proj._storage_format == "lil":
            proj._storage_format = "csr"
            if not isinstance(proj, SpecificProjection):
                Messages._info("LIL-type projections are not available for GPU devices ... default to CSR")

        if _check_paradigm("cuda") and proj._storage_format == "ell":
            Messages._info("We would recommend to use ELLPACK-R (format=ellr) on GPUs.")
        
def _check_prepost(populations, projections):
    """
    Checks that when a synapse uses pre.x r post.x, the variable x exists in the corresponding neuron
    """
    for proj in projections:

        for dep in  proj.synapse_type.description['dependencies']['pre']:
            if dep.startswith('sum('):
                target = re.findall(r'\(([\s\w]+)\)', dep)[0].strip()
                if not target in proj.pre.targets:
                    Messages._print(proj.synapse_type.equations)
                    Messages._error('The pre-synaptic population ' + proj.pre.name + ' receives no projection with the type ' + target)
                continue

            if not dep in proj.pre.attributes:
                Messages._print(proj.synapse_type.equations)
                Messages._error('The pre-synaptic population ' + proj.pre.name + ' has no variable called ' + dep)

        for dep in proj.synapse_type.description['dependencies']['post']:
            if dep.startswith('sum('):
                target = re.findall(r'\(([\s\w]+)\)', dep)[0].strip()
                if not target in proj.post.targets:
                    Messages._print(proj.synapse_type.equations)
                    Messages._error('The post-synaptic population ' + proj.post.name + ' receives no projection with the type ' + target)
                continue

            if not dep in proj.post.attributes:
                Messages._print(proj.synapse_type.equations)
                Messages._error('The post-synaptic population ' + proj.post.name + ' has no variable called ' + dep)


def _check_locality(populations, projections):
    """
    Checks that a global variable does not depend on local ones.
    """
    for proj in projections:

        for var in proj.synapse_type.description['variables']:

            if var['locality'] == 'global': # cannot depend on local or semiglobal variables
                # Inside the equation
                for v in var['dependencies']:
                    if _get_locality(v, proj.synapse_type.description) in ['local', 'semiglobal']:
                        Messages._print(var['eq'])
                        Messages._error('The global variable', var['name'], 'cannot depend on a synapse-specific/post-synaptic one:', v)

                # As pre/post dependencies
                deps = var['prepost_dependencies']
                if len(deps['pre']) > 0 or len(deps['post']) > 0 : 
                    Messages._print(proj.synapse_type.equations)
                    Messages._error('The global variable', var['name'], 'cannot depend on pre- or post-synaptic variables.')

            if var['locality'] == 'semiglobal': # cannot depend on pre-synaptic variables
                # Inside the equation
                for v in var['dependencies']:
                    if _get_locality(v, proj.synapse_type.description) == 'local':
                        Messages._print(var['eq'])
                        Messages._error('The postsynaptic variable', var['name'], 'cannot depend on a synapse-specific one:', v)

                # As pre/post dependencies
                deps = var['prepost_dependencies']
                if len(deps['pre']) > 0  : 
                    Messages._print(proj.synapse_type.equations)
                    Messages._error('The postsynaptic variable', var['name'], 'cannot depend on pre-synaptic ones (e.g. pre.r).')


def _get_locality(name, description):
    "Returns the locality of an attribute based on its name"
    for var in description['variables'] + description['parameters']:
        if var['name'] == name:
            return var['locality']
    return 'local'
