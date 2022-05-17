#===============================================================================
#
#     Sanity.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
import re

from ANNarchy.core import Global
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.models.Synapses import DefaultSpikingSynapse, DefaultRateCodedSynapse

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
            Global._error('The projection between populations', proj.pre.id, 'and', proj.post.id, 'has not been connected.',
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
    if Global.config['paradigm'] == "openmp":
        for proj in projections:
            if proj._storage_format == "csr" and proj._storage_order == "pre_to_post":
                Global._warning("Compressed sparse row (CSR) and pre_to_post ordering representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "bsr":
                Global._warning("Blocked sparse row (BSR) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "coo":
                Global._warning("Coordinate (COO) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "ellr":
                Global._warning("ELLPACK-R (ELLR) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "ell":
                Global._warning("ELLPACK (ELL) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "hyb":
                Global._warning("Hybrid (ELL + COO) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "dense" and proj.synapse_type.type=="spike":
                Global._warning("Dense representation is an experimental feature for spiking models, we greatly appreciate bug reports.")
                break

    # GPU-related formats
    elif Global.config['paradigm'] == "cuda":
        for pop in populations:
            if pop.neuron_type.description['type'] == "spike":
                Global._warning('Spiking neurons on GPUs is an experimental feature. We greatly appreciate bug reports.')
                break

        for proj in projections:
            if proj._storage_format == "ellr":
                Global._warning("ELLPACK-R (ELLR) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "bsr":
                Global._warning("Blocked sparse row (BSR) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "coo":
                Global._warning("Coordinate (COO) representation is an experimental feature, we greatly appreciate bug reports.")
                break

        for proj in projections:
            if proj._storage_format == "hyb":
                Global._warning("Hybrid (ELL + COO) representation is an experimental feature, we greatly appreciate bug reports.")
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
                Global._print(pop.neuron_type.parameters)
                Global._print(pop.neuron_type.equations)
                Global._error(term + ' is a reserved variable name')

    # Check projections
    for proj in projections:
        # Reserved variable names
        for term in reserved_variables:
            if term in proj.attributes:
                Global._print(proj.synapse_type.parameters)
                Global._print(proj.synapse_type.equations)
                Global._error(term + ' is a reserved variable name')

def _check_storage_formats(projections):
    """
    ANNarchy 4.7 introduced a set of sparse matrix formats. Some of them are not implemented for
    all paradigms or might not support specific optimizations.
    """
    for proj in projections:
        # Most of the sparse matrix formats are not trivially invertable and therefore we can not implement
        # spiking models with them
        if proj.synapse_type.type == "spike" and proj._storage_format in ["ell", "ellr", "coo", "hyb"]:
            raise Global.ANNarchyException("Using 'storage_format="+ proj._storage_format + "' is not allowed for spiking synapses.", True)

        # Dense format is not implemented for GPUs and spiking models
        if proj._storage_format == "dense" and proj.synapse_type.type=="spike" and Global._check_paradigm("cuda"):
            raise Global.ANNarchyException("Dense representation is not available for spiking models on GPUs yet.", True)

        # For some of the sparse matrix formats we don't implemented plasticity yet.
        if proj.synapse_type.type == "spike" and proj._storage_format in ["dense"] and not isinstance(proj.synapse_type, DefaultSpikingSynapse):
            raise Global.ANNarchyException("Using 'storage_format="+ proj._storage_format + "' is only allowed for default spiking synapses yet.", True)

        # For some of the sparse matrix formats we don't implemented plasticity yet.
        if proj.synapse_type.type == "rate" and proj._storage_format in ["coo", "hyb"] and not isinstance(proj.synapse_type, DefaultRateCodedSynapse):
            raise Global.ANNarchyException("Using 'storage_format="+ proj._storage_format + "' is only allowed for default rate-coded synapses yet.", True)

        # OpenMP disabled?
        if proj._storage_format in ["bsr"] and Global.config["num_threads"]>1:
            raise Global.ANNarchyException("Using 'storage_format="+ proj._storage_format + "' is not available for OpenMP yet.", True)

        # Single weight optimization available?
        if proj._has_single_weight() and proj._storage_format in ["dense"]:
            raise Global.ANNarchyException("Using 'storage_format="+ proj._storage_format + "' is not allowed for single weight projections.", True)

        # Slicing available?
        if isinstance(proj.post, PopulationView) and proj._storage_format in ["dense"]:
            raise Global.ANNarchyException("Using 'storage_format="+ proj._storage_format + "' is not allowed for PopulationViews as target.", True)

        # In some cases we don't allow the usage of non-unifom delay
        if (proj.max_delay > 1 and proj.uniform_delay == -1):
            if Global._check_paradigm("cuda"):
                raise Global.ANNarchyException("Using non-uniform delays is not available for CUDA devices.", True)

            else:
                if proj._storage_format == "ellr":
                    raise Global.ANNarchyException("Using 'storage_format="+ proj._storage_format + "' is and non-uniform delays is not implemented.", True)

        if Global._check_paradigm("cuda") and proj._storage_format == "lil":
            proj._storage_format = "csr"
            Global._info("LIL-type projections are not available for GPU devices ... default to CSR")

        if Global._check_paradigm("cuda") and proj._storage_format == "ell":
            Global._info("We would recommend to use ELLPACK-R (format=ellr) on GPUs.")

def _check_prepost(populations, projections):
    """
    Checks that when a synapse uses pre.x r post.x, the variable x exists in the corresponding neuron
    """
    for proj in projections:

        for dep in  proj.synapse_type.description['dependencies']['pre']:
            if dep.startswith('sum('):
                target = re.findall(r'\(([\s\w]+)\)', dep)[0].strip()
                if not target in proj.pre.targets:
                    Global._print(proj.synapse_type.equations)
                    Global._error('The pre-synaptic population ' + proj.pre.name + ' receives no projection with the type ' + target)
                continue

            if not dep in proj.pre.attributes:
                Global._print(proj.synapse_type.equations)
                Global._error('The pre-synaptic population ' + proj.pre.name + ' has no variable called ' + dep)

        for dep in proj.synapse_type.description['dependencies']['post']:
            if dep.startswith('sum('):
                target = re.findall(r'\(([\s\w]+)\)', dep)[0].strip()
                if not target in proj.post.targets:
                    Global._print(proj.synapse_type.equations)
                    Global._error('The post-synaptic population ' + proj.post.name + ' receives no projection with the type ' + target)
                continue

            if not dep in proj.post.attributes:
                Global._print(proj.synapse_type.equations)
                Global._error('The post-synaptic population ' + proj.post.name + ' has no variable called ' + dep)


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
                        Global._print(var['eq'])
                        Global._error('The global variable', var['name'], 'cannot depend on a synapse-specific/post-synaptic one:', v)

                # As pre/post dependencies
                deps = var['prepost_dependencies']
                if len(deps['pre']) > 0 or len(deps['post']) > 0 : 
                    Global._print(proj.synapse_type.equations)
                    Global._error('The global variable', var['name'], 'cannot depend on pre- or post-synaptic variables.')

            if var['locality'] == 'semiglobal': # cannot depend on pre-synaptic variables
                # Inside the equation
                for v in var['dependencies']:
                    if _get_locality(v, proj.synapse_type.description) == 'local':
                        Global._print(var['eq'])
                        Global._error('The postsynaptic variable', var['name'], 'cannot depend on a synapse-specific one:', v)

                # As pre/post dependencies
                deps = var['prepost_dependencies']
                if len(deps['pre']) > 0  : 
                    Global._print(proj.synapse_type.equations)
                    Global._error('The postsynaptic variable', var['name'], 'cannot depend on pre-synaptic ones (e.g. pre.r).')


def _get_locality(name, description):
    "Returns the locality of an attribute based on its name"
    for var in description['variables'] + description['parameters']:
        if var['name'] == name:
            return var['locality']
    return 'local'
