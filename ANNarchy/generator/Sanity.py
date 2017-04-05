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
import ANNarchy.core.Global as Global
import re

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

    # Check variable names
    _check_reserved_names(populations, projections)

    # Check that projections are created before compile
    for proj in projections:
        if not proj._connection_method:
            Global._error('The projection between populations', proj.pre.id, 'and', proj.post.id, 'has not been connected.',
                            ' Call a connector method before compiling the network.')

    # Check that synapses access existing variables in the pre or post neurons
    _check_prepost(populations, projections)

    # Check locality of variable is respected
    _check_locality(populations, projections)




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

