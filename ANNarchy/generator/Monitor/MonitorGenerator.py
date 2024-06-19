"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.ConfigManagement import get_global_config
from ANNarchy.generator.Monitor import BaseTemplates as RecTemplate
from ANNarchy.generator.Utils import tabify
from ANNarchy.extensions.bold import BoldMonitor

from ANNarchy.generator.Monitor import OpenMPTemplates
from ANNarchy.generator.Monitor import CUDATemplates

class MonitorGenerator(object):
    """
    Creates the required codes for recording population
    and projection data
    """
    def __init__(self, annarchy_dir, populations, projections, net_id):
        """
        Constructor, stores all required data for later
        following code generation step

        Parameters:

            * *annarchy_dir*: unique target directory for the generated code
              files; they are stored in 'generate' sub-folder
            * *populations*: list of populations
            * *populations*: list of projections
            * *net_id*: unique id for the current network

        """
        self._annarchy_dir = annarchy_dir
        self._populations = populations
        self._projections = projections
        self._net_id = net_id

    def generate(self):
        """
        Generate one file "Monitor.hpp" comprising of Monitor base class and inherited
        classes for each Population/Projection.

        Templates:

            record_base_class
        """
        record_class = ""
        # We generate for each of the population/projection in the network
        # a record class, containing by default all variables. The recording
        # is then enabled or disabled.
        for pop in self._populations:
            record_class += self._pop_recorder_class(pop)

        for proj in self._projections:
            record_class += self._proj_recorder_class(proj)

        code = RecTemplate.record_base_class % {'record_classes': record_class}

        # The approach for default populations/projections is not
        # feasible for specific monitors, so we handle them extra
        for mon in NetworkManager().get_monitors(net_id=self._net_id):
            if isinstance(mon, BoldMonitor):
                mon_dict = {
                    'pop_id': mon.object.id,
                    'pop_name': mon.object.name,
                    'mon_id': mon.id,
                    'float_prec': get_global_config('precision'),
                    'var_name': mon.variables[0],
                }
                code += mon._specific_template['cpp'] % mon_dict

        # Generate header code for the analysed pops and projs
        with open(self._annarchy_dir+'/generate/net'+str(self._net_id)+'/Monitor.hpp', 'w') as ofile:
            ofile.write(code)

    def _pop_recorder_class(self, pop):
        """
        Creates population recording class code.

        Returns:

            * complete code as string

        Templates:

            omp_population, cuda_population
        """
        if get_global_config('paradigm') == "openmp":
            template = OpenMPTemplates.omp_population
        elif get_global_config('paradigm') == "cuda":
            template = CUDATemplates.cuda_population
        else:
            raise NotImplementedError

        tpl_code = template['template']

        init_code = ""
        recording_code = ""
        recording_target_code = ""
        struct_code = ""
        size_in_bytes = ""
        clear_code = ""

        # The post-synaptic potential for rate-code (weighted sum) as well
        # as the conductance variables are handled seperatly.
        target_list = []
        targets = []
        for t in pop.neuron_type.description['targets']:
            if isinstance(t, list):
                for t2 in t:
                    targets.append(t2)
            else:
                targets.append(t)
        for t in pop.targets:
            if isinstance(t, list):
                for t2 in t:
                    targets.append(t2)
            else:
                targets.append(t)
        targets = sorted(list(set(targets)))

        if pop.neuron_type.type == 'rate':
            for target in targets:
                tar_dict = {'id': pop.id, 'type' : get_global_config('precision'), 'name': '_sum_'+target}
                struct_code += template['local']['struct'] % tar_dict
                init_code += template['local']['init'] % tar_dict
                recording_target_code += template['local']['recording'] % tar_dict
                clear_code += template['local']['clear'] % tar_dict
        else:
            for target in targets:
                tar_dict = {'id': pop.id, 'type' : get_global_config('precision'), 'name': 'g_'+target}
                struct_code += template['local']['struct'] % tar_dict
                init_code += template['local']['init'] % tar_dict
                recording_target_code += template['local']['recording'] % tar_dict
                clear_code += template['local']['clear'] % tar_dict

                # to skip this entry in the following loop
                target_list.append('g_'+target)

        # Record global and local attributes
        attributes = []
        for var in pop.neuron_type.description['variables']:
            # Skip targets
            if var['name'] in target_list:
                continue
            # Avoid doublons
            if var['name'] in attributes:
                continue
            attributes.append(var['name'])

            ids = {
                'id': pop.id,
                'name': var['name'],
                'type': var['ctype']
            }

            struct_code += template[var['locality']]['struct'] % ids
            init_code += template[var['locality']]['init'] % ids
            recording_code += template[var['locality']]['recording'] % ids
            clear_code += template[var['locality']]['clear'] % ids
            size_in_bytes += template[var['locality']]['size_in_bytes'] % ids

        # Record spike events
        if pop.neuron_type.type == 'spike':
            base_tpl = RecTemplate.recording_spike_tpl
            rec_dict = {
                'id': pop.id,
                'type' : 'long int',
                'name': 'spike',
                'rec_target': 'spiked'
            }

            struct_code += base_tpl['struct'] % rec_dict
            init_code += base_tpl['init'] % rec_dict
            recording_code += base_tpl['record'][get_global_config('paradigm')] % rec_dict
            size_in_bytes += base_tpl['size_in_bytes'][get_global_config('paradigm')] % rec_dict
            clear_code += base_tpl['clear'][get_global_config('paradigm')] % rec_dict

            # Record axon spike events
            if pop.neuron_type.axon_spike:
                rec_dict = {
                    'id': pop.id,
                    'type' : 'long int',
                    'name': 'axon_spike',
                    'rec_target': 'axonal'
                }

                struct_code += base_tpl['struct'] % rec_dict
                init_code += base_tpl['init'] % rec_dict
                recording_code += base_tpl['record'][get_global_config('paradigm')] % rec_dict

        ids = {
            'id': pop.id,
            'init_code': init_code,
            'struct_code': struct_code,
            'recording_code': recording_code,
            'recording_target_code': recording_target_code,
            'size_in_bytes': tabify(size_in_bytes, 2),
            'clear_monitor_code': clear_code
        }
        return tpl_code % ids

    def _proj_recorder_class(self, proj):
        """
        Generate the code for the recorder object.

        Returns:

            * complete code as string

        Templates:

            record
        """
        if get_global_config('paradigm') == "openmp":
            template = OpenMPTemplates.omp_projection
        elif get_global_config('paradigm') == "cuda":
            template = CUDATemplates.cuda_projection
        else:
            raise NotImplementedError

        # Specific template
        if 'monitor_class' in proj._specific_template.keys():
            return proj._specific_template['monitor_class']

        init_code = ""
        recording_code = ""
        size_in_bytes_code = ""
        clear_code = ""
        struct_code = ""

        attributes = []
        for var in proj.synapse_type.description['variables']:
            # Avoid doublons
            if var['name'] in attributes:
                continue
            attributes.append(var['name'])

            # Get the locality
            locality = var['locality']

            # Special case for single weights
            if var['name'] == "w" and proj._has_single_weight():
                locality = 'global'

            # Get the template for the structure declaration
            struct_code += template[locality]['struct'] % {'type' : var['ctype'], 'name': var['name']}

            # Get the initialization code
            init_code += template[locality]['init'] % {'type' : var['ctype'], 'name': var['name']}

            # Clear recorded data
            clear_code += template[locality]['clear'] % {'type' : var['ctype'], 'name': var['name']}

            # Memory requirement
            size_in_bytes_code += template[locality]['size_in_bytes'] % {'type' : var['ctype'], 'name': var['name']}

            # Get the recording code
            recording_code += template[locality]['recording'] % {
                'id': proj.id,
                'type' : var['ctype'],
                'name': var['name'],
                'float_prec': get_global_config('precision')
            }

        final_dict = {
            'id': proj.id,
            'init_code': init_code,
            'recording_code': recording_code,
            'size_in_bytes_code': size_in_bytes_code,
            'clear_code': clear_code,
            'struct_code': struct_code
        }

        return template['struct'] % final_dict
