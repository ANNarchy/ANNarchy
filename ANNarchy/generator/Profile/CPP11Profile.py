#===============================================================================
#
#     CPP11Profile.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2016-2018  Julien Vitay <julien.vitay@gmail.com>,
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
from ANNarchy.core import Global

from .ProfileGenerator import ProfileGenerator
from .ProfileTemplate import cpp11_profile_template, cpp11_profile_header

class CPP11Profile(ProfileGenerator):
    """
    Extent the generated code by profiling annotations using the C++11
    steady clock counter.
    """
    def __init__(self, annarchy_dir, net_id):
        ProfileGenerator.__init__(self, annarchy_dir, net_id)

    def generate(self):
        """
        Generate Profiling class code, called from Generator instance.
        """
        # Generate header for profiling
        with open(self.annarchy_dir+'/generate/net'+str(self._net_id)+'/Profiling.h', 'w') as ofile:
            ofile.write(self._generate_header())

    def generate_body_dict(self):
        """
        Creates a dictionary, contain profile code snippets.
        """
        body_dict = {
            'prof_include': cpp11_profile_template['include'],
            'prof_step_pre': cpp11_profile_template['step_pre'],
            'prof_step_post': cpp11_profile_template['step_post'],
            'prof_run_pre': cpp11_profile_template['run_pre'],
            'prof_run_post': cpp11_profile_template['run_post'],
            'prof_proj_psp_pre': cpp11_profile_template['proj_psp_pre'],
            'prof_proj_psp_post': cpp11_profile_template['proj_psp_post'],
            'prof_proj_step_pre': cpp11_profile_template['proj_step_pre'],
            'prof_proj_step_post': cpp11_profile_template['proj_step_post'],
            'prof_neur_step_pre': cpp11_profile_template['neur_step_pre'],
            'prof_neur_step_post': cpp11_profile_template['neur_step_post'],
            'prof_rng_pre': cpp11_profile_template['rng_pre'],
            'prof_rng_post': cpp11_profile_template['rng_post'],
            'prof_record_pre': cpp11_profile_template['record_pre'],
            'prof_record_post': cpp11_profile_template['record_post']
        }
        return body_dict

    def generate_init_network(self):
        return cpp11_profile_template['init']

    def generate_init_population(self, pop):
        """
        Generate initialization code for population
        """
        declare = """
    // Profiling
    Measurement* measure_step;   // update ODE/non-ODE
    Measurement* measure_rng;    // draw random numbers
    Measurement* measure_sc;     // spike condition
"""
        init = """        // Profiling
        measure_step = Profiling::get_instance()->register_function("pop", "%(name)s", %(id)s, "step", "%(label)s");
        measure_rng = Profiling::get_instance()->register_function("pop", "%(name)s", %(id)s, "rng", "%(label)s");
        measure_sc = Profiling::get_instance()->register_function("pop", "%(name)s", %(id)s, "spike", "%(label)s");
""" % {'name': pop.name, 'id': pop.id, 'label': pop.name}

        return declare, init

    def generate_init_projection(self, proj):
        """
        Generate initialization code for projection
        """
        declare = """
    Measurement* measure_psp;
    Measurement* measure_step;
"""
        init = """        // Profiling
        measure_psp = Profiling::get_instance()->register_function("proj", "%(name)s", %(id_proj)s, "psp", "%(label)s");
        measure_step = Profiling::get_instance()->register_function("proj", "%(name)s", %(id_proj)s, "step", "%(label)s");
""" % {'id_proj': proj.id, 'name': proj.name, 'label': proj.pre.name+'_'+proj.post.name+'_'+proj.target}

        return declare, init

    def annotate_computesum_rate(self, proj, code):
        """
        annotate the computesum compuation code
        """
        prof_begin = cpp11_profile_template['compute_psp']['before']
        prof_end = cpp11_profile_template['compute_psp']['after']

        prof_code = """
        // first run, measuring average time
        %(prof_begin)s
%(code)s
        %(prof_end)s
""" % {
        'code': code,
        'prof_begin': prof_begin,
        'prof_end': prof_end
        }

        return prof_code

    def annotate_computesum_spiking(self, proj, code):
        """
        annotate the computesum compuation code
        """
        prof_begin = cpp11_profile_template['compute_psp']['before'] % {'name': 'proj'+str(proj.id)}
        prof_end = cpp11_profile_template['compute_psp']['after'] % {'name': 'proj'+str(proj.id)}

        prof_code = """
        // first run, measuring average time
        %(prof_begin)s
%(code)s
        %(prof_end)s
""" % {'code': code,
       'prof_begin': prof_begin,
       'prof_end': prof_end
       }
        return prof_code

    def annotate_update_synapse(self, proj, code):
        """
        annotate the update synapse code, generated by ProjectionGenerator.update_synapse()
        """
        prof_begin = cpp11_profile_template['update_synapse']['before']
        prof_end = cpp11_profile_template['update_synapse']['after']

        prof_code = """
// first run, measuring average time
%(prof_begin)s
%(code)s
%(prof_end)s
""" % {'code': code,
       'prof_begin': prof_begin,
       'prof_end': prof_end
       }

        return prof_code

    def annotate_update_neuron(self, pop, code):
        """
        annotate the update neuron code
        """
        prof_begin = cpp11_profile_template['update_neuron']['before'] % {'name': pop.name}
        prof_end = cpp11_profile_template['update_neuron']['after'] % {'name': pop.name}

        prof_code = """
        // first run, measuring average time
        %(prof_begin)s
%(code)s
        %(prof_end)s
""" % {'code': code,
       'prof_begin': prof_begin,
       'prof_end': prof_end
       }
        return prof_code

    def annotate_spike_cond(self, pop, code):
        """
        annotate the spike condition code
        """
        prof_begin = cpp11_profile_template['spike_gather']['before'] % {'name': pop.name}
        prof_end = cpp11_profile_template['spike_gather']['after'] % {'name': pop.name}

        prof_dict = {
            'code': code,
            'prof_begin': prof_begin,
            'prof_end': prof_end
        }
        prof_code = """
        %(prof_begin)s
%(code)s
        %(prof_end)s
""" % prof_dict

        return prof_code

    def annotate_update_rng(self, pop, code):
        """
        annotate update rng kernel (only for CPUs available)
        """
        prof_begin = cpp11_profile_template['update_rng']['before'] % {'name': pop.name}
        prof_end = cpp11_profile_template['update_rng']['after'] % {'name': pop.name}

        prof_dict = {
            'code': code,
            'prof_begin': prof_begin,
            'prof_end': prof_end
        }
        prof_code = """
        %(prof_begin)s
%(code)s
        %(prof_end)s
"""
        return prof_code % prof_dict

    def _generate_header(self):
        """
        generate Profiling.h
        """
        config_xml = """
        _out_file << "  <config>" << std::endl;
        _out_file << "    <paradigm>%(paradigm)s</paradigm>" << std::endl;
        _out_file << "    <num_threads>%(num_threads)s</num_threads>" << std::endl;
        _out_file << "  </config>" << std::endl;
        """ % {
            'paradigm': Global.config["paradigm"],
            'num_threads': Global.config["num_threads"]
        }
        config = Global.config["paradigm"] + '_'  + str(Global.config["num_threads"]) + 'threads'
        return cpp11_profile_header % {
            'result_file': "results_%(config)s.xml" % {'config':config} if Global.config['profile_out'] == None else Global.config['profile_out'],
            'config_xml': config_xml
        }
