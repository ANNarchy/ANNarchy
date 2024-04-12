"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.generator.Utils import tabify
from ANNarchy.intern.ConfigManagement import get_global_config

from .ProfileGenerator import ProfileGenerator
from .ProfileTemplate import profile_base_template, cpp11_profile_template, cpp11_omp_profile_template, cpp11_profile_header

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
        if get_global_config('num_threads') == 1:
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
                'prof_proj_post_event_pre': cpp11_profile_template['proj_post_event_pre'],
                'prof_proj_post_event_post': cpp11_profile_template['proj_post_event_post'],
                'prof_neur_step_pre': cpp11_profile_template['neur_step_pre'],
                'prof_neur_step_post': cpp11_profile_template['neur_step_post'],
                'prof_rng_pre': cpp11_profile_template['rng_pre'],
                'prof_rng_post': cpp11_profile_template['rng_post'],
                'prof_record_pre': cpp11_profile_template['record_pre'],
                'prof_record_post': cpp11_profile_template['record_post'],
                'prof_global_ops_pre': cpp11_profile_template['global_op_pre'],
                'prof_global_ops_post': cpp11_profile_template['global_op_post']
            }
        else:
            body_dict = {
                'prof_include': cpp11_omp_profile_template['include'],
                'prof_step_pre': cpp11_omp_profile_template['step_pre'],
                'prof_step_post': cpp11_omp_profile_template['step_post'],
                'prof_run_pre': cpp11_omp_profile_template['run_pre'],
                'prof_run_post': cpp11_omp_profile_template['run_post'],
                'prof_proj_psp_pre': cpp11_omp_profile_template['proj_psp_pre'],
                'prof_proj_psp_post': cpp11_omp_profile_template['proj_psp_post'],
                'prof_proj_post_event_pre': cpp11_omp_profile_template['proj_post_event_pre'],
                'prof_proj_post_event_post': cpp11_omp_profile_template['proj_post_event_post'],
                'prof_proj_step_pre': cpp11_omp_profile_template['proj_step_pre'],
                'prof_proj_step_post': cpp11_omp_profile_template['proj_step_post'],
                'prof_neur_step_pre': cpp11_omp_profile_template['neur_step_pre'],
                'prof_neur_step_post': cpp11_omp_profile_template['neur_step_post'],
                'prof_rng_pre': cpp11_omp_profile_template['rng_pre'],
                'prof_rng_post': cpp11_omp_profile_template['rng_post'],
                'prof_record_pre': cpp11_omp_profile_template['record_pre'],
                'prof_record_post': cpp11_omp_profile_template['record_post'],
                'prof_global_ops_pre': cpp11_omp_profile_template['global_op_pre'],
                'prof_global_ops_post': cpp11_omp_profile_template['global_op_post']
            }

        return body_dict

    def generate_init_network(self):
        if get_global_config('num_threads') == 1:
            return cpp11_profile_template['init']
        else:
            return cpp11_omp_profile_template['init']

    def generate_init_population(self, pop):
        """
        Generate initialization code for population
        """
        declare = """
    // Profiling
    Measurement* measure_step;   // update ODE/non-ODE
    Measurement* measure_rng;    // draw random numbers
    Measurement* measure_delay;  // delay variables (in many cases "r")
    Measurement* measure_sc;     // spike condition
"""
        init = """        // Profiling
        measure_step = Profiling::get_instance()->register_function("pop", "%(name)s", %(id)s, "step", "%(label)s");
        measure_rng = Profiling::get_instance()->register_function("pop", "%(name)s", %(id)s, "rng", "%(label)s");
        measure_delay = Profiling::get_instance()->register_function("pop", "%(name)s", %(id)s, "delay", "%(label)s");
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
    Measurement* measure_pe;
"""
        if isinstance(proj.target, str):
            target = proj.target
        else:
            target = proj.target[0]
            for tar in proj.target[1:]:
                target += "_"+tar

        init = """        // Profiling
        measure_psp = Profiling::get_instance()->register_function("proj", "%(name)s", %(id_proj)s, "psp", "%(label)s");
        measure_step = Profiling::get_instance()->register_function("proj", "%(name)s", %(id_proj)s, "step", "%(label)s");
        measure_pe = Profiling::get_instance()->register_function("proj", "%(name)s", %(id_proj)s, "post_event", "%(label)s");
""" % {'id_proj': proj.id, 'name': proj.name, 'label': proj.pre.name+'_'+proj.post.name+'_'+target}

        return declare, init

    def annotate_computesum_rate(self, proj, code):
        """
        annotate the computesum compuation code
        """
        if get_global_config('num_threads') == 1:
            prof_begin = cpp11_profile_template['compute_psp']['before']
            prof_end = cpp11_profile_template['compute_psp']['after']
        else:
            prof_begin = cpp11_omp_profile_template['compute_psp']['before']
            prof_end = cpp11_omp_profile_template['compute_psp']['after']

        prof_code = """
        // first run, measuring average time
%(prof_begin)s
%(code)s
%(prof_end)s
""" % {
        'code': code,
        'prof_begin': tabify(prof_begin, 2),
        'prof_end': tabify(prof_end,2)
        }

        return prof_code

    def annotate_computesum_spiking(self, proj, code):
        """
        annotate the computesum compuation code
        """
        if get_global_config('num_threads') == 1:
            prof_begin = cpp11_profile_template['compute_psp']['before'] % {'name': 'proj'+str(proj.id)}
            prof_end = cpp11_profile_template['compute_psp']['after'] % {'name': 'proj'+str(proj.id)}
        else:
            prof_begin = cpp11_omp_profile_template['compute_psp']['before'] % {'name': 'proj'+str(proj.id)}
            prof_end = cpp11_omp_profile_template['compute_psp']['after'] % {'name': 'proj'+str(proj.id)}

        prof_code = """
        // first run, measuring average time
%(prof_begin)s
%(code)s
%(prof_end)s
""" % {'code': code,
       'prof_begin': tabify(prof_begin, 2),
       'prof_end': tabify(prof_end,2)
       }
        return prof_code

    def annotate_update_synapse(self, proj, code):
        """
        annotate the update synapse code, generated by ProjectionGenerator.update_synapse()
        """
        if get_global_config('num_threads') == 1:        
            prof_begin = cpp11_profile_template['update_synapse']['before']
            prof_end = cpp11_profile_template['update_synapse']['after']
        else:
            prof_begin = cpp11_omp_profile_template['update_synapse']['before']
            prof_end = cpp11_omp_profile_template['update_synapse']['after']

        prof_code = """
// first run, measuring average time
%(prof_begin)s
%(code)s
%(prof_end)s
""" % {'code': code,
       'prof_begin': tabify(prof_begin, 2),
       'prof_end': tabify(prof_end,2)
       }

        return prof_code

    def annotate_post_event(self, proj, code):
        """
        annotate the post-event code
        """
        if get_global_config('num_threads') == 1:
            prof_begin = cpp11_profile_template['post_event']['before']
            prof_end = cpp11_profile_template['post_event']['after']
        else:
            prof_begin = cpp11_omp_profile_template['post_event']['before']
            prof_end = cpp11_omp_profile_template['post_event']['after']

        prof_dict = {
            'code': code,
            'prof_begin': tabify(prof_begin,2),
            'prof_end': tabify(prof_end,2)
        }
        prof_code = """
%(prof_begin)s
%(code)s
%(prof_end)s
""" % prof_dict

        return prof_code

    def annotate_update_neuron(self, pop, code):
        """
        annotate the update neuron code
        """
        if get_global_config('num_threads') == 1:        
            prof_begin = cpp11_profile_template['update_neuron']['before'] % {'name': pop.name}
            prof_end = cpp11_profile_template['update_neuron']['after'] % {'name': pop.name}
        else:
            prof_begin = cpp11_omp_profile_template['update_neuron']['before'] % {'name': pop.name}
            prof_end = cpp11_omp_profile_template['update_neuron']['after'] % {'name': pop.name}

        prof_code = """
        // first run, measuring average time
%(prof_begin)s
%(code)s
%(prof_end)s
""" % {'code': code,
       'prof_begin': tabify(prof_begin, 2),
       'prof_end': tabify(prof_end,2)
       }
        return prof_code

    def annotate_spike_cond(self, pop, code):
        """
        annotate the spike condition code
        """
        if get_global_config('num_threads') == 1:
            prof_begin = cpp11_profile_template['spike_gather']['before'] % {'name': pop.name}
            prof_end = cpp11_profile_template['spike_gather']['after'] % {'name': pop.name}
        else:
            prof_begin = cpp11_omp_profile_template['spike_gather']['before'] % {'name': pop.name}
            prof_end = cpp11_omp_profile_template['spike_gather']['after'] % {'name': pop.name}

        prof_dict = {
            'code': code,
            'prof_begin': tabify(prof_begin,2),
            'prof_end': tabify(prof_end,2)
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
        if get_global_config('num_threads') == 1:
            prof_begin = cpp11_profile_template['update_rng']['before'] % {'name': pop.name}
            prof_end = cpp11_profile_template['update_rng']['after'] % {'name': pop.name}
        else:
            prof_begin = cpp11_omp_profile_template['update_rng']['before'] % {'name': pop.name}
            prof_end = cpp11_omp_profile_template['update_rng']['after'] % {'name': pop.name}

        prof_dict = {
            'code': code,
            'prof_begin': tabify(prof_begin,2),
            'prof_end': tabify(prof_end,2)
        }
        prof_code = """
%(prof_begin)s
%(code)s
%(prof_end)s
"""
        return prof_code % prof_dict

    def annotate_update_delay(self, pop, code):
        """
        annotate update delay kernel (only for CPUs available)
        """
        if get_global_config('num_threads') == 1:
            prof_begin = cpp11_profile_template['update_delay']['before'] % {'name': pop.name}
            prof_end = cpp11_profile_template['update_delay']['after'] % {'name': pop.name}
        else:
            prof_begin = cpp11_omp_profile_template['update_delay']['before'] % {'name': pop.name}
            prof_end = cpp11_omp_profile_template['update_delay']['after'] % {'name': pop.name}

        prof_dict = {
            'code': code,
            'prof_begin': tabify(prof_begin,2),
            'prof_end': tabify(prof_end,2)
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
            'paradigm': get_global_config('paradigm'),
            'num_threads': get_global_config('num_threads')
        }
        config = get_global_config('paradigm') + '_'  + str(get_global_config('num_threads')) + 'threads'

        timer_import = "#include <chrono>"
        timer_start = "std::chrono::time_point<std::chrono::steady_clock> _profiler_start;"
        timer_init = "_profiler_start = std::chrono::steady_clock::now();"
        return profile_base_template % {
            'timer_import': timer_import,
            'timer_start_decl': timer_start,
            'timer_init': timer_init,
            'config': config,
            'result_file': "results_%(config)s.xml" % {'config':config} if get_global_config('profile_out') == None else get_global_config('profile_out'),
            'config_xml': config_xml,
            'measurement_class': cpp11_profile_header
        }
