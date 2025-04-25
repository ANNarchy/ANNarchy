"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

class ProfileGenerator(object):
    """
    Base class to extent the generated code by profiling annotations.
    """
    def __init__(self, annarchy_dir, net_id):
        """
        Initialize ProfileGenerator.
        """
        self.annarchy_dir = annarchy_dir
        self._net_id = net_id

    def generate(self):
        """
        Called from Codegenerator, this method is responsible for
        creation of header file for the Profiling class.
        """
        raise NotImplementedError

    def generate_body_dict(self):
        """
        Creates a dictionary, contain profile code snippets. Should
        be overwritten by the specific measurement classes. This
        dict is also used in absence of profiling.
        """
        body_dict = {
            'prof_include': "",
            'prof_step_pre': "",
            'prof_step_post': "",
            'prof_run_pre': "",
            'prof_run_post': "",
            'prof_proj_psp_pre': "",
            'prof_proj_psp_post': "",
            'prof_proj_step_pre': "",
            'prof_proj_step_post': "",
            'prof_proj_post_event_pre': "",
            'prof_proj_post_event_post': "",
            'prof_neur_step_pre': "",
            'prof_neur_step_post': "",
            'prof_record_pre': "",
            'prof_record_post': "",
            'prof_rng_pre': "",
            'prof_rng_post': "",
            'prof_global_ops_pre': "",
            'prof_global_ops_post': ""
        }
        return body_dict

    def generate_include(self):
        "Implemented by child class"
        raise NotImplementedError

    def generate_init_network(self):
        "Implemented by child class"
        raise NotImplementedError

    def generate_init_population(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def generate_init_projection(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def annotate_computesum_rate(self, proj, code):
        "Implemented by child class"
        raise NotImplementedError

    def annotate_computesum_spiking(self, proj, code):
        "Implemented by child class"
        raise NotImplementedError

    def annotate_update_synapse(self, proj, code):
        "Implemented by child class"
        raise NotImplementedError

    def annotate_post_event(self, proj, code):
        "Implemented by child class"
        raise NotImplementedError

    def annotate_update_neuron(self, pop, code):
        "Implemented by child class"
        raise NotImplementedError

    def annotate_spike_cond(self, pop, code):
        "Implemented by child class"
        raise NotImplementedError

    def annotate_update_rng(self, pop, code):
        "Implemented by child class"
        raise NotImplementedError

    def annotate_update_delay(self, pop, code):
        "Implemented by child class"
        raise NotImplementedError

    def _generate_header(self):
        "Implemented by child class"
        raise NotImplementedError