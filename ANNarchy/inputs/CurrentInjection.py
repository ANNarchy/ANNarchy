"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.SpecificProjection import SpecificProjection
from ANNarchy.intern.ConfigManagement import ConfigManager, _check_paradigm
from ANNarchy.intern import Messages

from ANNarchy.core.PopulationView import PopulationView

class CurrentInjection(SpecificProjection):
    """
    Inject current from a rate-coded population into a spiking population.

    The pre-synaptic population must be be rate-coded, the post-synaptic one must be spiking, both must have the same size and no plasticity is allowed.

    For each post-synaptic neuron, the current `g_target` (e.g. `g_exc`or `g_inh`) will be set at each time step to the firing rate `r` of the pre-synaptic neuron with the same rank.

    The projection must be connected with `connect_current()`, which takes no parameter and does not accept delays. It is equivalent to `one_to_one(weights=1.0)`.

    Example:

    ```python
    inp = net.create(100, ann.Neuron(equations="r = 5*sin(t/1000)"))
    pop = net.create(100, ann.Izhikevich)

    proj = net.connect(ann.CurrentInjection(inp, pop, 'exc'))
    proj.connect_current()
    ```

    :param pre: pre-synaptic population.
    :param post: post-synaptic population.
    :param target: type of the connection.
    :param name: optional name.

    """
    def __init__(self, pre:"Population", post:"Population", target:str, name:str=None, copied=False, net_id=0):
        """
        """
        # Instantiate the projection
        SpecificProjection.__init__(self, pre, post, target, None, name, copied, net_id)

        # Check populations
        if not self.pre.neuron_type.type == 'rate':
            Messages._error('The pre-synaptic population of a CurrentInjection must be rate-coded.')

        if not self.post.neuron_type.type == 'spike':
            Messages._error('The post-synaptic population of a CurrentInjection must be spiking.')

        if not self.post.size == self.pre.size:
            Messages._error('CurrentInjection: The pre- and post-synaptic populations must have the same size.')

        if _check_paradigm("cuda", self.net_id) and (isinstance(pre, PopulationView) or isinstance(post, PopulationView)):
            Messages._error("CurrentInjection on GPUs is not allowed for PopulationViews")

        # Prevent automatic split of matrices
        self._no_split_matrix = True

    def _copy(self, pre, post, net_id=None):
        "Returns a copy of the population when creating networks. Internal use only."
        return CurrentInjection(pre=pre, post=post, target=self.target, name=self.name, copied=True, net_id = self.net_id if net_id is None else net_id)

    def _generate_st(self):
        # Generate the code
        self._specific_template['psp_code'] = """
        if (pop%(id_post)s->_active) {
            for (int i=0; i<post_rank.size(); i++) {
                pop%(id_post)s->g_%(target)s[post_rank[i]] += pop%(id_pre)s->r[pre_rank[i][0]];
            }
        } // active
""" % { 'id_pre': self.pre.id, 'id_post': self.post.id, 'target': self.target}

    def _generate_omp(self):
        # Generate the code
        self._specific_template['psp_code'] = """
        if (pop%(id_post)s->_active) {
            #pragma omp for
            for (int i=0; i<post_rank.size(); i++) {
                pop%(id_post)s->g_%(target)s[post_rank[i]] += pop%(id_pre)s->r[pre_rank[i][0]];
            }
        } // active
""" % { 'id_pre': self.pre.id, 'id_post': self.post.id, 'target': self.target}

    def _generate_cuda(self):
        """
        Generate the CUDA code.

        For a first implementation we take a rather simple approach:

        * We use only one block for the kernel and each thread computes
        one synapse/post-neuron entry (which is equal as we use one2one).

        * We use only the pre-synaptic firing rate, no other variables.

        * We ignore the synaptic weight.
        """
        ids = {
            'id_proj': self.id,
            'id_post': self.post.id,
            'id_pre': self.pre.id,
            'target': self.target,
            'float_prec': ConfigManager().get('precision', self.net_id)
        }

        self._specific_template['psp_body'] = """
__global__ void cu_proj%(id_proj)s_psp(int post_size, %(float_prec)s *pre_r, %(float_prec)s *g_%(target)s) {
    int n = threadIdx.x;

    while (n < post_size) {
        g_%(target)s[n] += pre_r[n];

        n += blockDim.x;
    }
}
""" % ids
        self._specific_template['psp_invoke'] = """
void proj%(id_proj)s_psp(RunConfig cfg, int post_size, %(float_prec)s *pre_r, %(float_prec)s *g_%(target)s) {
    cu_proj%(id_proj)s_psp<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>>(post_size, pre_r, g_%(target)s);
}
""" % ids
        self._specific_template['psp_header'] = """void proj%(id_proj)s_psp(RunConfig cfg, int post_size, %(float_prec)s *pre_r, %(float_prec)s *g_%(target)s);""" % ids
        self._specific_template['psp_call'] = """
    proj%(id_proj)s_psp(
        RunConfig(1, 192, 0, proj%(id_proj)s->stream),
        pop%(id_post)s->size,
        pop%(id_pre)s->gpu_r,
        pop%(id_post)s->gpu_g_%(target)s
    );
""" % ids

    def connect_current(self):
        return self.one_to_one(weights=1.0)