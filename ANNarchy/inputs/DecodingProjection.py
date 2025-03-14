"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.SpecificProjection import SpecificProjection
from ANNarchy.intern.ConfigManagement import ConfigManager, _check_paradigm
from ANNarchy.intern import Messages
from ANNarchy.core import Global

class DecodingProjection(SpecificProjection):
    """
    Decoding projection to transform spike trains into firing rates.

    The pre-synaptic population must be a spiking population, while the post-synaptic one must be rate-coded.

    Pre-synaptic spikes are accumulated for each post-synaptic neuron. A sliding window can be used to smoothen the results with the ``window`` parameter.

    The decoded firing rate is accessible in the post-synaptic neurons with `sum(target)`.

    The projection can be connected using any method available in `Projection` (although all-to-all or many-to-one makes mostly sense). Delays are ignored.

    The weight value allows to scale the firing rate: if you want a pre-synaptic firing rate of 100 Hz to correspond to a post-synaptic rate of 1.0, use ``w = 1./100.``.

    Example:

    ```python
    net = ann.Network()

    pop1 = net.create(ann.PoissonPopulation(1000, rates=100.))
    pop2 = net.create(1, ann.Neuron(equations="r=sum(exc)"))

    proj = net.connect(ann.DecodingProjection(pop1, pop2, 'exc', window=10.0))
    proj.all_to_all(1.0)
    ```

    :param pre: pre-synaptic population.
    :param post: post-synaptic population.
    :param target: type of the connection.
    :param window: duration of the time window to collect spikes (default: dt).
    :param name: optional name.
    """
    def __init__(
            self, 
            pre:"Population", 
            post:"Population", 
            target:str, 
            window:float=0.0, 
            name:str=None, 
            copied:bool=False, 
            net_id=0):

        # Instantiate the projection
        SpecificProjection.__init__(self, pre, post, target, None, name, copied, net_id)

        # Check populations
        if not self.pre.neuron_type.type == 'spike':
            Messages._error('The pre-synaptic population of a DecodingProjection must be spiking.')

        if not self.post.neuron_type.type == 'rate':
            Messages._error('The post-synaptic population of a DecodingProjection must be rate-coded.')

        # Process window argument
        if window == 0.0:
            window = ConfigManager().get('dt', self.net_id)
        self.window = window

        # Disable openMP post-synaptic matrix split
        self._no_split_matrix = True

        # Not on CUDA
        if _check_paradigm('cuda', self.net_id):
            Messages._error('DecodingProjections are not available on CUDA yet.')

    def _copy(self, pre, post, net_id=None):
        "Returns a copy of the population when creating networks. Internal use only."
        copied_proj = DecodingProjection(
            pre=pre, post=post, target=self.target, window=self.window, 
            name=self.name, copied=True,
            net_id = self.net_id if net_id is None else net_id,
        )
        copied_proj._no_split_matrix = True
        return copied_proj

    def _generate_st(self):
        # Generate the code
        self._specific_template['declare_additional'] = """
    // Window
    int window = %(window)s;
    std::deque< std::vector< %(float_prec)s > > rates_history ;
""" % { 'window': int(self.window/ConfigManager().get('dt', self.net_id)), 'float_prec': ConfigManager().get('precision', self.net_id) }

        self._specific_template['init_additional'] = """
        rates_history = std::deque< std::vector< %(float_prec)s > >(%(window)s, std::vector< %(float_prec)s >(%(post_size)s, 0.0));
""" % { 'window': int(self.window/ConfigManager().get('dt', self.net_id)),'post_size': self.post.size, 'float_prec': ConfigManager().get('precision', self.net_id) }

        self._specific_template['psp_code'] = """
        if (pop%(id_post)s->_active) {
            std::vector< std::pair<int, int> > inv_post;
            std::vector< %(float_prec)s > rates = std::vector< %(float_prec)s >(%(post_size)s, 0.0);
            // Iterate over all incoming spikes
            for(int _idx_j = 0; _idx_j < pop%(id_pre)s->spiked.size(); _idx_j++){
                rk_j = pop%(id_pre)s->spiked[_idx_j];
                inv_post = inv_pre_rank[rk_j];
                nb_post = inv_post.size();
                // Iterate over connected post neurons
                for(int _idx_i = 0; _idx_i < nb_post; _idx_i++){
                    // Retrieve the correct indices
                    i = inv_post[_idx_i].first;
                    j = inv_post[_idx_i].second;

                    // Increase the post-synaptic conductance
                    rates[post_rank[i]] +=  %(weight)s;
                }
            }

            rates_history.push_front(rates);
            rates_history.pop_back();
            for(int i=0; i<post_rank.size(); i++){
                sum = 0.0;
                for(int step=0; step<window; step++){
                    sum += rates_history[step][post_rank[i]];
                }
                pop%(id_post)s->_sum_%(target)s[post_rank[i]] += sum / %(float_prec)s(window) * 1000. / dt / %(float_prec)s(pre_rank[i].size());
            }
        } // active
""" % { 
        'id_proj': self.id, 
        'id_pre': self.pre.id, 
        'id_post': self.post.id, 
        'target': self.target,
        'post_size': self.post.size, 
        'float_prec': ConfigManager().get('precision', self.net_id),
        'weight': "w" if self._has_single_weight() else "w[i][j]"
      }

        self._specific_template['psp_prefix'] = """
        int nb_post, i, j, rk_j, rk_post, rk_pre;
        %(float_prec)s sum;
""" % { 'float_prec': ConfigManager().get('precision', self.net_id) }

    def _generate_omp(self):
        # Generate the code
        self._specific_template['declare_additional'] = """
    // Window
    int window = %(window)s;
    std::deque< std::vector< %(float_prec)s > > rates_history ;
""" % { 'window': int(self.window/ConfigManager().get('dt', self.net_id)), 'float_prec': ConfigManager().get('precision', self.net_id) }

        self._specific_template['init_additional'] = """
        rates_history = std::deque< std::vector< %(float_prec)s > >(%(window)s, std::vector< %(float_prec)s >(%(post_size)s, 0.0));
""" % { 'window': int(self.window/ConfigManager().get('dt', self.net_id)),'post_size': self.post.size, 'float_prec': ConfigManager().get('precision', self.net_id) }

        self._specific_template['psp_code'] = """
        #pragma omp single
        {
            if (pop%(id_post)s->_active) {
                std::vector< std::pair<int, int> > inv_post;
                std::vector< %(float_prec)s > rates = std::vector< %(float_prec)s >(%(post_size)s, 0.0);
                // Iterate over all incoming spikes
                for(int _idx_j = 0; _idx_j < pop%(id_pre)s->spiked.size(); _idx_j++){
                    rk_j = pop%(id_pre)s->spiked[_idx_j];
                    inv_post = inv_pre_rank[rk_j];
                    nb_post = inv_post.size();
                    // Iterate over connected post neurons
                    for(int _idx_i = 0; _idx_i < nb_post; _idx_i++){
                        // Retrieve the correct indices
                        i = inv_post[_idx_i].first;
                        j = inv_post[_idx_i].second;

                        // Increase the post-synaptic conductance
                        rates[post_rank[i]] +=  %(weight)s;
                    }
                }

                rates_history.push_front(rates);
                rates_history.pop_back();
                for(int i=0; i<post_rank.size(); i++){
                    sum = 0.0;
                    for(int step=0; step<window; step++){
                        sum += rates_history[step][post_rank[i]];
                    }
                    pop%(id_post)s->_sum_%(target)s[post_rank[i]] += sum / %(float_prec)s(window) * 1000. / dt / %(float_prec)s(pre_rank[i].size());
                }
            } // active
        }
""" %   { 
        'id_proj': self.id, 
        'id_pre': self.pre.id, 
        'id_post': self.post.id, 
        'target': self.target,
        'post_size': self.post.size, 
        'float_prec': ConfigManager().get('precision', self.net_id),
        'weight': "w" if self._has_single_weight() else "w[i][j]"
        }

        self._specific_template['psp_prefix'] = """
        int nb_post, i, j, rk_j, rk_post, rk_pre;
        %(float_prec)s sum;
""" % { 'float_prec': ConfigManager().get('precision', self.net_id) }

    def _generate_cuda(self):
        raise Global.ANNarchyException("The DecodingProjection is not available on CUDA devices.", True)