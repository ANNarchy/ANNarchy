from ANNarchy.core.Population import Population
from ANNarchy.core.Projection import Projection
import ANNarchy.core.Global as Global
import numpy as np

class DecodingProjection(Projection):
    """ 
    Decoding projection to transform spike trains into firing rates.

    The pre-synaptic population must be a spiking population, while the post-synaptic one must be rate-coded.

    Pre-synaptic spikes are accumulated for each post-synaptic neuron. A sliding window can be used to smoothen the results with the ``window`` parameter.

    The decoded firing rate is accessible in the post-synaptic neurons with ``sum(target)``.

    The projection can be connected using any method available in ``Projection`` (although all-to-all or many-to-one makes mostly sense). Delays are ignored. 

    The weight value allows to scale the firing rate: if you want a pre-synaptic firing rate of 100 Hz to correspond to a post-synaptic rate of 1.0, use ``w = 1./100.``.

    Example::

        pop1 = PoissonPopulation(1000, rates=100.)
        pop2 = Population(1, Neuron(equations="r=sum(exc)"))
        proj = DecodingProjection(pop1, pop2, 'exc', window=10.0)
        proj.connect_all_to_all(1.0)
    """
    def __init__(self, pre, post, target, window=0.0):
        """
        *Parameters*:
                
            * **pre**: pre-synaptic population.
            * **post**: post-synaptic population.
            * **target**: type of the connection.
            * **window**: duration of the time window to collect spikes (default: dt).
        """
        Projection.__init__(self, pre, post, target, None)
        # Check populations
        if not self.pre.neuron_type.type == 'spike':
            Global._error('The pre-synaptic population of a DecodingProjection must be spiking.')
            exit(0)
        if not self.post.neuron_type.type == 'rate':
            Global._error('The post-synaptic population of a DecodingProjection must be rate-coded.')
            exit(0)

        # Process window argument
        if window == 0.0:
            window = Global.config['dt']
        self.window = window

    def _generate(self):
        # Generate the code
        self._specific_template['declare_additional'] = """
    // Window
    int window = %(window)s;
    std::deque< std::vector< double > > rates_history ;
""" % { 'window': int(self.window/Global.config['dt']) }

        self._specific_template['init_additional'] = """
        rates_history = std::deque< std::vector<double> >(%(window)s, std::vector<double>(%(post_size)s, 0.0));
""" % { 'window': int(self.window/Global.config['dt']),'post_size': self.post.size }

        self._specific_template['psp_code'] = """
        // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s with target exc. event-based
        if (pop%(id_post)s._active){
            std::vector< std::pair<int, int> > proj%(id_proj)s_inv_post;
            std::vector<double> rates = std::vector<double>(%(post_size)s, 0.0);
            // Iterate over all incoming spikes
            for(int _idx_j = 0; _idx_j < pop%(id_pre)s.spiked.size(); _idx_j++){
                rk_j = pop%(id_pre)s.spiked[_idx_j];
                proj%(id_proj)s_inv_post = inv_rank[rk_j];
                nb_post = proj%(id_proj)s_inv_post.size();
                // Iterate over connected post neurons
                for(int _idx_i = 0; _idx_i < nb_post; _idx_i++){
                    // Retrieve the correct indices
                    i = proj%(id_proj)s_inv_post[_idx_i].first;
                    j = proj%(id_proj)s_inv_post[_idx_i].second;

                    // Increase the post-synaptic conductance
                    rates[post_rank[i]] +=  w[i][j];
                }
            }

            rates_history.push_front(rates);
            rates_history.pop_back();
            for(int i=0; i<post_rank.size(); i++){
                sum = 0.0;
                for(int step=0; step<window; step++){
                    sum += rates_history[step][post_rank[i]];
                }
                pop%(id_post)s._sum_%(target)s[post_rank[i]] += sum /float(window) * 1000. / dt  / float(pre_rank[i].size());
            }
        } // active
""" % { 'id_proj': self.id, 'id_pre': self.pre.id, 'id_post': self.post.id, 'target': self.target,
        'post_size': self.post.size}

        self._specific_template['psp_prefix'] = """
        int nb_post, i, j, rk_j, rk_post, rk_pre;
        double sum;
"""