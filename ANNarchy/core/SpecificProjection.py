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
        # Generate the code
        self._generate_code()

    def _generate_code(self):
        # Generate the code
        self.generator['omp']['header_proj_struct'] = """#pragma once

#include "pop%(id_pre)s.hpp"
#include "pop%(id_post)s.hpp"

extern PopStruct%(id_pre)s pop%(id_pre)s;
extern PopStruct%(id_post)s pop%(id_post)s;

/////////////////////////////////////////
// proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s with target %(target)s
/////////////////////////////////////////
struct ProjStruct%(id_proj)s{
    // number of dendrites
    int size;

    // Learning flag
    bool _learning;

    // Connectivity
    std::vector<int> post_rank ;
    std::vector< std::vector< int > > pre_rank ;

    std::map< int, std::vector< std::pair<int, int> > > inv_rank ;
    
    // Local parameter w
    std::vector< std::vector<double > > w;

    // Window
    int window = %(window)s;
    std::deque< std::vector< double > > rates_history ;


    void init_projection() {

        _learning = true;

        inv_rank =  std::map< int, std::vector< std::pair<int, int> > > ();
        for(int i=0; i<pre_rank.size(); i++){
            for(int j=0; j<pre_rank[i].size(); j++){
                inv_rank[pre_rank[i][j]].push_back(std::pair<int, int>(i,j));
            }
        }

        rates_history = std::deque< std::vector<double> >(%(window)s, std::vector<double>(%(post_size)s, 0.0));

    }

    void compute_psp() {
        int nb_post, i, j, rk_j, rk_post, rk_pre;
        double sum;

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

    }
    
    void update_synapse() {
    }

    // Accessors for c-wrapper
    int get_size() { return size; }
    void set_size(int new_size) { size = new_size; }
    std::vector<int> get_post_rank() { return post_rank; }
    void set_post_rank(std::vector<int> ranks) { post_rank = ranks; }
    std::vector< std::vector<int> > get_pre_rank() { return pre_rank; }
    void set_pre_rank(std::vector< std::vector<int> > ranks) { pre_rank = ranks; }
    int nb_synapses(int n) { return pre_rank[n].size(); }


    // Local parameter w
    std::vector<std::vector< double > > get_w() { return w; }
    std::vector<double> get_dendrite_w(int rk) { return w[rk]; }
    double get_synapse_w(int rk_post, int rk_pre) { return w[rk_post][rk_pre]; }
    void set_w(std::vector<std::vector< double > >value) { w = value; }
    void set_dendrite_w(int rk, std::vector<double> value) { w[rk] = value; }
    void set_synapse_w(int rk_post, int rk_pre, double value) { w[rk_post][rk_pre] = value; }

};

""" % {'id_proj': self.id, 'id_pre': self.pre.id, 'id_post': self.post.id, 'target': self.target,
        'window': int(self.window/Global.config['dt']), 'post_size': self.post.size}
