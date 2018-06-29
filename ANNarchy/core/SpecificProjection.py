#===============================================================================
#
#     SpecificProjection.py
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
from ANNarchy.core.Projection import Projection
import ANNarchy.core.Global as Global

class SpecificProjection(Projection):
    """
    Interface class for user-defined definition of Projection objects. An inheriting
    class need to override the implementor functions _generate_[paradigm], otherwise
    a NotImplementedError exception will be thrown.
    """
    def __init__(self, pre, post, target, synapse=None, name=None, copied=False):
        """
        Initialization, receive parameters of Projection objects.

        *Parameters*:

            * **pre**: pre-synaptic population.
            * **post**: post-synaptic population.
            * **target**: type of the connection.
            * **window**: duration of the time window to collect spikes (default: dt).
        """
        Projection.__init__(self, pre=pre, post=post, target=target, synapse=synapse, name=name, copied=copied)

    def _generate(self):
        """
        Overridden method of Population, called during the code generation process.
        This function selects dependent on the chosen paradigm the correct implementor
        functions defined by the user.
        """
        if Global.config['paradigm'] == "openmp":
            self._generate_omp()
        elif Global.config['paradigm'] == "cuda":
            self._generate_cuda()
        else:
            raise NotImplementedError

    def _generate_omp(self):
        """
        Intended to be overridden by child class. Implememt code adjustments intended for single thread and openMP paradigm.
        """
        raise NotImplementedError

    def _generate_cuda(self):
        """
        Intended to be overridden by child class. Implememt code adjustments intended for CUDA paradigm.
        """
        raise NotImplementedError

class DecodingProjection(SpecificProjection):
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
        proj.connect_all_to_all(1.0, force_multiple_weights=True)

    """
    def __init__(self, pre, post, target, window=0.0, name=None, copied=False):
        """
        *Parameters*:
                
        * **pre**: pre-synaptic population.
        * **post**: post-synaptic population.
        * **target**: type of the connection.
        * **window**: duration of the time window to collect spikes (default: dt).
        """
        # Instantiate the projection
        SpecificProjection.__init__(self, pre, post, target, None, name, copied)

        # Check populations
        if not self.pre.neuron_type.type == 'spike':
            Global._error('The pre-synaptic population of a DecodingProjection must be spiking.')
            
        if not self.post.neuron_type.type == 'rate':
            Global._error('The post-synaptic population of a DecodingProjection must be rate-coded.')

        # Process window argument
        if window == 0.0:
            window = Global.config['dt']
        self.window = window

        # Not on CUDA
        if Global._check_paradigm('cuda'):
            Global._error('DecodingProjections are not available on CUDA yet.')

    def _copy(self, pre, post):
        "Returns a copy of the population when creating networks. Internal use only."
        return DecodingProjection(pre=pre, post=post, target=self.target, window=self.window, name=self.name, copied=True)

    def _generate_omp(self):
        # Generate the code
        self._specific_template['declare_additional'] = """
    // Window
    int window = %(window)s;
    std::deque< std::vector< %(float_prec)s > > rates_history ;
""" % { 'window': int(self.window/Global.config['dt']), 'float_prec': Global.config['precision'] }

        self._specific_template['init_additional'] = """
        rates_history = std::deque< std::vector< %(float_prec)s > >(%(window)s, std::vector< %(float_prec)s >(%(post_size)s, 0.0));
""" % { 'window': int(self.window/Global.config['dt']),'post_size': self.post.size, 'float_prec': Global.config['precision'] }

        self._specific_template['psp_code'] = """
        if (pop%(id_post)s._active){
            std::vector< std::pair<int, int> > inv_post;
            std::vector< %(float_prec)s > rates = std::vector< %(float_prec)s >(%(post_size)s, 0.0);
            // Iterate over all incoming spikes
            for(int _idx_j = 0; _idx_j < pop%(id_pre)s.spiked.size(); _idx_j++){
                rk_j = pop%(id_pre)s.spiked[_idx_j];
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
                pop%(id_post)s._sum_%(target)s[post_rank[i]] += sum /float(window) * 1000. / dt / float(pre_rank[i].size());
            }
        } // active
""" % { 'id_proj': self.id, 'id_pre': self.pre.id, 'id_post': self.post.id, 'target': self.target,
        'post_size': self.post.size, 'float_prec': Global.config['precision'],
        'weight': "w" if self._has_single_weight() else "w[i][j]"}

        self._specific_template['psp_prefix'] = """
        int nb_post, i, j, rk_j, rk_post, rk_pre;
        %(float_prec)s sum;
""" % { 'float_prec': Global.config['precision'] }
