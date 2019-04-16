#===============================================================================
#
#     NormProjection.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2018-2019  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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
import numpy as np

from ANNarchy.core.SpecificProjection import SpecificProjection
from ANNarchy.core import Global

class NormProjection(SpecificProjection):
    """
    Behaves like normal spike synapse but also generates a normalized conductance
    which means that the increase is normalized by the number of afferent synapses.

    Important is that the number of afferent synapses is across all dendrites
    of a neuron.

    Parameters:

    * **pre**: pre-synaptic Population or PopulationView.
    * **post**: pre-synaptic Population or PopulationView.
    * **target**: type of the connection.
    * **synapse**: a ``Synapse`` instance.
    * **variable**: target variable for normalized current.

    See also Projection.__init__().

    """
    def __init__(self, pre, post, target, variable, synapse=None, name=None, copied=False):
        # Instantiate the projection
        SpecificProjection.__init__(self, pre, post, target, synapse=synapse, name=name, copied=copied)
        self._variable = variable

        # Check populations
        if not self.pre.neuron_type.type == 'spike':
            Global._error('The pre-synaptic population of an NormProjection must be spiking.')
            
        if not self.pre.neuron_type.type == 'spike':
            Global._error('The post-synaptic population of an NormProjection must be spiking.')

        # Not on CUDA
        if Global._check_paradigm('cuda'):
            Global._error('AccProjections are not available on CUDA yet.')

    def _copy(self, pre, post):
        "Returns a copy of the population when creating networks. Internal use only."
        return NormProjection(pre=pre, post=post, target=self.target, variable=self._variable, synapse=self.synapse_type, name=self.name, copied=True)

    def _generate_omp(self):
        """
        """
        # Sanity Check
        found = False
        for var in self.pre.neuron_type.description['variables']:
            if var['name'] == self._variable:
                found = True                
                break

        if not found:
            Global._warning("NormProjection: variable might be invalid ...")

        # TODO: delays???
        if self.synapse_type.pre_axon_spike:
            pre_array = "tmp_spiked"
            pre_array_fusion = """
            std::vector<int> tmp_spiked = %(pre_array)s;
            tmp_spiked.insert( tmp_spiked.end(), pop%(id_pre)s.axonal.begin(), pop%(id_pre)s.axonal.end() );
""" % {'id_pre': self.pre.id, 'pre_array': 'pop'+str(self.pre.id)+'.spiked'}
        else:
            pre_array = 'pop'+str(self.pre.id)+'.spiked'
            pre_array_fusion = ""

        # nb_aff_synapse contains the number of all afferent synapses of this neuron
        # set after compile()
        if 'nb_aff_synapse' not in self.attributes:
            self.synapse_type.description['parameters'].append({'name': 'nb_aff_synapse',
                                                                'bounds': {},
                                                                'ctype': 'double',
                                                                'init': 1.0,
                                                                'locality': 'semiglobal'})
            self.attributes.append(['nb_aff_synapse'])

        # Generate Code Template Projection
        self._specific_template['psp_prefix'] = ""        
        self._specific_template['psp_code'] = """
        // Event-based summation
        if (_transmission && pop%(id_post)s._active){
            // Iterate over all incoming spikes
            %(spiked_array_fusion)s

            for(int _idx_j = 0; _idx_j < %(pre_array)s.size(); _idx_j++){
                int rk_j = %(pre_array)s[_idx_j];
                auto inv_post_ptr = inv_pre_rank.find(rk_j);
                if (inv_post_ptr == inv_pre_rank.end())
                    continue;
                std::vector< std::pair<int, int> >& inv_post = inv_post_ptr->second;
                int nb_post = inv_post.size();

                // Iterate over connected post neurons
                for(int _idx_i = 0; _idx_i < nb_post; _idx_i++){
                    // Retrieve the correct indices
                    int i = inv_post[_idx_i].first;
                    int j = inv_post[_idx_i].second;

                    pop%(id_post)s.g_%(target)s[post_rank[i]] += w[i][j];
                    pop%(id_post)s.%(var)s[post_rank[i]] += 1.0 / nb_aff_synapse[i];
                }
            }
        } // active
""" % { 'id_post': self.post.id,
        'target': self.target,
        'var': self._variable,
        'spiked_array_fusion': pre_array_fusion,
        'pre_array': pre_array 
      }

def update_num_aff_connections(net=None, verbose=False):
    """
    We need to set for all NormProjections the number of afferent
    connections for each projection (across multiple projections,
    otherwise pre_rank[i].size() would be sufficient.

    TODO: maybe we can launch this automatically after compile.__instantiate__()?
    """
    #
    # Thats basically the same code, but I have now no better idea ...
    if net != None:
        # iterate over all populations
        for pop in net.get_populations():
            nb_synapses = np.zeros((pop.size))

            # get afferent projections of this layer
            aff_proj = net.get_projections(pop.name)

            # we accumulate the number of synapses per dendrite
            # across all afferent projections of this layer.
            for i, proj in enumerate(aff_proj):
                # nb synapses per dendrite is oriented at the post_ranks list. If a neuron
                # does not receive any connection in THIS projection there is no entry
                nb_synapses_per_dend = np.array(net.get(proj).nb_synapses_per_dendrite())

                # Update global count
                for idx, rank in enumerate(proj.post_ranks):
                    nb_synapses[rank] += nb_synapses_per_dend[idx]

            # set number of afferent connections for correct normalization
            for i, proj in enumerate(aff_proj):
                if not isinstance(proj, NormProjection):
                    continue

                if verbose:
                    print('Update:', proj.pre.name,'->', proj.post.name, '(', proj.target, ') -- (', i+1, 'of', len(aff_proj),')')
                    if len(proj.post_ranks) != proj.post.size:
                        print('ranks:', proj.post_ranks)

                nb_synapses_per_dend = np.array(net.get(proj).nb_synapses_per_dendrite())

                if verbose:
                    print('before:', nb_synapses_per_dend)
                for idx, rank in enumerate(proj.post_ranks):
                    nb_synapses_per_dend[idx] = nb_synapses[rank]

                if verbose:
                    print('after:', nb_synapses_per_dend)
                net.get(proj).cyInstance.set_nb_aff_synapse(nb_synapses_per_dend)
    else:
        # iterate over all populations
        for pop in Global.populations():
            nb_synapses = np.zeros((pop.size))

            # get afferent projections of this layer
            aff_proj = Global.projections(0, post=pop.name)

            # we accumulate the number of synapses per dendrite
            # across all afferent projections of this layer.
            for i, proj in enumerate(aff_proj):
                # nb synapses per dendrite is oriented at the post_ranks list. If a neuron
                # does not receive any connection in THIS projection there is no entry
                nb_synapses_per_dend = np.array(proj.nb_synapses_per_dendrite())

                # Update global count
                for idx, rank in enumerate(proj.post_ranks):
                    nb_synapses[rank] += nb_synapses_per_dend[idx]

            # set number of afferent connections for correct normalization
            for i, proj in enumerate(aff_proj):
                if not isinstance(proj, NormProjection):
                    continue

                if verbose:
                    print('Update:', proj.pre.name,'->', proj.post.name, '(', proj.target, ') -- (', i+1, 'of', len(aff_proj),')')
                    if len(proj.post_ranks) != proj.post.size:
                        print('ranks:', proj.post_ranks)

                nb_synapses_per_dend = np.array(proj.nb_synapses_per_dendrite())

                if verbose:
                    print('before:', nb_synapses_per_dend)
                for idx, rank in enumerate(proj.post_ranks):
                    nb_synapses_per_dend[idx] = nb_synapses[rank]

                if verbose:
                    print('after:', nb_synapses_per_dend)
                proj.cyInstance.set_nb_aff_synapse(nb_synapses_per_dend)
