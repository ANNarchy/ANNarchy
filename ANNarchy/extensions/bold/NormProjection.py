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
from ANNarchy.core.SpecificProjection import SpecificProjection
from ANNarchy.core import Global    

class NormProjection(SpecificProjection):
    """
    Behaves like normal spike synapse but also generates a normalized conductance
    which means that the increase is normalized by the number of afferent synapses.

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
