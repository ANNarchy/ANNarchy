"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import numpy as np

from ANNarchy.intern.SpecificProjection import SpecificProjection
from ANNarchy.intern.ConfigManagement import _check_paradigm
from ANNarchy.intern import Messages
from ANNarchy.core import Global

class NormProjection(SpecificProjection):
    """
    Behaves like normal spike synapse but also generates a normalized conductance
    which means that the increase is normalized by the number of afferent synapses.

    Important is that the number of afferent synapses is across all dendrites
    of a neuron.

    :param pre: pre-synaptic Population or PopulationView.
    :param post: pre-synaptic Population or PopulationView.
    :param target: type of the connection.
    :param synapse: a ``Synapse`` instance.
    :param variable: target variable for normalized current.

    See also Projection.__init__().

    """
    def __init__(self, pre, post, target, variable, synapse=None, name=None, copied=False, net_id=0):
        # Instantiate the projection
        SpecificProjection.__init__(self, pre, post, target, synapse=synapse, name=name, copied=copied, net_id=net_id)
        self._variable = variable

        # Check populations
        if not self.pre.neuron_type.type == 'spike':
            Messages._error('The pre-synaptic population of an NormProjection must be spiking.')

        if not self.pre.neuron_type.type == 'spike':
            Messages._error('The post-synaptic population of an NormProjection must be spiking.')

        if synapse != None and not copied:
            Messages._error('NormProjection does not allow the usage of customized spiking synapses yet.')

        # Not on CUDA
        if _check_paradigm('cuda', self.net_id):
            Messages._error('NormProjections are not available on CUDA yet.')

        # Prevent automatic split of matrices
        self._no_split_matrix = True

    def _copy(self, pre, post, net_id=None):
        "Returns a copy of the population when creating networks. Internal use only."
        return NormProjection(pre=pre, post=post, target=self.target, variable=self._variable, synapse=self.synapse_type, name=self.name, copied=True, net_id=self.net_id if net_id is None else net_id)

    def _generate_st(self):
        """
        """
        # Sanity Check
        found = False
        for var in self.post.neuron_type.description['variables']:
            if var['name'] == self._variable:
                found = True
                break

        if not found:
            Messages._error("NormProjection: variable `"+self._variable+"` might be invalid. Please check the neuron model of population", self.post.name)

        # TODO: delays???
        if self.synapse_type.pre_axon_spike:
            pre_array = "tmp_spiked"
            pre_array_fusion = """
            std::vector<int> tmp_spiked = %(pre_array)s;
            tmp_spiked.insert( tmp_spiked.end(), pop%(id_pre)s->axonal.begin(), pop%(id_pre)s->axonal.end() );
""" % {'id_pre': self.pre.id, 'pre_array': 'pop'+str(self.pre.id)+'.spiked'}
        else:
            pre_array = 'pop'+str(self.pre.id)+'->spiked'
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

        # Get some more statements from user, normally done by the CodeGenerator
        if self._has_single_weight():
            psp_rside = "w"
        else:
            psp_rside = "w[i][j]" # default psp: g_target += w
        axon_code = ""
        indices = {
            'local_index': "[i][j]",
            'semiglobal_index': "[j]",
            'global_index': "[]"
        } # TODO: only true for openMP

        for var in self.synapse_type.description['pre_axon_spike']:
            if var['name'] == "g_target":
                psp_rside = var['cpp'].split("=")[1] % indices
            else:
                axon_code += var['cpp'] % indices

        # Only if needed. I don't really like the second loop, but it's for testing first
        if len(axon_code) > 0:
            axon_code = """
            for(int _idx_j = 0; _idx_j < pop%(id_pre)s->axonal.size(); _idx_j++){
                int rk_j = pop%(id_pre)s->axonal[_idx_j];
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

                    %(code)s
                }
            }
""" % {
    'id_pre': self.pre.id,
    'code': axon_code
}

        #
        # Generate Code Template Projection
        self._specific_template['psp_prefix'] = ""        
        self._specific_template['psp_code'] = """
        // Event-based summation
        if (_transmission && pop%(id_post)s->_active) {
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

                    pop%(id_post)s->g_%(target)s[post_rank[i]] += %(psp_rside)s;
                    pop%(id_post)s->%(var)s[post_rank[i]] += 1.0 / nb_aff_synapse[i];
                }
            }

            // axonal only
            %(axon_loop)s
        } // active
""" % { 'id_post': self.post.id,
        'target': self.target,
        'var': self._variable,
        'spiked_array_fusion': pre_array_fusion,
        'pre_array': pre_array,
        'psp_rside': psp_rside,
        'axon_loop': axon_code
    }

    def _generate_omp(self):
        """
        """
        # Sanity Check
        found = False
        for var in self.post.neuron_type.description['variables']:
            if var['name'] == self._variable:
                found = True
                break

        if not found:
            Messages._error("NormProjection: variable `"+self._variable+"` might be invalid. Please check the neuron model of population", self.post.name)

        # TODO: delays???
        if self.synapse_type.pre_axon_spike:
            pre_array = "tmp_spiked"
            pre_array_fusion = """
            std::vector<int> tmp_spiked = %(pre_array)s;
            tmp_spiked.insert( tmp_spiked.end(), pop%(id_pre)s->axonal.begin(), pop%(id_pre)s->axonal.end() );
""" % {'id_pre': self.pre.id, 'pre_array': 'pop'+str(self.pre.id)+'.spiked'}
        else:
            pre_array = 'pop'+str(self.pre.id)+'->spiked'
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

        # Get some more statements from user, normally done by the CodeGenerator
        if self._has_single_weight():
            psp_rside = "w"
        else:
            psp_rside = "w[i][j]" # default psp: g_target += w
        axon_code = ""
        indices = {
            'local_index': "[i][j]",
            'semiglobal_index': "[j]",
            'global_index': "[]"
        } # TODO: only true for openMP

        for var in self.synapse_type.description['pre_axon_spike']:
            if var['name'] == "g_target":
                psp_rside = var['cpp'].split("=")[1] % indices
            else:
                axon_code += var['cpp'] % indices

        # Only if needed. I don't really like the second loop, but it's for testing first
        if len(axon_code) > 0:
            axon_code = """
            for(int _idx_j = 0; _idx_j < pop%(id_pre)s->axonal.size(); _idx_j++){
                int rk_j = pop%(id_pre)s->axonal[_idx_j];
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

                    %(code)s
                }
            }
""" % {
    'id_pre': self.pre.id,
    'code': axon_code
}

        #
        # Generate Code Template Projection
        self._specific_template['psp_prefix'] = ""        
        self._specific_template['psp_code'] = """
        #pragma omp single
        {
            // Event-based summation
            if (_transmission && pop%(id_post)s->_active) {
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

                        pop%(id_post)s->g_%(target)s[post_rank[i]] += %(psp_rside)s;
                        pop%(id_post)s->%(var)s[post_rank[i]] += 1.0 / nb_aff_synapse[i];
                    }
                }

                // axonal only
                %(axon_loop)s
            } // active
        }
""" % { 'id_post': self.post.id,
        'target': self.target,
        'var': self._variable,
        'spiked_array_fusion': pre_array_fusion,
        'pre_array': pre_array,
        'psp_rside': psp_rside,
        'axon_loop': axon_code
    }

    def _generate_cuda(self):
        raise NotImplementedError("BOLD monitor is not available for CUDA devices yet.")

def _update_num_aff_connections(net_id=0, verbose=False):
    """
    We need to set for all NormProjections the number of afferent
    connections for each projection (across multiple projections,
    otherwise pre_rank[i].size() would be sufficient.

    Attention:

    This function should be used only internally!!! The function is
    called during Compiler.compile() call.
    """
    # Do we need to execute this procedure?
    need_to_execute = False
    for proj in Global.projections():
        if isinstance(proj, NormProjection):
            need_to_execute = True
            break

    if need_to_execute == False:
        return

    # iterate over all populations
    for pop in Global.populations(net_id):
        nb_synapses = np.zeros((pop.size))

        # get afferent projections of this layer
        aff_proj = Global.projections(net_id, pop.name)

        # we accumulate the number of synapses per dendrite
        # across all afferent projections of this layer.
        for i, proj in enumerate(aff_proj):
            # nb synapses per dendrite is oriented at the post_ranks list. If a neuron
            # does not receive any connection in THIS projection there is no entry
            nb_synapses_per_dend = np.array(proj.nb_synapses_per_dendrite)
            if len(nb_synapses_per_dend) == 0:
                continue

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

            nb_synapses_per_dend = np.array(proj.nb_synapses_per_dendrite)

            if verbose:
                print('before:', nb_synapses_per_dend)
            for idx, rank in enumerate(proj.post_ranks):
                nb_synapses_per_dend[idx] = nb_synapses[rank]

            if verbose:
                print('after:', nb_synapses_per_dend)
            proj.cyInstance.set_semiglobal_attribute_all("nb_aff_synapse", nb_synapses_per_dend, "double")
