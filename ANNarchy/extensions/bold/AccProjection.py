#===============================================================================
#
#     AccProjection.py
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

class AccProjection(SpecificProjection):
    """
    Accumulates the values of a given variable.
    """
    def __init__(self, pre, post, target, variable, name=None, copied=False):
        # Instantiate the projection
        SpecificProjection.__init__(self, pre, post, target, None, name, copied)
        self._variable = variable

        # Check populations
        if not self.pre.neuron_type.type == 'spike':
            Global._error('The pre-synaptic population of an AccProjection must be spiking.')

        if not self.post.neuron_type.type == 'rate':
            Global._error('The post-synaptic population of an AccProjection must be rate-coded.')

        # Not on CUDA
        if Global._check_paradigm('cuda'):
            Global._error('AccProjections are not available on CUDA yet.')

    def _copy(self, pre, post):
        "Returns a copy of the population when creating networks. Internal use only."
        return AccProjection(pre=pre, post=post, target=self.target, variable=self._variable, name=self.name, copied=True)

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
            Global._warning("Variable might be invalid ...")

        # Generate Code Template
        self._specific_template['psp_prefix'] = ""
        self._specific_template['psp_code'] = """
        for(int post_idx = 0; post_idx < post_rank.size(); post_idx++) {
            %(float_prec)s lsum = 0.0;

            for(auto it = pre_rank[post_idx].begin(); it != pre_rank[post_idx].end(); it++) {
                lsum += pop%(id_pre)s.%(var)s[*it];
            }

            pop%(id_post)s._sum_%(target)s[post_rank[post_idx]] += lsum/pre_rank[post_idx].size();
        }
""" % {
    'id_post': self.post.id,
    'id_pre': self.pre.id,
    'var': self._variable,
    'target': self.target,
    'float_prec': Global.config['precision']
}
