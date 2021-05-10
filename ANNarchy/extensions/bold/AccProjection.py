#===============================================================================
#
#     AccProjection.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2018-2019  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#     Oliver Maith
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
    def __init__(self, pre, post, target, variable, name=None, normalize_input=0, scale_factor=1.0, copied=False):
        # Instantiate the projection
        SpecificProjection.__init__(self, pre, post, target, None, name, copied)
        self._variable = variable
        self._scale_factor = scale_factor
        self._normalize_input = normalize_input

        # Check populations
        if not self.pre.neuron_type.type == 'spike':
            Global._error('The pre-synaptic population of an AccProjection must be spiking.')

        if not self.post.neuron_type.type == 'rate':
            Global._error('The post-synaptic population of an AccProjection must be rate-coded.')

        # Not on CUDA
        if Global._check_paradigm('cuda'):
            Global._error('AccProjections are not available on CUDA yet.')

        # Prevent automatic split of matrices
        self._no_split_matrix = True

    def _copy(self, pre, post):
        "Returns a copy of the population when creating networks. Internal use only."
        return AccProjection(pre=pre, post=post, target=self.target, variable=self._variable, name=self.name, normalize_input=self._normalize_input, scale_factor=self._scale_factor, copied=True)

    def _generate_st(self):
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

        single_ids = {'id_pre': self.pre.id,'var': self._variable}

        if self._normalize_input == 0:
            # Generate Code Template
            self._specific_template['psp_prefix'] = ""
            self._specific_template['psp_code'] = """
        for(int post_idx = 0; post_idx < post_rank.size(); post_idx++) {
        %(float_prec)s lsum = 0.0;

            for(auto it = pre_rank[post_idx].begin(); it != pre_rank[post_idx].end(); it++) {
                lsum += pop%(id_pre)s.%(var)s[*it];
            }

            pop%(id_post)s._sum_%(target)s[post_rank[post_idx]] += %(scale_factor)s * lsum/pre_rank[post_idx].size();
        }
""" % {
    'id_post': self.post.id,
    'id_pre': self.pre.id,
    'var': self._variable,
    'target': self.target,
    'scale_factor': self._scale_factor,
    'float_prec': Global.config['precision']
}

        else:
            # Generate Code Template
            self._specific_template['declare_additional'] = """
    std::vector< std::vector<%(float_prec)s> > baseline;
    long time_for_init_baseline;
    int init_baseline_period;
    void start(int baseline_period) {
        init_baseline_period=baseline_period;
        time_for_init_baseline = t + baseline_period;
    #ifdef _DEBUG
        std::cout << "ProjStruct%(id_proj)s: set new baseline period from step " << t << " to step " << time_for_init_baseline << std::endl;
    #endif
    }
""" % {'id_proj': self.id, 'float_prec': Global.config['precision']}
            self._specific_template['export_additional'] = """
        void start(int)
"""
            self._specific_template['wrapper_access_additional'] = """
    def start(self, baseline_period):
        proj%(id_proj)s.start(baseline_period)
""" % {'id_proj': self.id}

            self._specific_template['init_additional'] = """
        baseline = init_matrix_variable<%(float_prec)s>(static_cast<%(float_prec)s>(0.0));
        time_for_init_baseline = -1;
        init_baseline_period=1;
""" % {'float_prec': Global.config['precision']}

            self._specific_template['psp_prefix'] = ""
            self._specific_template['psp_code'] = """
        bool compute_baseline = (t < time_for_init_baseline) ? true : false;

        for(int post_idx = 0; post_idx < post_rank.size(); post_idx++) {
            %(float_prec)s lsum = 0.0;

            auto it = pre_rank[post_idx].begin();
            int j = 0;
            for(; it != pre_rank[post_idx].end(); it++, j++) {
                if(compute_baseline)
                    baseline[post_idx][j] += pop%(id_pre)s.%(var)s[*it]/static_cast<%(float_prec)s>(init_baseline_period);
                else
                    lsum += tanh( (pop%(id_pre)s.%(var)s[*it] - baseline[post_idx][j])/(baseline[post_idx][j] + 0.00000001) );
            }

            pop%(id_post)s._sum_%(target)s[post_rank[post_idx]] += %(scale_factor)s * lsum/pre_rank[post_idx].size();
        }
""" % {
    'id_post': self.post.id,
    'id_pre': self.pre.id,
    'var': self._variable,
    'target': self.target,
    'scale_factor': self._scale_factor,
    'float_prec': Global.config['precision']
}


    def _generate_omp(self):
        raise NotImplementedError("BOLD monitor is not available for openMP yet.")

    def generate_cuda(self):
        raise NotImplementedError("BOLD monitor is not available for CUDA devices yet.")