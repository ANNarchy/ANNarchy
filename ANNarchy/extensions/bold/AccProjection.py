"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.SpecificProjection import SpecificProjection
from ANNarchy.intern.ConfigManagement import get_global_config

from ANNarchy.intern import Messages

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

        # Check population type of the receiving population
        if not self.post.neuron_type.type == 'rate':
            Messages._error('The post-synaptic population of an AccProjection must be rate-coded.')

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
            Messages._warning("Variable might be invalid ...")

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
    'float_prec': get_global_config('precision')
}

        else:
            # Generate Code Template
            self._specific_template['declare_additional'] = """
    std::vector<std::vector<%(float_prec)s>> baseline;
    std::vector<%(float_prec)s> baseline_mean;
    std::vector<%(float_prec)s> baseline_std;
    long time_for_init_baseline;
    int init_baseline_period;
    void start(int baseline_period) {
        init_baseline_period=baseline_period;
        time_for_init_baseline = t + baseline_period;
    #ifdef _DEBUG
        std::cout << "ProjStruct%(id_proj)s: set new baseline period from step " << t << " to step " << time_for_init_baseline << std::endl;
    #endif
    }
""" % {'id_proj': self.id, 'float_prec': get_global_config('precision')}
            self._specific_template['export_additional'] = """
        void start(int)
"""
            self._specific_template['wrapper_access_additional'] = """
    def start(self, baseline_period):
        proj%(id_proj)s.start(baseline_period)
""" % {'id_proj': self.id}

            self._specific_template['init_additional'] = """
        time_for_init_baseline = -1;
        init_baseline_period=1;
        baseline = std::vector<std::vector<%(float_prec)s>>(post_rank.size(), std::vector<%(float_prec)s>() );
        baseline_mean = std::vector<%(float_prec)s>(post_rank.size(), 0);
        baseline_std = std::vector<%(float_prec)s>(post_rank.size(), 1);
""" % {'float_prec': get_global_config('precision')}
            self._specific_template['clear_additional'] = """
        for(auto it = baseline.begin(); it != baseline.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        baseline.clear();
        baseline.shrink_to_fit();

        baseline_mean.clear();
        baseline_mean.shrink_to_fit();

        baseline_std.clear();
        baseline_std.shrink_to_fit();
"""
            self._specific_template['psp_prefix'] = ""
            self._specific_template['psp_code'] = """
        bool compute_baseline = (t < time_for_init_baseline) ? true : false;
        bool compute_average = (t == time_for_init_baseline) ? true : false;

        for(int post_idx = 0; post_idx < post_rank.size(); post_idx++) {
            %(float_prec)s lsum = 0.0;

            // accumulate the input variable
            auto it = pre_rank[post_idx].begin();
            int j = 0;
            for(; it != pre_rank[post_idx].end(); it++, j++) {
                lsum += pop%(id_pre)s.%(var)s[*it];
            }

            // we want to use the average across incoming connections
            // ( i. e. the recorded population )
            lsum /= static_cast<%(float_prec)s>(pre_rank[post_idx].size());

            // if the init time is over compute the mean/standard
            // deviation across time
            if (compute_average) {
                %(float_prec)s sum = std::accumulate(std::begin(baseline[post_idx]), std::end(baseline[post_idx]), static_cast<%(float_prec)s>(0.0));
                baseline_mean[post_idx] = sum / static_cast<%(float_prec)s>(baseline[post_idx].size());

                %(float_prec)s accum = 0.0;
                std::for_each (baseline[post_idx].begin(), baseline[post_idx].end(), [&](const %(float_prec)s value) {
                    accum += (value - baseline_mean[post_idx]) * (value - baseline_mean[post_idx]);
                });
                baseline_std[post_idx] = sqrt(accum / static_cast<%(float_prec)s>(baseline[post_idx].size()-1));
            }

            // until init time is reached we store the rescaled sum,
            // otherwise we use the baseline mean and standard deviation to rescale
            if (compute_baseline) {
                // enqueue the value to baseline vector
                baseline[post_idx].push_back(lsum);

                // don't store the result
                pop%(id_post)s._sum_%(target)s[post_rank[post_idx]] += 0.0;
            } else {
                // apply relative deviation normalization
                lsum = ((lsum - baseline_mean[post_idx]) / (std::abs(baseline_mean[post_idx]) + 0.0000001));

                // store the result
                pop%(id_post)s._sum_%(target)s[post_rank[post_idx]] += %(scale_factor)s * lsum;
            }
        }
""" % {
    'id_post': self.post.id,
    'id_pre': self.pre.id,
    'var': self._variable,
    'target': self.target,
    'scale_factor': self._scale_factor,
    'float_prec': get_global_config('precision')
}

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
            Messages._warning("Variable might be invalid ...")

        single_ids = {'id_pre': self.pre.id,'var': self._variable}

        if self._normalize_input == 0:
            # Generate Code Template
            self._specific_template['psp_prefix'] = ""
            self._specific_template['psp_code'] = """
        #pragma omp for
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
    'float_prec': get_global_config('precision')
}

        else:
            # Generate Code Template
            self._specific_template['declare_additional'] = """
    std::vector<std::vector<%(float_prec)s>> baseline;
    std::vector<%(float_prec)s> baseline_mean;
    std::vector<%(float_prec)s> baseline_std;
    long time_for_init_baseline;
    int init_baseline_period;
    void start(int baseline_period) {
        init_baseline_period=baseline_period;
        time_for_init_baseline = t + baseline_period;
    #ifdef _DEBUG
        std::cout << "ProjStruct%(id_proj)s: set new baseline period from step " << t << " to step " << time_for_init_baseline << std::endl;
    #endif
    }
""" % {'id_proj': self.id, 'float_prec': get_global_config('precision')}
            self._specific_template['export_additional'] = """
        void start(int)
"""
            self._specific_template['wrapper_access_additional'] = """
    def start(self, baseline_period):
        proj%(id_proj)s.start(baseline_period)
""" % {'id_proj': self.id}

            self._specific_template['init_additional'] = """
        time_for_init_baseline = -1;
        init_baseline_period=1;
        baseline = std::vector<std::vector<%(float_prec)s>>(post_rank.size(), std::vector<%(float_prec)s>() );
        baseline_mean = std::vector<%(float_prec)s>(post_rank.size(), 0);
        baseline_std = std::vector<%(float_prec)s>(post_rank.size(), 1);
""" % {'float_prec': get_global_config('precision')}

            self._specific_template['psp_prefix'] = ""
            self._specific_template['psp_code'] = """
        bool compute_baseline = (t < time_for_init_baseline) ? true : false;
        bool compute_average = (t == time_for_init_baseline) ? true : false;

        #pragma omp for
        for(int post_idx = 0; post_idx < post_rank.size(); post_idx++) {
            %(float_prec)s lsum = 0.0;

            // accumulate the input variable
            auto it = pre_rank[post_idx].begin();
            int j = 0;
            for(; it != pre_rank[post_idx].end(); it++, j++) {
                lsum += pop%(id_pre)s.%(var)s[*it];
            }

            // we want to use the average across incoming connections
            // ( i. e. the recorded population )
            lsum /= static_cast<%(float_prec)s>(pre_rank[post_idx].size());

            // if the init time is over compute the mean/standard
            // deviation across time
            if (compute_average) {
                %(float_prec)s sum = std::accumulate(std::begin(baseline[post_idx]), std::end(baseline[post_idx]), static_cast<%(float_prec)s>(0.0));
                baseline_mean[post_idx] = sum / static_cast<%(float_prec)s>(baseline[post_idx].size());

                %(float_prec)s accum = 0.0;
                std::for_each (baseline[post_idx].begin(), baseline[post_idx].end(), [&](const %(float_prec)s value) {
                    accum += (value - baseline_mean[post_idx]) * (value - baseline_mean[post_idx]);
                });
                baseline_std[post_idx] = sqrt(accum / static_cast<%(float_prec)s>(baseline[post_idx].size()-1));
            }

            // until init time is reached we store the rescaled sum,
            // otherwise we use the baseline mean and standard deviation to rescale
            if (compute_baseline) {
                // enqueue the value to baseline vector
                baseline[post_idx].push_back(lsum);

                // don't store the result
                pop%(id_post)s._sum_%(target)s[post_rank[post_idx]] += 0.0;
            } else {
                // apply relative deviation normalization
                lsum = ((lsum - baseline_mean[post_idx]) / (std::abs(baseline_mean[post_idx]) + 0.0000001));

                // store the result
                pop%(id_post)s._sum_%(target)s[post_rank[post_idx]] += %(scale_factor)s * lsum;
            }
        }
""" % {
    'id_post': self.post.id,
    'id_pre': self.pre.id,
    'var': self._variable,
    'target': self.target,
    'scale_factor': self._scale_factor,
    'float_prec': get_global_config('precision')
}

    def _generate_cuda(self):
        raise NotImplementedError("The AccProjection (part of the BOLD monitor) is not available for CUDA devices yet.")
