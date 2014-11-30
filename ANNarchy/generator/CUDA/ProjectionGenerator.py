import ANNarchy.core.Global as Global

import numpy as np

class ProjectionGenerator(object):
    
    def header_struct(self, proj):    
        # Is it a specific projection?
        if proj.generator['cuda']['header_proj_struct']:
            return proj.generator['cuda']['header_proj_struct']

        code = """
struct ProjStruct%(id)s{
    int size;

    // stream
    cudaStream_t stream;
    
    // Learning flag
    bool _learning;
    // Connectivity
    std::vector<int> post_rank ;
    int* gpu_post_rank;
    std::vector< std::vector< int > > pre_rank ;
    int* gpu_pre_rank;
    int* gpu_nb_synapses;
    int* gpu_off_synapses;
    
    // flat connectivity parameters 
    int overallSynapses;
    std::vector<int> flat_idx;
    std::vector<int> flat_off;
"""

        # Delays
        if proj.max_delay > 1 and proj._synapses.uniform_delay == -1:
            if proj._synapses.type == "rate":
                Global._error("only uniform delays are supported ...")
                exit(0)
            
        # Parameters
        for var in proj.synapse.description['parameters']:
            if var['name'] in proj.synapse.description['local']:
                code += """
    // Local parameter %(name)s
    std::vector< std::vector< %(type)s > > %(name)s;    // host
    %(type)s* gpu_%(name)s;    // device
    bool %(name)s_dirty;    
""" % {'type' : var['ctype'], 'name': var['name']}
            elif var['name'] in proj.synapse.description['global']:
                code += """
    // Global parameter %(name)s
    std::vector<%(type)s>  %(name)s;
    %(type)s* gpu_%(name)s;    // device
    bool %(name)s_dirty;
""" % {'type' : var['ctype'], 'name': var['name']}

        # Variables
        for var in proj.synapse.description['variables']:
            if var['name'] in proj.synapse.description['local']:
                code += """
    // Local variable %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;    // host
    %(type)s* gpu_%(name)s;    // device
    bool %(name)s_dirty;
    // std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    // bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}
            elif var['name'] in proj.synapse.description['global']:
                code += """
    // Global variable %(name)s
    std::vector<%(type)s>  %(name)s;    // host
    %(type)s* gpu_%(name)s;    // device
    bool %(name)s_dirty;
    // std::vector< %(type)s > recorded_%(name)s ;
    // bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

        # Pre- or post_spike variables (including w)
        if proj.synapse.description['type'] == 'spike':
            for var in proj.synapse.description['pre_spike']:
                if not var['name'] in proj.synapse.description['attributes'] + ['g_target']:
                    code += """
    // Local variable %(name)s added by default
    std::vector< std::vector< %(type)s > > %(name)s ;
    // std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    // bool record_%(name)s ;
""" % {'type' : 'double', 'name': var['name']}

        code += """
};    
""" 
        return code % {'id': proj.id}
