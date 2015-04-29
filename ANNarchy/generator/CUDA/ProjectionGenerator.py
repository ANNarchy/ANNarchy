"""

    ProjectionGenerator.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import ANNarchy.core.Global as Global

class ProjectionGenerator(object):

    def __init__(self):
        if Global.config['profiling']:
            from ..Profile.ProfileGenerator import ProfileGenerator
            self._prof_gen = ProfileGenerator(Global._populations, Global._projections)
        else:
            self._prof_gen = None

#######################################################################
############## HEADER #################################################
#######################################################################
    def header_struct(self, proj):
        # Is it a specific projection?
        if proj.generator['cuda']['header_proj_struct']:
            return proj.generator['cuda']['header_proj_struct']

        code = """
// %(pre_name)s -> %(post_name)s
struct ProjStruct%(id_proj)s{
    // number of dendrites
    int size;

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

    // stream
    cudaStream_t stream;
"""

        # Spiking neurons have aditional data
        if proj.synapse.type == 'spike':
            Global._error("No GPU implementation ...")
            exit(0)

        # Delays
        if proj.max_delay > 1 and proj._synapses.uniform_delay == -1:
            Global._error("only uniform delays are supported ...")
            exit(0)

        # Arrays for the random numbers
        code += """
    // cudaRNG states, per dendrite
"""
        for rd in proj.synapse.description['random_distributions']:
            code += """    curandState* gpu_%(rd_name)s;
""" % { 'rd_name' : rd['name'] }

        # Parameters
        for var in proj.synapse.description['parameters']:
            if var['name'] in proj.synapse.description['local']:
                code += """
    // Local parameter %(name)s
    std::vector< std::vector< %(type)s > > %(name)s;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;    
""" % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in proj.synapse.description['global']:
                code += """
    // Global parameter %(name)s
    std::vector<%(type)s>  %(name)s;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
""" % {'type' : var['ctype'], 'name': var['name']}

        # Variables
        for var in proj.synapse.description['variables']:
            if var['name'] in proj.synapse.description['local']:
                code += """
    // Local variable %(name)s
    std::vector< std::vector< %(type)s > > %(name)s;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
""" % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in proj.synapse.description['global']:
                code += """
    // Global variable %(name)s
    std::vector<%(type)s>  %(name)s;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
""" % {'type' : var['ctype'], 'name': var['name']}

        # Local functions
        if len(proj.synapse.description['functions'])>0:
            code += """
    // Local functions
"""
            for func in proj.synapse.description['functions']:
                code += ' '*4 + func['cpp'] + '\n'

        # Structural plasticity
        if Global.config['structural_plasticity']:
            code += self.header_structural_plasticity(proj)

        # Finish the structure
        code += """
};    
""" 
        return code % {'id_proj': proj.id, 'pre_name': proj.pre.name, 'post_name': proj.post.name}

    def recorder_class(self, proj):
        tpl_code = """
class ProjRecorder%(id)s : public Monitor
{
public:
    ProjRecorder%(id)s(std::vector<int> ranks, int period, long int offset)
        : Monitor(ranks, period, offset)
    {
%(init_code)s
    };
    virtual void record() {
%(recording_code)s
    };
%(struct_code)s
};
""" 
        init_code = ""
        recording_code = ""
        struct_code = ""

        for var in proj.synapse.description['variables']:
            if var['name'] in proj.synapse.description['local']:
                struct_code += """
    // Local variable %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;
    bool record_%(name)s ; """ % {'type' : var['ctype'], 'name': var['name']}
                init_code += """
        this->%(name)s = std::vector< std::vector< %(type)s > >();
        this->record_%(name)s = false; """ % {'type' : var['ctype'], 'name': var['name']}
                recording_code += """
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 )){
            this->%(name)s.push_back(proj%(id)s.%(name)s[this->ranks[0]]);
        }""" % {'id': proj.id, 'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in proj.synapse.description['global']:
                struct_code += """
    // Global variable %(name)s
    std::vector< %(type)s > %(name)s ;
    bool record_%(name)s ; """ % {'type' : var['ctype'], 'name': var['name']}
                init_code += """
        this->%(name)s = std::vector< %(type)s >();
        this->record_%(name)s = false; """ % {'type' : var['ctype'], 'name': var['name']}
                recording_code += """
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 )){
            // Do something
        } """ % {'id': proj.id, 'type' : var['ctype'], 'name': var['name']}
        
        return tpl_code % {'id': proj.id, 'init_code': init_code, 'recording_code': recording_code, 'struct_code': struct_code}

#######################################################################
############## BODY ###################################################
#######################################################################


    def computesum_rate(self, proj):
        """
        returns all data needed for compute postsynaptic sum kernels:

        header:  kernel prototypes
        body:    kernel implementation
        call:    kernel call
        """
        code = ""

        psp = proj.synapse.description['psp']['cpp'] % {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id}
        psp = psp.replace('rk_pre', 'rank_pre[j]')

        from .cuBodyTemplate import psp_kernel
        body_code = psp_kernel % { 'id': proj.id,
                                   'pre': proj.pre.id,
                                   'post': proj.post.id,
                                   'target': proj.target,
                                   'psp': psp
                                  }

        header_code = """__global__ void cuPop%(pre)s_Pop%(post)s_%(target)s_psp( int* pre_rank, int* nb_synapses, int *offsets, double *pre_r, double* w, double *sum_%(target)s );
""" % { 'pre': proj.pre.id,
        'post': proj.post.id,
        'target': proj.target,
      }

        from .cuBodyTemplate import psp_kernel_call
        call_code = psp_kernel_call % { 'id': proj.id,
                                        'pre': proj.pre.id,
                                        'post': proj.post.id,
                                        'target': proj.target,
                                      }

        # Take delays into account if any
        if proj.max_delay > 1:
            if proj.uniform_delay == -1:
                Global._error("only uniform delays are supported on GPUs.")
                exit(0)
            else:
                call_code = call_code.replace("gpu_r", "gpu_delayed_r["+str(proj.max_delay-1)+"]")

        return body_code, header_code, call_code

    def computesum_spiking(self, proj):
        # Is it a specific projection?
        if proj.generator['cuda']['body_compute_psp']:
            Global._error("Customized spiking projections are not usable on CUDA yet.")
            return "", "", ""

        Global._error("Spiking models are not supported currently on CUDA devices.")
        return "", "", ""

    def postevent(self, proj):
        code = ""
        Global._error("Spiking models are not supported currently on CUDA devices.")

        return ""

    def update_synapse(self, proj):
        from ..Utils import generate_equation_code

        # Global variables
        global_eq = generate_equation_code(proj.id, proj.synapse.description, 'global', 'proj') %{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id}

        # Local variables
        local_eq =  generate_equation_code(proj.id, proj.synapse.description, 'local', 'proj') %{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id}  

        if global_eq.strip() == '' and local_eq.strip() == '':
            return "", "", ""

        # pre- and postsynaptic global operations
        pre_global_ops = []
        for pre_glob in proj.synapse.description['pre_global_operations']:
            pre_global_ops.append( """_%(func)s_%(name)s""" % { 'func': pre_glob['function'], 'name': pre_glob['variable'] } )

        post_global_ops = []
        for post_glob in proj.synapse.description['post_global_operations']:
            post_global_ops.append( """_%(func)s_%(name)s""" % { 'func': post_glob['function'], 'name': post_glob['variable'] } )

        # remove doubled entries
        pre_dependencies = list(set(proj.synapse.description['dependencies']['pre']))
        pre_global_ops = list(set(pre_global_ops))
        post_dependencies = list(set(proj.synapse.description['dependencies']['post']))
        post_global_ops = list(set(post_global_ops))

        var = ""
        par = ""
        
        # synaptic variables / parameters
        for attr in proj.synapse.description['variables'] + proj.synapse.description['parameters']:
            var += """, %(type)s* %(name)s """ % { 'type': attr['ctype'], 'name': attr['name'] }

        # replace pre- and postsynaptic global operations / variable accesses
        for pre_var in pre_dependencies:
            var += """, double* pre_%(name)s""" % { 'id': proj.pre.id, 'name': pre_var}
        for g_op in pre_global_ops:
            par += """, double pre_%(name)s""" % { 'id': proj.pre.id, 'name': g_op}
        for post_var in post_dependencies:
            var += """, double* post_%(name)s""" % { 'id': proj.post.id, 'name': post_var}
        for g_op in post_global_ops:
            old = """, double post_%(name)s""" % { 'id': proj.post.id, 'name': g_op}
             
        # random variables
        for rd in proj.synapse.description['random_distributions']:
            par += """, curandState* %(rd_name)s""" % { 'rd_name' : rd['name'] }

        # we replace the rand_%(id)s by the corresponding curand... term
        for rd in proj.synapse.description['random_distributions']:
            if rd['dist'] == "Uniform":
                term = """curand_uniform_double( &%(rd)s[i]) * (%(max)s - %(min)s) + %(min)s""" % { 'rd': rd['name'], 'min': rd['args'].split(',')[0], 'max': rd['args'].split(',')[1] };
                local_eq = local_eq.replace(rd['name']+"[j]", term)
            elif rd['dist'] == "Normal":
                term = """curand_normal_double( &%(rd)s[i])""" % { 'rd': rd['name'] };
                local_eq = local_eq.replace(rd['name']+"[j]", term)
            elif rd['dist'] == "LogNormal":
                term = """curand_log_normal_double( &%(rd)s[i], %(mean)s, %(std_dev)s)""" % { 'rd': rd['name'], 'mean': rd['args'].split(',')[0], 'std_dev': rd['args'].split(',')[1] };
                local_eq = local_eq.replace(rd['name']+"[j]", term)
            else:
                Global._error("Unsupported random distribution on GPUs: " + rd['dist'])

        # remove all types
        repl_types = ["double*", "float*", "int*", "curandState*", "double", "float", "int"]
        var_wo_types = var
        par_wo_types = par
        for type in repl_types:
            var_wo_types = var_wo_types.replace(type, "")
            par_wo_types = par_wo_types.replace(type, "")

        from .cuBodyTemplate import syn_kernel
        body = syn_kernel % { 'id': proj.id,
                               'par': par,
                               'par2': par_wo_types,
                               'var': var,
                               'var2': var_wo_types,
                               'global_eqs': global_eq,
                               'local_eqs': local_eq,
                               'target': proj.target,
                               'pre': proj.pre.id,
                               'post': proj.post.id,
                             }

        header = """__global__ void cuProj%(id)s_step(int* post_rank, int *pre_rank, int *offsets, int *nb_synapses, double dt%(var)s%(par)s);
""" % { 'id': proj.id,
        'var': var,
        'par': par
      }
        
        #
        # calling entity
        local = ""
        for attr in proj.synapse.description['variables'] + proj.synapse.description['parameters']:
            local += """, proj%(id)s.gpu_%(name)s """ % { 'id': proj.id, 'name': attr['name'] }
            
        for pre_var in pre_dependencies:
            local += """, pop%(id)s.gpu_%(name)s """ % { 'id': proj.pre.id, 'name': pre_var }
        for post_var in post_dependencies:
            local += """, pop%(id)s.gpu_%(name)s """ % { 'id': proj.post.id, 'name': post_var }

        glob = ""
        for g_op in pre_global_ops:
            glob += """, pop%(id)s.%(name)s """ % { 'id': proj.pre.id, 'name': g_op }
        for g_op in post_global_ops:
            glob += """, pop%(id)s.%(name)s """ % { 'id': proj.post.id, 'name': g_op }

        # random variables
        for rd in proj.synapse.description['random_distributions']:
            glob += """, proj%(id)s.gpu_%(rd_name)s""" % { 'id': proj.id, 'rd_name' : rd['name'] }

        # generate code
        from .cuBodyTemplate import syn_kernel_call
        call = syn_kernel_call % { 'id': proj.id,
                                   'post': proj.post.id,
                                   'pre': proj.pre.id,
                                   'local': local,
                                   'glob': glob,
                                   'target': proj.target
                                 }

        return body, header, call

    def pruning(self, proj):
        Global._error("Pruning is not implemented for CUDA ... ")
        return ""

    def init_random_distributions(self, proj):
        # Is it a specific projection?
        if proj.generator['omp']['body_random_dist_init']:
            return proj.generator['omp']['body_random_dist_init'] %{'id_proj': proj.id}

        code = ""
        for rd in proj.synapse.description['random_distributions']:
            code += """    cudaMalloc((void**)&proj%(id)s.gpu_%(rd_name)s, pop%(post)s.size * sizeof(curandState));
    init_curand_states( pop%(post)s.size, proj%(id)s.gpu_%(rd_name)s, seed );
""" % {'id': proj.id, 'post': proj.post.id, 'rd_name': rd['name'] }

        return code

    def init_projection(self, proj):

        # Is it a specific projection?
        if proj.generator['cuda']['body_proj_init']:
            return proj.generator['cuda']['body_proj_init']

        # Learning by default
        code = """
    /////////////////////////////////////////
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s
    /////////////////////////////////////////
    proj%(id_proj)s._learning = true;
""" % { 'id_proj': proj.id, 'target': proj.target,
        'id_post': proj.post.id, 'id_pre': proj.pre.id,
        'name_post': proj.post.name, 'name_pre': proj.pre.name}

        # Initialize parameters
        for var in proj.synapse.description['parameters']:
            if var['name'] == 'w':
                continue
            if var['name'] in proj.synapse.description['local']:
                init = 0.0 if var['ctype'] == 'double' else 0
                code += """
    // Local parameter %(name)s
    proj%(id)s.%(name)s = std::vector< std::vector<%(type)s> >(proj%(id)s.post_rank.size(), std::vector<%(type)s>());
    proj%(id)s.%(name)s_dirty = true;
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
            else:
                init = 0.0 if var['ctype'] == 'double' else 0
                code += """
    // Global parameter %(name)s
    proj%(id)s.%(name)s = std::vector<%(type)s>(proj%(id)s.post_rank.size(), %(init)s);
    proj%(id)s.%(name)s_dirty = true;
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

        # Initialize variables
        for var in proj.synapse.description['variables']:
            if var['name'] == 'w':
                continue
            if var['name'] in proj.synapse.description['local']:
                init = 0.0 if var['ctype'] == 'double' else 0
                code += """
    // Local variable %(name)s
    proj%(id)s.%(name)s = std::vector< std::vector<%(type)s> >(proj%(id)s.post_rank.size(), std::vector<%(type)s>());
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
            else:
                init = 0.0 if var['ctype'] == 'double' else 0
                code += """
    // Global variable %(name)s
    proj%(id)s.%(name)s = std::vector<%(type)s>(proj%(id)s.post_rank.size(), %(init)s);
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

        # Spiking neurons have aditional data
        if proj.synapse.type == 'spike':
            Global._error("Spiking models are not supported on GPUs yet ...")
            exit(0)

            debug = """
// For debug...
    for (std::map< int, std::vector< std::pair<int, int> > >::iterator it=proj%(id_proj)s.inv_rank.begin(); it!=proj%(id_proj)s.inv_rank.end(); ++it) {
        std::cout << it->first << ": " ;
        for(int _id=0; _id<it->second.size(); _id++){
            std::pair<int, int> val = it->second[_id];
            std::cout << "(" << val.first << ", " << val.second << "), " ;
        }
        std::cout << std::endl ;
    }
"""
        # Pruning
        if Global.config['structural_plasticity']:
            Global._error("Structural plasticity is not supported on GPUs yet ...")
            exit(0)

        return code


#######################################################################
############## PYX ####################################################
#######################################################################

    def pyx_struct(self, proj):
        # Is it a specific projection?
        if proj.generator['cuda']['pyx_proj_struct']:
            return proj.generator['cuda']['pyx_proj_struct'] % {'id_proj': proj.id}

        code = """
    cdef struct ProjStruct%(id_proj)s :
        int size
        bool _learning
        vector[int] post_rank
        vector[vector[int]] pre_rank
"""         

        # Exact integration
        has_exact = False
        for var in proj.synapse.description['variables']:
            if var['method'] == 'exact':
                has_exact = True
        if has_exact:
            code += """
        vector[vector[long]] _last_event
"""% {'id': proj.id}

        # Delays
        if proj.max_delay > 1 and proj.uniform_delay == -1:
            code +="""
        vector[vector[int]] delay
"""
        # Parameters
        for var in proj.synapse.description['parameters']:
            if var['name'] in proj.synapse.description['local']:
                code += """
        # Local parameter %(name)s
        vector[vector[%(type)s]] %(name)s
        bool %(name)s_dirty
""" % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in proj.synapse.description['global']:
                code += """
        # Global parameter %(name)s
        vector[%(type)s]  %(name)s
        bool %(name)s_dirty 
""" % {'type' : var['ctype'], 'name': var['name']}

        # Variables
        for var in proj.synapse.description['variables']:
            if var['name'] in proj.synapse.description['local']:
                code += """
        # Local variable %(name)s
        vector[vector[%(type)s]] %(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in proj.synapse.description['global']:
                code += """
        # Global variable %(name)s
        vector[%(type)s]  %(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}

        # Structural plasticity
        if Global.config['structural_plasticity']:
            Global._error("Structural plasticity is not supported yet on GPUs")

        # Finalize the code
        return code % {'id_proj': proj.id}

    def pyx_wrapper(self, proj):

        # Is it a specific projection?
        if proj.generator['cuda']['pyx_proj_class']:
            return proj.generator['cuda']['pyx_proj_class'] % { 'proj_id': proj.id }

        # Init
        code = """
cdef class proj%(id)s_wrapper :

    def __cinit__(self, synapses):

        cdef CSR syn = synapses
        cdef int size = syn.size
        cdef int nb_post = syn.post_rank.size()

        proj%(id)s.size = size
        proj%(id)s.post_rank = syn.post_rank
        proj%(id)s.pre_rank = syn.pre_rank
        proj%(id)s.w = syn.w
"""% {'id': proj.id}

        # Exact integration
        has_exact = False
        for var in proj.synapse.description['variables']:
            if var['method'] == 'exact':
                has_exact = True
        if has_exact:
            code += """
        proj%(id)s._last_event = vector[vector[long]](nb_post, vector[long]())
        for n in range(nb_post):
            proj%(id)s._last_event[n] = vector[long](proj%(id)s.pre_rank[n].size(), -10000)
"""% {'id': proj.id}

        # Delays
        if proj.max_delay > 1 and proj.uniform_delay == -1:
            code +="""
        proj%(id)s.delay = syn.delay
"""% {'id': proj.id}

        # Size property
        code += """

    property size:
        def __get__(self):
            return proj%(id)s.size

    def nb_synapses(self, int n):
        return proj%(id)s.pre_rank[n].size()

    def _set_learning(self, bool l):
        proj%(id)s._learning = l

    def post_rank(self):
        return proj%(id)s.post_rank
    def pre_rank(self, int n):
        return proj%(id)s.pre_rank[n]
""" % {'id': proj.id}

        # Delays
        if proj.max_delay > 1 and proj.uniform_delay == -1:
            code +="""
    def get_delay(self):
        return proj%(id)s.delay
    def set_delay(self, value):
        proj%(id)s.delay = value
"""% {'id': proj.id}

        # Parameters
        for var in proj.synapse.description['parameters']:
            if var['name'] in proj.synapse.description['local']:
                code += """
    # Local parameter %(name)s
    def get_%(name)s(self):
        return proj%(id)s.%(name)s
    def set_%(name)s(self, value):
        proj%(id)s.%(name)s = value
        proj%(id)s.%(name)s_dirty = True
    def get_dendrite_%(name)s(self, int rank):
        return proj%(id)s.%(name)s[rank]
    def set_dendrite_%(name)s(self, int rank, vector[%(type)s] value):
        proj%(id)s.%(name)s[rank] = value
        proj%(id)s.%(name)s_dirty = True
    def get_synapse_%(name)s(self, int rank_post, int rank_pre):
        return proj%(id)s.%(name)s[rank_post][rank_pre]
    def set_synapse_%(name)s(self, int rank_post, int rank_pre, %(type)s value):
        proj%(id)s.%(name)s[rank_post][rank_pre] = value
""" % {'id' : proj.id, 'name': var['name'], 'type': var['ctype']}

            elif var['name'] in proj.synapse.description['global']:
                code += """
    # Global parameter %(name)s
    def get_%(name)s(self):
        return proj%(id)s.%(name)s
    def set_%(name)s(self, value):
        proj%(id)s.%(name)s = value
    def get_dendrite_%(name)s(self, int rank):
        return proj%(id)s.%(name)s[rank]
    def set_dendrite_%(name)s(self, int rank, %(type)s value):
        proj%(id)s.%(name)s[rank] = value
""" % {'id' : proj.id, 'name': var['name'], 'type': var['ctype']}

        # Variables
        for var in proj.synapse.description['variables']:
            if var['name'] in proj.synapse.description['local']:
                code += """
    # Local variable %(name)s
    def get_%(name)s(self):
        return proj%(id)s.%(name)s
    def set_%(name)s(self, value):
        proj%(id)s.%(name)s = value
        proj%(id)s.%(name)s_dirty = True
    def get_dendrite_%(name)s(self, int rank):
        return proj%(id)s.%(name)s[rank]
    def set_dendrite_%(name)s(self, int rank, vector[%(type)s] value):
        proj%(id)s.%(name)s[rank] = value
        proj%(id)s.%(name)s_dirty = True        
    def get_synapse_%(name)s(self, int rank_post, int rank_pre):
        return proj%(id)s.%(name)s[rank_post][rank_pre]
    def set_synapse_%(name)s(self, int rank_post, int rank_pre, %(type)s value):
        proj%(id)s.%(name)s[rank_post][rank_pre] = value
        proj%(id)s.%(name)s_dirty = True        
""" % {'id' : proj.id, 'name': var['name'], 'type': var['ctype']}

            elif var['name'] in proj.synapse.description['global']:
                code += """
    # Global variable %(name)s
    def get_%(name)s(self):
        return proj%(id)s.%(name)s
    def set_%(name)s(self, value):
        proj%(id)s.%(name)s = value
    def get_dendrite_%(name)s(self, int rank):
        return proj%(id)s.%(name)s[rank]
    def set_dendrite_%(name)s(self, int rank, %(type)s value):
        proj%(id)s.%(name)s[rank] = value
""" % {'id' : proj.id, 'name': var['name'], 'type': var['ctype']}

         # Structural plasticity
        if Global.config['structural_plasticity']:
            Global._error('structural plasticity is not supported yet on CUDA ...')

        return code

#######################################################################
############## Recording ##############################################
#######################################################################

    def pyx_monitor_struct(self, proj):
        tpl_code = """
    # Projection %(id)s : Monitor
    cdef cppclass ProjRecorder%(id)s (Monitor):
        ProjRecorder%(id)s(vector[int], int, long) except +    
"""
        for var in proj.synapse.description['variables']:
            if var['name'] in proj.synapse.description['local']:
                tpl_code += """
        vector[vector[%(type)s]] %(name)s
        bool record_%(name)s""" % {'name': var['name'], 'type': var['ctype']}
            elif var['name'] in proj.synapse.description['global']:
                tpl_code += """
        vector[%(type)s] %(name)s
        bool record_%(name)s""" % {'name': var['name'], 'type': var['ctype']}


        return tpl_code % {'id' : proj.id}

    def pyx_monitor_wrapper(self, proj):
        tpl_code = """
# Projection Monitor wrapper
cdef class ProjRecorder%(id)s_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, long offset):
        self.thisptr = new ProjRecorder%(id)s(ranks, period, offset)
"""

        for var in proj.synapse.description['variables']:
            tpl_code += """
    property %(name)s:
        def __get__(self): return (<ProjRecorder%(id)s *>self.thisptr).%(name)s
        def __set__(self, val): (<ProjRecorder%(id)s *>self.thisptr).%(name)s = val 
    property record_%(name)s:
        def __get__(self): return (<ProjRecorder%(id)s *>self.thisptr).record_%(name)s
        def __set__(self, val): (<ProjRecorder%(id)s *>self.thisptr).record_%(name)s = val 
    def clear_%(name)s(self):
        (<ProjRecorder%(id)s *>self.thisptr).%(name)s.clear()""" % {'id' : proj.id, 'name': var['name']}

        return tpl_code % {'id' : proj.id}