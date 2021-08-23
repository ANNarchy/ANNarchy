#===============================================================================
#
#     HYB.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2021  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

#TODO: maybe it would make more sense to split up the nb_blocks and tpb for ELL and COO
init_launch_config = """
        // Generate the kernel launch configuration
        _threads_per_block = 64;
        _nb_blocks = static_cast<unsigned short int>( std::min<unsigned int>(nb_dendrites(), 65535) );
    
    #ifdef _DEBUG
        std::cout << "Kernel configuration: " << _nb_blocks << ", " << _threads_per_block << std::endl;
    #endif
"""

attribute_decl = {
    'local': """
    // Local %(attr_type)s %(name)s
    hyb_local<%(type)s> %(name)s;
    hyb_local_gpu<%(type)s> gpu_%(name)s;
    long int %(name)s_device_to_host;
    bool %(name)s_host_to_device;
""",
    'semiglobal': "",
    'global': ""
}

attribute_cpp_init = {
    'local': """
        // Local %(attr_type)s %(name)s
        %(name)s = init_matrix_variable<%(type)s>(static_cast<%(type)s>(%(init)s));
        gpu_%(name)s = init_matrix_variable_gpu<%(type)s>(%(name)s);
        %(name)s_host_to_device = true;
""",
    'semiglobal': "",
    'global': ""
}

attribute_cpp_size = {
    'local': "",
    'semiglobal': "",
    'global': ""
}

attribute_host_to_device = {
    'local': """
        // %(name)s: local
        if ( %(name)s_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( gpu_%(name)s.ell, %(name)s.ell.data(), %(name)s.ell.size() * sizeof( %(type)s ), cudaMemcpyHostToDevice);
            cudaMemcpy( gpu_%(name)s.coo, %(name)s.coo.data(), %(name)s.coo.size() * sizeof( %(type)s ), cudaMemcpyHostToDevice);
            %(name)s_host_to_device = false;
        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }
""",
    'semiglobal': "",
    'global': ""
}

attribute_device_to_host = {
    'local': "",
    'semiglobal': "",
    'global': ""
}

#
#   Synaptic delays
#
delay = {
    'uniform': {
        'declare': """
    // Uniform delay
    int delay ;""",
        'pyx_struct': """
        # Non-uniform delay
        int delay""",
        'init': "delay = delays[0][0];",
        'pyx_wrapper_init': """
        proj%(id_proj)s.delay = syn.uniform_delay""",
        'pyx_wrapper_accessor': """
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.delay
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.delay
    def set_delay(self, value):
        proj%(id_proj)s.delay = value
"""
    }
}

#
# Implement the continuous transmission for rate-code synapses.
#
rate_psp_kernel = {
    'body': {
        'sum': ""        
    },
    'header': "",
    'call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s._active && proj%(id_proj)s._transmission ) {
        // ELLPACK - partition
        int nb_dendrites = proj%(id_proj)s.nb_dendrites();
        int nb_blocks = static_cast<int>(ceil(static_cast<double>(nb_dendrites)/32.0));
        cu_proj%(id_proj)s_psp_ell_r<<< nb_blocks, 32>>>(
                       nb_dendrites,
                       /* ranks and offsets */
                       proj%(id_proj)s.get_ell()->gpu_post_ranks_, proj%(id_proj)s.get_ell()->gpu_col_idx_, proj%(id_proj)s.get_ell()->gpu_rl_
                       /* computation data */
                       %(add_args_ell)s
                       /* result */
                       %(target_arg)s );

        // Coordinate - partition
        size_t nb_synapses = proj%(id_proj)s.get_coo()->nb_synapses();
        nb_blocks = std::min(65535, int(ceil(double(nb_synapses)/double( proj%(id_proj)s._threads_per_block))));
        cu_proj%(id_proj)s_psp_coo<<< nb_blocks, proj%(id_proj)s._threads_per_block >>>(
                       nb_synapses,
                       /* connectivity */
                       proj%(id_proj)s.get_coo()->gpu_row_indices(), proj%(id_proj)s.get_coo()->gpu_column_indices()
                       /* other variables */
                       %(add_args_coo)s
                       /* result */
                       %(target_arg)s );

    #ifdef _DEBUG
        auto err = cudaGetLastError();
        if ( err != cudaSuccess ) {
            std::cout << "cu_proj%(id_proj)s_psp: " << cudaGetErrorString(err) << std::endl;
        }
    #endif
    }    
""",
    'thread_init': {
        'float': {
            'sum': "0.0f",
            'min': "FLT_MAX",
            'max': "FLT_MIN",
            'mean': "0.0f"
        },
        'double': {
            'sum': "0.0",
            'min': "DBL_MAX",
            'max': "DBL_MIN",
            'mean': "0.0"
        }
    }    
}

synapse_update = {}

conn_templates = {
    # launch config
    'launch_config': init_launch_config,

    # accessors
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'host_to_device': attribute_host_to_device,
    'device_to_host': attribute_device_to_host,
    'delay': delay,

    # operations
    'rate_psp': rate_psp_kernel,
    'synapse_update': synapse_update
}