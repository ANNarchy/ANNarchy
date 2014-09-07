cu_body_template=\
"""
#include "cuANNarchy.h"

%(kernel_config)s

/****************************************
 * population function kernels          *
 ****************************************/
%(pop_kernel)s
 
/****************************************
 * weighted sum kernels                 *
 ****************************************/
%(psp_kernel)s

/****************************************
 * update synapses kernel               *
 ****************************************/
%(syn_kernel)s

/****************************************
 * call kernels                         *
 ****************************************/
%(call_kernel)s
"""

pop_kernel=\
"""
__global__ void cu_%(name)_step(int N, %(arguments)s)
{
        int i = threadIdx.x + blockIdx.x * blockDim.x;  // neuron idx

        if (i < N)
        {
%(equation)s
        }
}
"""