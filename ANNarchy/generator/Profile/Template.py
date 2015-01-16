profile_generator_omp_template = {
    'include': """//Profiling
#include "Profiling.h"
Profiling* profiler;
int rc;
""",
    'init': """
    //initialize profiler
    profiler = new Profiling();
    profiler->init(0);
    std::cout << "Profile " << omp_get_max_threads() << std::endl;
    """,
    'run_pre': """  rc=0;
    """,
    'run_post': """// after
    std::cout << "Profile " << omp_get_max_threads() << std::endl;
    profiler->evaluate(1,1);
    """,
    #
    # Operations
    'compute_psp': {
            'before' : """profiler->start_CPU_time_prof( (omp_get_max_threads()-1) * %(num_ops)s + rc );""",
            'after' : """profiler->stop_CPU_time_prof( (omp_get_max_threads()-1) * %(num_ops)s + rc );
        rc++;"""
    },
    'update_synapse': {
            'before' : """profiler->start_CPU_time_prof( (omp_get_max_threads()-1) * %(num_ops)s + rc );""",
            'after' : """profiler->stop_CPU_time_prof( (omp_get_max_threads()-1) * %(num_ops)s + rc );
        rc++;"""
    },
    'update_neuron': {
            'before' : """profiler->start_CPU_time_prof( (omp_get_max_threads()-1) * %(num_ops)s + rc );""",
            'after' : """profiler->stop_CPU_time_prof( (omp_get_max_threads()-1) * %(num_ops)s + rc );
        rc++;"""
    }
}
