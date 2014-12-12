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
    rc=0;
    std::cout << "Profile " << omp_get_max_threads() << std::endl;
    """,
    'run_pre': """// before
    """,
    'run_post': """// after
    std::cout << "Profile " << omp_get_max_threads() << std::endl;
    profiler->evaluate(1,1);
    """,
    #
    # Operations
    'compute_psp': {
            'before' : """    profiler->start_CPU_time_prof( (omp_get_max_threads()-1) * %(num_ops)s + %(off)s );""",
            'after' : """    profiler->stop_CPU_time_prof( (omp_get_max_threads()-1) * %(num_ops)s + %(off)s );
    rc++;"""
    },
    'update_neuron': {
            'before' : """        profiler->start_CPU_time_prof( (omp_get_max_threads()-1) * %(num_ops)s + %(off)s );""",
            'after' : """        profiler->stop_CPU_time_prof( (omp_get_max_threads()-1) * %(num_ops)s + %(off)s );
        rc++;"""
    }
}
