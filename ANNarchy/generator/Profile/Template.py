profile_generator_omp_template = {
    'include': """//Profiling
#include "Profiling.h"
Profiling* profiler;
int rc;
""",
    'init': """
    //initialize profiler
    profiler = new Profiling();
    profiler->init(); // call with default arguments
    std::cout << "Profile " << omp_get_max_threads() << std::endl;
    """,
    'step_pre': """  rc=0;
    """,
    'run_pre': """// before
    profiler->start_overall_time_prof(omp_get_max_threads()-1);
    """,
    'run_post': """// after
    std::cout << "Profile " << omp_get_max_threads() << std::endl;
    profiler->stop_overall_time_prof(omp_get_max_threads()-1);
    profiler->evaluate(0,1);
    """,
    #
    # Operations
    'compute_psp': {
            'before' : """profiler->start_CPU_time_prof( (omp_get_max_threads()-1) * %(num_ops)s + rc );
        profiler->start_total_cache_miss( (omp_get_max_threads()-1) * %(num_ops)s + rc );
    """,
            'after' : """profiler->stop_total_cache_miss( (omp_get_max_threads()-1) * %(num_ops)s + rc );
        profiler->stop_CPU_time_prof( (omp_get_max_threads()-1) * %(num_ops)s + rc );
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
