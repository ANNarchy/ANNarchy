body_template = '''
#include "ANNarchy.h"
#include "cuANNarchy.h"

#include <cuda_runtime_api.h>
#include <math.h>

/*
 * Internal data
 *
*/
double dt;
long int t;
std::vector< std::mt19937 >  rng;

// Populations
%(pop_ptr)s

// Projections
%(proj_ptr)s

template<typename T>
std::vector<int> flattenIdx(std::vector<std::vector<T> > in)
{
    std::vector<T> flatIdx = std::vector<T>();
    
    for ( auto it = in.begin(); it != in.end(); it++)
    {
        flatIdx.push_back(it->size());
    }
    
    return flatIdx;
}

template<typename T>
std::vector<int> flattenOff(std::vector<std::vector<T> > in)
{
    std::vector<T> flatOff = std::vector<T>();

    int t = 0;
    for ( auto it = in.begin(); it != in.end(); it++)
    {
        flatOff.push_back(t);
        t+= it->size();
    }

    return flatOff;
}

template<typename T> 
std::vector<T> flattenArray(std::vector<std::vector<T> > in) 
{
    std::vector<T> flatVec = std::vector<T>();

    for ( auto it = in.begin(); it != in.end(); it++)
    {
        flatVec.insert(flatVec.end(), it->begin(), it->end());
    }

    return flatVec;
}

template<typename T> 
std::vector<std::vector<T> > deFlattenArray(std::vector<T> in, std::vector<int> idx) 
{
    std::vector<std::vector<T> > deFlatVec = std::vector<std::vector<T> >();

    int t=0;
    for ( auto it = idx.begin(); it != idx.end(); it++)
    {
        auto tmp = std::vector<T>(in.begin()+t, in.begin()+t+*it);
        t += *it;

        deFlatVec.push_back(tmp);
    }

    return deFlatVec;
}

// Simulate the network for the given number of steps
void run(int nbSteps) {
    %(host_device_transfer)s

    // simulation loop
    for(int i=0; i<nbSteps; i++)
    {
        step();
    }

    %(device_host_transfer)s
}

// Initialize the internal data and random numbers generators
void initialize(double _dt) {

    dt = _dt;
    t = (long int)(0);

    int threads = std::max(1, omp_get_max_threads());
    for(int seed = 0; seed < threads; ++seed)
    {
        rng.push_back(std::mt19937(time(NULL)*seed));
    }
%(device_init)s
%(random_dist_init)s
%(delay_init)s
%(spike_init)s
}

// Step method. Generated by ANNarchy.
void step()
{
    int numThreads=0, numBlocks=0;
    
    ////////////////////////////////
    // Presynaptic events
    ////////////////////////////////
    double start, sum;
%(compute_sums)s

    ////////////////////////////////
    // Reset spikes
    ////////////////////////////////


    ////////////////////////////////
    // Update random distributions
    ////////////////////////////////
%(random_dist_update)s

    ////////////////////////////////
    // Update neural variables
    ////////////////////////////////
%(update_neuron)s    

    ////////////////////////////////
    // Delay outputs
    ////////////////////////////////
%(delay_code)s

    ////////////////////////////////
    // Global operations (min/max/mean)
    ////////////////////////////////


    ////////////////////////////////
    // Update synaptic variables
    ////////////////////////////////
%(update_synapse)s    


    ////////////////////////////////
    // Postsynaptic events
    ////////////////////////////////

    ////////////////////////////////
    // Recording
    ////////////////////////////////
%(record)s

    ////////////////////////////////
    // Increase internal time
    ////////////////////////////////
    t++;
}


/*
 * Access to time and dt
 *
*/
long int getTime() {return t;}
void setTime(long int t_) { t=t_;}
double getDt() { return dt;}
void setDt(double dt_) { dt=dt_;}

/*
 * Number of threads
 *
*/
void setNumThreads(int threads)
{
    omp_set_num_threads(threads);
}
'''