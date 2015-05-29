header_template = """#ifndef __ANNARCHY_H__
#define __ANNARCHY_H__

#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <deque>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

/*
 * Structures for the populations
 *
 */
%(pop_struct)s
/*
 * Structures for the projections
 *
 */
%(proj_struct)s

/*
 * Internal data
 *
*/
extern double dt;
extern long int t;

/*
 * Declaration of the populations
 *
 */
%(pop_ptr)s

/*
 * Declaration of the projections
 *
 */
%(proj_ptr)s

/*
 * (De-)Flattening of LIL structures
 */
template<typename T>
std::vector<int> flattenIdx(std::vector<std::vector<T> > in)
{
    std::vector<T> flatIdx = std::vector<T>();
    typename std::vector<std::vector<T> >::iterator it;

    for ( it = in.begin(); it != in.end(); it++)
    {
        flatIdx.push_back(it->size());
    }

    return flatIdx;
}

template<typename T>
std::vector<int> flattenOff(std::vector<std::vector<T> > in)
{
    std::vector<T> flatOff = std::vector<T>();
    typename std::vector<std::vector<T> >::iterator it;

    int currOffset = 0;
    for ( it = in.begin(); it != in.end(); it++)
    {
        flatOff.push_back(currOffset);
        currOffset += it->size();
    }

    return flatOff;
}

template<typename T>
std::vector<T> flattenArray(std::vector<std::vector<T> > in)
{
    std::vector<T> flatVec = std::vector<T>();
    typename std::vector<std::vector<T> >::iterator it;

    for ( it = in.begin(); it != in.end(); it++)
    {
        flatVec.insert(flatVec.end(), it->begin(), it->end());
    }

    return flatVec;
}

template<typename T>
std::vector<std::vector<T> > deFlattenArray(std::vector<T> in, std::vector<int> idx)
{
    std::vector<std::vector<T> > deFlatVec = std::vector<std::vector<T> >();
    std::vector<int>::iterator it;

    int t=0;
    for ( it = idx.begin(); it != idx.end(); it++)
    {
        std::vector<T> tmp = std::vector<T>(in.begin()+t, in.begin()+t+*it);
        t += *it;

        deFlatVec.push_back(tmp);
    }

    return deFlatVec;
}

/*
 * Recorders
 *
 */
class Monitor
{
public:
    Monitor(std::vector<int> ranks, int period, long int offset){
        this->ranks = ranks;
        this->period = period;
        this->offset = offset;
        if(this->ranks.size() ==1 && this->ranks[0]==-1) // All neurons should be recorded
            this->partial = false;
        else
            this->partial = true;
    };

    virtual void record() {std::cout << "recording" << std::endl;};

    // Attributes
    bool partial;
    std::vector<int> ranks;
    int period;
    long int offset;

};
%(record_classes)s

extern std::vector<Monitor*> recorders;
void addRecorder(Monitor* recorder);
void removeRecorder(Monitor* recorder);

/*
 * Simulation methods
 *
*/

void initialize(double _dt, long seed) ;

void run(int nbSteps);

void step();


/*
 * Time export
 *
*/
long int getTime() ;
void setTime(long int t_) ;

double getDt() ;
void setDt(double dt_);

#endif
"""