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