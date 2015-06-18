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
#include <random>
%(include_omp)s

/*
 * Built-in functions
 *
 */
#define positive(x) (x>0.0? x : 0.0)
#define negative(x) (x<0.0? x : 0.0)
#define clip(x, a, b) (x<a? a : (x>b? b :x))
#define modulo(a, b) long(a) %% long(b)

/*
 * Custom functions
 *
 */
%(custom_func)s

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
extern std::mt19937  rng;


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

void initialize(double _dt, long int seed) ;

void run(int nbSteps);

int run_until(int steps, std::vector<int> populations, bool or_and);

void step();


/*
 * Time export
 *
*/
long int getTime() ;
void setTime(long int t_) ;

double getDt() ;
void setDt(double dt_);

/*
 * Number of threads
 *
*/
void setNumberThreads(int threads);

#endif
"""