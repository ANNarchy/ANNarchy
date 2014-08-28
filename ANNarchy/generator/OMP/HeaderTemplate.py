header_template = """#ifndef __ANNARCHY_H__
#define __ANNARCHY_H__

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <omp.h>
#include <random>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>

/*
 * Built-in functions
 *
 */
#define positive(x) (x>0.0? x : 0.0)
#define negative(x) (x<0.0? x : 0.0)
#define clip(x, a, b) (x<a? a : (x>b? b :x))

/*
 * Structures for the populations
 *
 */
%(pop_struct)s
/*
 * Structures for the projections
 *
 */
struct BaseProjection{    
	std::vector<int> post_rank ;
    std::vector<int> pre_rank ;
    std::vector<int> delay ;
    std::vector<double> w ;	
};
BaseProjection flattenConnectivityMatrix(
    int size,
    std::vector<int> post, 
    std::vector< std::vector <int> > pre, 
    std::vector< std::vector <double> > w, 
    std::vector< std::vector <int> > delay );
%(proj_struct)s


/*
 * Internal data
 *
*/
extern double dt;
extern long int t;

// Random number generators
extern std::vector< std::mt19937 >  rng;
extern std::uniform_real_distribution<double> uniform;
extern std::normal_distribution<double> normal;


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
 * Simulation methods
 *
*/

void initialize(double _dt) ;

void run(int nbSteps);

void step();


/*
 * Time export
 *
*/
int getTime() ;
void setTime(int t_) ;

double getDt() ;
void setDt(double dt_);

/*
 * Number of threads
 *
*/
void setNumThreads(int threads);

#endif
"""