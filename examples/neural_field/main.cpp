#include "annarchy/build/ANNarchy.h"
#include <omp.h>

int main(int argc, char* argv[]) {

	ANNarchy* ann = new ANNarchy();	

	for (int nt=1; nt < 80; nt*=2) {
		omp_set_num_threads(nt);

		printf("OMP_THREADS=%i\n", omp_get_max_threads());
 
		double time=0.0;
	        for(int i= 0; i<10; i++){
			double start = omp_get_wtime();
			ann->run(1000);
			double stop = omp_get_wtime();
			time+= stop-start;
		}

		printf("Elapsed time: %0.2f ms\n", (time/10)*1000);
	}

	delete ann;
	return 0;
}
