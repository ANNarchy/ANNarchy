#include "cuda_check.h"

#include <stdio.h> 
#include <stdlib.h> // malloc
#include <cuda_runtime_api.h>

static cudaDeviceProp *prop = NULL;
static int nb_devices = 0;

void get_properties() {
    if(!prop) {
        cudaGetDeviceCount(&nb_devices);
        prop = (cudaDeviceProp*)malloc(sizeof(cudaDeviceProp)*nb_devices);

        for(int i = 0; i < nb_devices; i++) {
            cudaGetDeviceProperties(&prop[i], i);
            /*
            printf("Device Number: %d\n", i);
            printf("  Device name: %s\n", prop[i].name);
            printf("  Memory Clock Rate (KHz): %d\n", prop[i].memoryClockRate);
            printf("  Memory Bus Width (bits): %d\n", prop[i].memoryBusWidth);
            printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop[i].memoryClockRate*(prop[i].memoryBusWidth/8)/1.0e6);
            */
        }
    }
}

int num_devices() {
	get_properties();
	return nb_devices;
}

int get_major_version(int device) {
    get_properties();

    if ( device < nb_devices ) {
        return prop[device].major; 
    }

    return -1;
}

int get_minor_version(int device) {
    get_properties();

    if ( device < nb_devices ) {
        return prop[device].minor; 
    }

    return -1;
}

int get_runtime_version() {
	int version = 0;
	cudaRuntimeGetVersion(&version);
	return version;
}

int get_max_threads_per_block(int device) {
    get_properties();

    if ( device < nb_devices ) {
        return prop[device].maxThreadsPerBlock;
    }

    return -1;
}

int get_warp_size(int device) {
    get_properties();

    if ( device < nb_devices ) {
        return prop[device].warpSize;
    }

    return -1;
}