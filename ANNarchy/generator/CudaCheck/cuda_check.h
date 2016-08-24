#ifndef __ANNARCHY_CUDA_CHECK_H__
#define __ANNARCHY_CUDA_CHECK_H__

extern int num_devices();

extern int get_major_version(int);
extern int get_minor_version(int);
extern int get_runtime_version();

extern int get_max_threads_per_block(int);
extern int get_warp_size(int);

#endif
