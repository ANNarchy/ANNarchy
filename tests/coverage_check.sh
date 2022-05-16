coverage erase
coverage run --source /scratch/alexschw/annarchy_install/lib/python3.6/site-packages/ANNarchy/ -a test_single_thread.py
coverage run --source /scratch/alexschw/annarchy_install/lib/python3.6/site-packages/ANNarchy/ -a test_openmp.py
coverage run --source /scratch/alexschw/annarchy_install/lib/python3.6/site-packages/ANNarchy/ -a test_CUDA.py
coverage report
