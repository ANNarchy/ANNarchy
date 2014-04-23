#ifndef __SIMPLE_CU_TEST__
#define __SIMPLE_CU_TEST__

#include <vector>

void helloCuda();

DATA_TYPE weightedSum(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<DATA_TYPE> preRates);

#endif
