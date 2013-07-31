#ifndef __PROJECTION_H__
#define	__PROJECTION_H__

#include "Global.h"

class Projection{
public:
	Projection(Population* pre, Population* post, int post_rank, int target);

	Projection(int pre, int post, int post_rank, int target);

	int getWeightCount() { return (int)rank_.size(); }

	virtual void initValues(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay = std::vector<int>());

	virtual void computeSum();

	virtual void globalLearn() {}

	virtual void localLearn() {}

	class Population* getPrePopulation() { return pre_population_; }

	int getTarget() { return target_; }

	DATA_TYPE getSum() { return sum_; }

	std::vector<int> getRank() { return rank_; }

	std::vector<DATA_TYPE> getValue() { return value_; }
protected:
	Population* pre_population_;
	Population* post_population_;
	int post_neuron_rank_;
	int target_;

	std::vector<int> rank_;
	std::vector<int> delay_;
	std::vector<DATA_TYPE> value_;

	DATA_TYPE sum_;
	std::vector<DATA_TYPE>* pre_rates_;
	std::vector<DATA_TYPE>* post_rates_;

	DATA_TYPE tau_;
	DATA_TYPE dt_;

	bool constDelay_;
	int maxDelay_;
};
#endif
