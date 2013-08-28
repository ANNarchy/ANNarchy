#ifndef __PROJECTION_H__
#define	__PROJECTION_H__

#include "Global.h"

class Projection{
public:
	Projection(Population* pre, Population* post, int post_rank, int target);

	Projection(int pre, int post, int post_rank, int target);

	int getSynapseCount() { return (int)rank_.size(); }

	virtual void initValues(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay = std::vector<int>());

	virtual void computeSum();

	virtual void globalLearn() {}

	virtual void localLearn() {}

	class Population* getPrePopulation() { return pre_population_; }

	int getTarget() { return target_; }

	DATA_TYPE getSum() { return sum_; }

	std::vector<int> getDelay() { return delay_; }

    void setDelay(std::vector<int> delay) { delay_ = delay; }
    	
	std::vector<int> getRank() { return rank_; }

    void setRank(std::vector<int> rank) { rank_ = rank; }
    
	std::vector<DATA_TYPE> getValue() { return value_; }
	
	void setValue(std::vector<DATA_TYPE> value) { value_ = value; }
	
	DATA_TYPE getDt() { return dt_; }

	void setDt(DATA_TYPE dt) { dt_ = dt; }

	DATA_TYPE getTau() { return tau_; }

	void setTau(DATA_TYPE tau) { tau_ = tau; }

	/**
	 *  \brief      Add synapse.
	 *  \param[IN]  rank    rank of the presynaptic neuron
	 *  \param[IN]  value   synaptic weight
	 *  \param[IN]  delay   delay
	 *  \return     error code: 0 (success), -1 (already existant)
	 */
	virtual int addSynapse(int rank, DATA_TYPE value, int delay);
	
	/**
	 *  \brief      Remove synapse.
	 *  \param[IN]  rank    rank of the presynaptic neuron
	 *  \return     error code: 0 (success), -1 (not existant)
	 */
    virtual int removeSynapse(int rank);
    
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
