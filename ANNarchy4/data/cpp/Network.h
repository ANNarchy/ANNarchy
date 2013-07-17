#ifndef __NETWORK_H__
#define __NETWORK_H__

#include "Global.h"

class Network {
public:
	// functions
	static Network* instance() {
		if(instance_==NULL) {
			instance_ = new Network();
		}

		return instance_;
	}

	~Network();

	class Population* getPopulation(std::string name);

	class Population* getPopulation(unsigned int id) {
		if ( id < populations_.size() )
			return populations_[id];
		else
			return NULL;
	}

	std::vector<DATA_TYPE> getRates(int populationID);

	std::vector<DATA_TYPE> getRates(int populationID,int delay);

	std::vector<DATA_TYPE> getRates(int populationID, std::vector<int> delays, std::vector<int> ranks);

	void connect(int prePopulationID, int postPopulationID, class Connector *connector, int projectionID);

	void disconnect(int prePopulationID, int postPopulationID);

	void run(int steps);

	void addPopulation(class Population* population);

	int getTime() { return time_; }

	void setTime(int time) { time_ = time; }
protected:
	Network();

private:
	static Network *instance_;
	int time_;

	// data
	std::vector<class Population*>	populations_;
};
#endif
