#ifndef __CONNECTOR_H__
#define __CONNECTOR_H__

#include "Global.h"

/**
 *	\brief		Abstract interface for all connector classes.
 */
class Connector {
public:
	/**
	 *	\brief		Constructor
	 */
	Connector() {}

	/**
	 *	\brief		Destructor
	 */
	~Connector() {}

	/**
	 *	\brief		Connects the two populations pre and post with each other. <br>
	 *				The derived function implements the algorithm, how the two populations are connected.
	 */
	virtual void connect(Population *pre, Population *post, int projectionID)=0;

	class Projection* instantiateProj(int projectionID, Population *prePopulation, Population *postPopulation, int postID, int target );
};

/**
 *	\brief		All2All pattern
 */
class All2AllConnector: public Connector {
public:
	All2AllConnector(bool allowSelfConnections, Distribution<DATA_TYPE> *weight, Distribution<int> *delay=NULL);

	~All2AllConnector();

	void connect(Population *pre, Population *post, int projectionID, int target);
private:
	bool allowSelfConnections_;
	Distribution<DATA_TYPE>* weight_;
	Distribution<int>* delay_;
};

/**
 *	\brief		All2All pattern
 */
class One2OneConnector: public Connector {
public:
	One2OneConnector(Distribution<DATA_TYPE> *weight, Distribution<int> *delay=NULL);

	~One2OneConnector();

	void connect(Population *pre, Population *post, int projectionID, int target);
private:
	bool allowSelfConnections_;
	Distribution<DATA_TYPE>* weight_;
	Distribution<int>* delay_;
};
#endif
