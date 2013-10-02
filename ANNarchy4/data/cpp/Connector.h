/*
 *    Connector.h
 *
 *    This file is part of ANNarchy.
 *   
 *   Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
 *   Helge Ülo Dinkelbach <helge.dinkelbach@gmail.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   ANNarchy is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
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
	virtual void connect(Population *pre, Population *post, int projectionID, int target)=0;

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
 *	\brief		One2One pattern
 */
class One2OneConnector: public Connector {
public:
	One2OneConnector(Distribution<DATA_TYPE> *weight, Distribution<int> *delay=NULL);

	~One2OneConnector();

	void connect(Population *pre, Population *post, int projectionID, int target);
private:
	Distribution<DATA_TYPE>* weight_;
	Distribution<int>* delay_;
};
#endif
