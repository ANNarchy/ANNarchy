/*
 *    Projection.cpp
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
#include "Projection.h"

Projection::Projection()
{
	isLearning_ = true;
	learnFrequency_ = 1;
	learnOffset_ = 0;
}

Projection::~Projection()
{

}

void Projection::addSynapse(int postNeuronRank, int pre, double w, int delay)
{
#ifdef _DEBUG
	std::cout << "Dendrite "<<postNeuronRank << " extend by synapse ( pre = " << pre << ", w = "<< w << ", delay = "<< delay << ")." << std::endl;
#endif

	if (dendrites_[postNeuronRank] != NULL)
		dendrites_[postNeuronRank]->addSynapse(pre, w, delay);
}

void Projection::removeSynapse(int postNeuronRank, int pre)
{
#ifdef _DEBUG
	std::cout << "Dendrite "<< postNeuronRank << " remove synapse to pre = " << pre << "." << std::endl;
#endif

	if (dendrites_[postNeuronRank] != NULL)
		dendrites_[postNeuronRank]->removeSynapse(pre);
}
