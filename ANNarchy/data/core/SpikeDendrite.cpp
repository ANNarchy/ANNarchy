/*
 *    SpikeDendrite.cpp
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
#include "SpikeDendrite.h"

SpikeDendrite::SpikeDendrite(class SpikeProjection* proj): Dendrite( false, (Projection*)proj )
{
	delayed_pre_spikes_ = std::deque<std::vector<int> >(maxDelay_, std::vector<int>());
}

void SpikeDendrite::set_delay(std::vector<int> delay)
{
	Dendrite::set_delay(delay);

	while(delayed_pre_spikes_.size() < maxDelay_ )
		delayed_pre_spikes_.push_back(std::vector<int>());
}

void SpikeDendrite::preEvent(int rank)
{
	int rk = inv_rank_[rank];
	if (maxDelay_ > 0)
	{
		if (constDelay_) {
			delayed_pre_spikes_[delay_[0]-1].push_back(rk);
		}else{
			delayed_pre_spikes_[delay_[rk]-1].push_back(rk);
		}
	}
	else
	{
		pre_spikes_.push_back(rk);
	}
}
