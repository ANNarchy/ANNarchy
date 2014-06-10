/*    ParallelLogger.h
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
#ifndef __ANNARCHY_PARALLEL_LOGGER_H__
#define __ANNARCHY_PARALLEL_LOGGER_H__

class ParallelLogger
{
private:
	std::vector<int>** data_;
public:
	/**
	 * 	\brief	create a parallel logging instance
	 * 	\param[in]	t	number of threads
	 * 	\param[out]	N	number of elements maximum logged
	 */
	ParallelLogger (int t, int N)
	{
		std::cout << "ParallelLogger::ParallelLogger(t = "<< t << ", N = "<< N <<")" << std::endl;
		data_ = new std::vector<int>*[t];

		#pragma omp parallel
		{
			//
			// we need to allocate enough elements here, otherwise the
			// reallocation will cause a runtime exception (segmentation fault)
			data_[omp_get_thread_num()] = new std::vector<int>();
			data_[omp_get_thread_num()]->reserve(N);
		}
	}

	~ParallelLogger ()
	{
		std::cout << "ParallelLogger::~ParallelLogger" << std::endl;
	}

	void resize(int N)
	{
		std::cout << "ParallelLogger::resize( N = "<< N <<")" << std::endl;
		#pragma omp parallel
		{
			//
			// we need to allocate enough elements here, otherwise the
			// reallocation will cause a runtime exception (segmentation fault)
			data_[omp_get_thread_num()]->reserve(N);
		}
	}

	void add(int t, int n)
	{
		#pragma omp critical
		{
		#ifdef _DEBUG
			std::cout << "ParallelLogger::add( t = "<< t << ", n = " << n << ")" << std::endl;
		#endif
			data_[t]->push_back(n);
		}
	}

	void neuron_on_thread_statistic( bool clear_all = false )
	{
		for ( int t = 0; t < omp_get_max_threads(); t++)
		{
			std::cout << "Thread " << t << ": ";
			for( auto it = data_[t]->begin(); it != data_[t]->end(); it++)
				std::cout << *it << " ";
			std::cout << std::endl;

			if(clear_all)
				data_[t]->clear();
		}
	}

	void number_neurons_per_thread( bool clear_all = false )
	{
		for ( int t = 0; t < omp_get_max_threads(); t++)
		{
			std::cout << "Thread " << t << ": "<< data_[t]->size()<< " neuron(s)"<< std::endl;
			if(clear_all)
				data_[t]->clear();
		}
	}

	void clear()
	{
		#pragma omp critical
		{
			std::cout << "ParallelLogger::clear" << std::endl;
			for ( int t = 0; t < omp_get_max_threads(); t++)
			{
				data_[t]->clear();
			}
		}
	}
};

#endif
