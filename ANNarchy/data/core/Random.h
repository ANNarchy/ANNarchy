/*
 *    Random.h
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
#ifndef __ANNARCHY_RANDOM_H__
#define __ANNARCHY_RANDOM_H__

#include <time.h>

#include <map>
#include <random>

/**
 *	\brief		pseudo-random generator.
 *	\details	uses the mersenne_twister_engine, provide access to different seeds.
 */
class generator {
public:
	/**
	 *	\brief		access to the generator.
	 *	\details	the class is implemented as singleton. creates an instance on first call.
	 */
	static generator* instance() { 
		if(instance_==NULL)
			instance_ = new generator();
		
		return instance_;
	}

	/**
	 *	\brief		access to the pseudo-random generator.
	 *	\details	Every distribution should work on the same random-generator stream. Additionally multiple seeds are 
	 *				allowed.
	 *	\param[IN]	seed	seed, the numbers [0..MAX_INT] are allowed. Use -1 to random initialize with (time(NULL)).
	 *	\return		reference to the generator fitting to the seed.
	 */
	std::mt19937 *getGenerator(int seed) {
		if(seeds_.count(seed)==1){
			return &seeds_[seed];
		}else{
			
			std::random_device rd;
			std::mt19937 gen(rd());

			if(seed == -1)
				gen.seed(static_cast<unsigned long>(time(NULL)));
			else
				gen.seed(seed);

			seeds_[seed] = gen;
			return &seeds_[seed];
		}
	}
private:
	/**
	 *	\brief		constructor
	 */
	generator() {

	}

	std::map<int, std::mt19937> seeds_;	///< stores all generators corresponding to their seeds.
	static generator *instance_;	///< reference of this class (singleton implementation)
};

/**
 *	\brief	Absract interface for random objects.
 */
template<typename T> 
class Distribution{
public:
	/**
	 * 	\brief		Constructor.
	 */
	Distribution() {}

	/**
	 * 	\brief		retrieve a single value of type T.
	 */
	virtual T getValue()=0;

	/**
	 * 	\brief		retrieve N values of type T as std::vector<T>.
	 */
	virtual std::vector<T> getValues(int N)=0;
};

/**
 *	\brief		Constant
 *	\details	Delivers always the same value.
 */
template <typename T>
class Constant : public Distribution<T> {
public:
	/**
	 * 	\brief		Constructor.
	 * 	\param[in]	value	value that will returned.
	 */
	Constant(T value) {
		value_ = value;
	}

	/**
	 * 	\brief		retrieve a single value of type T.
	 */
	T getValue() {
		return value_;
	}

	/**
	 * 	\brief		retrieve N values of type T as std::vector<T>.
	 */
	std::vector<T> getValues(int N) {
		return std::vector<T>(N, value_);
	}
private:
	T value_;	///< constant value
};

/**
 *	\brief		Uniform distribution
 *	\details	For floating and integer values.
 */
template <typename T>
class UniformDistribution : public Distribution<T> {
public:
	UniformDistribution(T a, T b, int seed = -1) {
		a_ = a;
		b_ = b;
		seed_ = seed;
		gen_ = generator::instance()->getGenerator(seed_);
	}

	T getValue() {
		std::uniform_real_distribution<T> dis(a_, b_);
		return dis(*gen_);
	}

	std::vector<T> getValues(int N) {
		std::vector<T> tmp = std::vector<T>();

		std::uniform_real_distribution<T> dis(a_, b_);
		for(int i =0; i< N; i++)
			tmp.push_back(dis(*gen_));

		return tmp;
	}
private:
	T a_;
	T b_;
	int seed_;
	std::mt19937 *gen_;
};

template <>
class UniformDistribution<int> : public Distribution<int> {
public:
	UniformDistribution(int a, int b, int seed = -1) {
		a_ = a;
		b_ = b;
		seed_ = seed;
		gen_ = generator::instance()->getGenerator(seed_);
	}

	int getValue() {
		std::uniform_int_distribution<> dis(a_, b_);
		return dis(*gen_);
	}

	std::vector<int> getValues(int N) {
		std::vector<int> tmp = std::vector<int>();

		std::uniform_int_distribution<> dis(a_, b_);
		for(int i =0; i< N; i++)
			tmp.push_back(dis(*gen_));

		return tmp;
	}
private:
	int a_;
	int b_;
	int seed_;
	std::mt19937 *gen_;
};

/**
 *	\brief		normal (gaussian) distribution
 *	\details	For floating and integer values.
 */
template <typename T>
class NormalDistribution : public Distribution<T> {
public:
	NormalDistribution(T mean, T sigma, int seed = -1) {
		mean_ = mean;
		sigma_ = sigma;
		seed_ = seed;
		gen_ = generator::instance()->getGenerator(seed_);
	}

	T getValue() {
		std::normal_distribution<T> dis(mean_, sigma_);
		return dis(*gen_);
	}

	std::vector<T> getValues(int N) {
		std::vector<T> tmp = std::vector<T>();

		std::normal_distribution<T> dis(mean_, sigma_);
		for(int i =0; i< N; i++)
			tmp.push_back(dis(*gen_));

		return tmp;
	}
private:
	T mean_;
	T sigma_;
	int seed_;
	std::mt19937 *gen_;
};

/**
 *	\brief		poisson distribution
 */
class PoissonDistribution : public Distribution<int> {
public:
	PoissonDistribution(int interval, int seed = -1) {
		interval_ = interval;
		gen_ = generator::instance()->getGenerator(seed_);
	}

	int getValue() {
		std::poisson_distribution<> dis(interval_);
		return dis(*gen_);
	}

	std::vector<int> getValues(int N) {
		std::vector<int> tmp = std::vector<int>();

		std::poisson_distribution<int> dis(interval_);
		for(int i =0; i< N; i++)
			tmp.push_back(dis(*gen_));

		return tmp;
	}
private:
	int interval_;
	int seed_;
	std::mt19937 *gen_;
};
#endif
