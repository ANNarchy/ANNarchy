/**
 *
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
	Distribution() {

	}

	virtual T getValue()=0;

	virtual std::vector<T> getValues(int N)=0;
};

/**
 *	\brief		Constant
 *	\details	Delivers always the same value.
 */
template <typename T>
class Constant : public Distribution<T> {
public:
	Constant(T value) {
		value_ = value;
	}

	T getValue() {
		return value_;
	}

	std::vector<T> getValues(int N) {
		return std::vector<T>(N, value_);
	}
private:
	T value_;
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
	}

	T getValue() {
		std::uniform_real_distribution<T> dis(a_, b_);
		return dis(*generator::instance()->getGenerator(seed_));
	}

	std::vector<T> getValues(int N) {
		std::vector<T> tmp = std::vector<T>();

		std::uniform_real_distribution<T> dis(a_, b_);
		for(int i =0; i< N; i++)
			tmp.push_back(dis(*generator::instance()->getGenerator(seed_)));

		return tmp;
	}
private:
	T a_;
	T b_;
	int seed_;
};

template <>
class UniformDistribution<int> : public Distribution<int> {
public:
	UniformDistribution(int a, int b, int seed = -1) {
		a_ = a;
		b_ = b;
		seed_ = seed;
	}

	int getValue() {
		std::uniform_int_distribution<> dis(a_, b_);
		return dis(*generator::instance()->getGenerator(seed_));
	}

	std::vector<int> getValues(int N) {
		std::vector<int> tmp = std::vector<int>();

		std::uniform_int_distribution<> dis(a_, b_);
		for(int i =0; i< N; i++)
			tmp.push_back(dis(*generator::instance()->getGenerator(seed_)));

		return tmp;
	}
private:
	int a_;
	int b_;
	int seed_;
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
	}

	T getValue() {
		std::normal_distribution<T> dis(mean_, sigma_);
		return dis(*generator::instance()->getGenerator(seed_));
	}

	std::vector<T> getValues(int N) {
		std::vector<T> tmp = std::vector<T>();

		std::normal_distribution<T> dis(mean_, sigma_);
		for(int i =0; i< N; i++)
			tmp.push_back(dis(*generator::instance()->getGenerator(seed_)));

		return tmp;
	}
private:
	T mean_;
	T sigma_;
	int seed_;
};

/**
 *	\brief		poisson distribution
 */
class PoissonDistribution : public Distribution<int> {
public:
	PoissonDistribution(int interval, int seed = -1) {
		interval_ = interval;
	}

	int getValue() {
		std::poisson_distribution<> dis(interval_);
		return dis(*generator::instance()->getGenerator(seed_));
	}

	std::vector<int> getValues(int N) {
		std::vector<int> tmp = std::vector<int>();

		std::poisson_distribution<int> dis(interval_);
		for(int i =0; i< N; i++)
			tmp.push_back(dis(*generator::instance()->getGenerator(seed_)));

		return tmp;
	}
private:
	int interval_;
	int seed_;
};
#endif