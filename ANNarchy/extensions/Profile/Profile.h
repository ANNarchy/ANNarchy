/*
 *    Profile.h
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
#ifndef __PROFILE_H__
#define __PROFILE_H__

#include "Global.h"
#include <map>
#include <vector>

class Profile{
public:
    static Profile *profileInstance()
    {
        if(instance_==NULL)
        {
        #ifdef _DEBUG_PROFILER
            std::cout<<"Profile instance created"<<std::endl;
        #endif
            instance_ = new Profile();
        }

        return instance_;
    }

    ~Profile()
    {
        // rate coding
        timesSum_.clear();
        timesStep_.clear();
        timesLocal_.clear();
        timesGlobal_.clear();

        // spiking
        timesConductance_.clear();
        timesSpikeDelivery_.clear();
        timesPreEvent_.clear();
        timesPostEvent_.clear();
    }

    void resetTimer()
    {
	#ifdef _DEBUG_PROFILER
    	std::cout << "Reset cpp profiler ... "<< std::endl;
	#endif
        timesNetwork_.clear();

        // rate coded
    	for (auto it = timesSum_.begin(); it != timesSum_.end(); it++)
                it->second.clear();
        for (auto it = timesStep_.begin(); it != timesStep_.end(); it++)
                it->second.clear();
        for (auto it = timesLocal_.begin(); it != timesLocal_.end(); it++)
                it->second.clear();
        for (auto it = timesGlobal_.begin(); it != timesGlobal_.end(); it++)
                it->second.clear();

        // spiking
        for (auto it = timesConductance_.begin(); it != timesConductance_.end(); it++)
                it->second.clear();
        for (auto it = timesSpikeDelivery_.begin(); it != timesSpikeDelivery_.end(); it++)
                it->second.clear();
        for (auto it = timesPreEvent_.begin(); it != timesPreEvent_.end(); it++)
                it->second.clear();
        for (auto it = timesPostEvent_.begin(); it != timesPostEvent_.end(); it++)
                it->second.clear();
    }

    void addLayer(std::string name)
    {
    #ifdef _DEBUG_PROFILER
        std::cout<<"Population '"<<name<<"' added to profiler."<<std::endl;
    #endif
        // rate coding
        timesSum_.insert(std::pair<std::string, std::vector<double> >(name, std::vector<double>()));
        timesStep_.insert(std::pair<std::string, std::vector<double> >(name, std::vector<double>()));
        timesLocal_.insert(std::pair<std::string, std::vector<double> >(name, std::vector<double>()));
        timesGlobal_.insert(std::pair<std::string, std::vector<double> >(name, std::vector<double>()));

        // spiking
        timesConductance_.insert(std::pair<std::string, std::vector<double> >(name, std::vector<double>()));
        timesSpikeDelivery_.insert(std::pair<std::string, std::vector<double> >(name, std::vector<double>()));
        timesPreEvent_.insert(std::pair<std::string, std::vector<double> >(name, std::vector<double>()));
        timesPostEvent_.insert(std::pair<std::string, std::vector<double> >(name, std::vector<double>()));
    }

    void appendTimeNet(double time)
    {
        timesNetwork_.push_back(time);
    }

    /*
     * exported to python
     */
    double getAvgTimeNet(int begin, int end)
    {
        return mean(timesNetwork_, begin, end, "network");
    }

    /*
     * exported to python
     */
    double lastRecordedTimeNet()
    {
    	return timesNetwork_.back();
    }

    void appendTimeSum(std::string name, double time)
    {
        if (timesSum_.count(name) > 0)
        {
        #ifdef _DEBUG_PROFILER
            std::cout<<"'"<<name<<"' time "<< time<< " ms."<< std::endl;
        #endif
            timesSum_[name].push_back(time);
        }
        else
        {
            std::cout << name << " is not registered (append time)."<< std::endl;
        }
    }

    /*
     * exported to python
     */
    double getAvgTimeSum(std::string name, int begin, int end, bool remove_outlier)
    {
    	if (begin == 0 && end == 0)
    	{
    		end = timesSum_[name].size();
    	}

    	if (timesSum_.count(name) > 0)
        {
            if (remove_outlier)
                return mean_without_outlier(timesSum_[name], begin, end, name);
            else
                return mean(timesSum_[name], begin, end, name);
        }
        else
        {
        	std::cout << name << " is not registered."<< std::endl;
        	return 0.0;
        }
    }

    double getStdDevSum(std::string name, int begin, int end, bool remove_outlier)
    {
    	if (begin == 0 && end == 0)
    	{
    		end = timesSum_[name].size();
    	}

    	if (timesSum_.count(name) > 0)
        {
            if (remove_outlier)
                return standard_deviation_without_outlier(timesSum_[name], begin, end, name);
            else
                return standard_deviation(timesSum_[name], begin, end, name);
        }
        else
        {
        	std::cout << name << " is not registered."<< std::endl;
        	return 0.0;
        }

    }
    /*
     * exported to python
     */
    double lastRecordedTimeSum(std::string name)
    {
        if (timesSum_.count(name) > 0)
        {
            if (timesSum_[name].size() > 0)
            {
                return timesSum_[name].back();
            }
            else
            {
                std::cout << "no data in list."<< std::endl;
            }
        }
        else
        {
            std::cout << name << " is not registered."<< std::endl;
        }
        return 0.0;
    }

    void appendTimeStep(std::string name, double time)
    {
        if (timesStep_.count(name) > 0)
        {
        #ifdef _DEBUG_PROFILER
            std::cout<<"'"<<name<<"' time "<< time<< " ms."<< std::endl;
        #endif
            timesStep_[name].push_back(time);
        }
        else
        {
            std::cout << name << " is not registered (append time)."<< std::endl;
        }
    }

    /*
     * exported to python
     */
    double getAvgTimeStep(std::string name, int begin, int end, bool remove_outliers)
    {
    	if (begin == 0 && end == 0)
    	{
    		end = timesSum_[name].size();
    	}

        if (timesStep_.count(name) > 0)
        {
        	if ( remove_outliers )
        		return mean_without_outlier(timesStep_[name], begin, end, name);
        	else
        		return mean(timesStep_[name], begin, end, name);
        }
        else
        {
        	std::cout << name << " is not registered."<< std::endl;
        	return 0.0;
        }
    }

    /*
     * exported to python
     */
    double getStdDevStep(std::string name, int begin, int end, bool remove_outliers = false)
    {
    	if (begin == 0 && end == 0)
    	{
    		end = timesSum_[name].size();
    	}

        if (timesStep_.count(name) > 0)
        {
        	if ( remove_outliers )
        		return standard_deviation_without_outlier(timesStep_[name], begin, end, name);
        	else
        		return standard_deviation(timesStep_[name], begin, end, name);
        }
        else
        {
        	std::cout << name << " is not registered."<< std::endl;
        	return 0.0;
        }
    }

    /*
     * exported to python
     */
    double lastRecordedTimeStep(std::string name)
    {
        if (timesStep_.count(name) > 0)
        {
            if (timesStep_[name].size() > 0)
            {
                return timesStep_[name].back();
            }
            else
            {
                std::cout << "no data in list."<< std::endl;
            }
        } else {
            std::cout << name << " is not registered."<< std::endl;
        }

        return 0.0;
    }

    void appendTimeLocal(std::string name, double time)
    {
        if (timesLocal_.count(name) > 0)
        {
        #ifdef _DEBUG_PROFILER
            std::cout<<"'"<<name<<"' time "<< time<< " ms."<< std::endl;
        #endif
            timesLocal_[name].push_back(time);
        }
        else
        {
            std::cout << name << " is not registered (append time)."<< std::endl;
        }
    }

    /*
     * exported to python
     */
    double getAvgTimeLocal(std::string name, int begin, int end, bool remove_outlier)
    {
        if (timesLocal_.count(name) > 0)
        {
            if(remove_outlier)
                return mean_without_outlier(timesLocal_[name], begin, end, name);
            else
                return mean(timesLocal_[name], begin, end, name);
        }
        else
        {
            std::cout << name << " is not registered."<< std::endl;
            return 0.0;
        }
    }

    /*
     * exported to python
     */
    double lastRecordedTimeLocal(std::string name)
    {
        if (timesLocal_.count(name) > 0)
        {
            if (timesLocal_[name].size() > 0)
            {
                return timesLocal_[name].back();
            }
            else
            {
                std::cout << "no data in list."<< std::endl;
            }
        }
        else
        {
            std::cout << name << " is not registered."<< std::endl;
        }
        return 0.0;
    }

    void appendTimeGlobal(std::string name, double time)
    {
        if (timesGlobal_.count(name) > 0)
        {
        #ifdef _DEBUG_PROFILER
            std::cout<<"'"<<name<<"' time "<< time<< " ms."<< std::endl;
        #endif
            timesGlobal_[name].push_back(time);
        }
        else
        {
            std::cout << name << " is not registered (append time)."<< std::endl;
        }
    }

    /*
     * exported to python
     */
    double getAvgTimeGlobal(std::string name, int begin, int end, bool remove_outlier)
    {
    	if (timesGlobal_.count(name) > 0)
    	{
    	    if( remove_outlier )
    	        return mean_without_outlier(timesGlobal_[name], begin, end, name);
    	    else
    	        return mean(timesGlobal_[name], begin, end, name);
    	}
    	else
    	{
    		std::cout << name << " is not registered."<< std::endl;
    		return 0.0;
    	}
    }

    /*
     * exported to python
     */
    double lastRecordedTimeGlobal(std::string name)
    {
        if (timesGlobal_.count(name) > 0)
        {
            if (timesGlobal_[name].size() > 0)
            {
                return timesGlobal_[name].back();
            }
            else
            {
                std::cout << "no data in list."<< std::endl;
            }
        }
        else
        {
            std::cout << name << " is not registered."<< std::endl;
        }

        return 0.0;
    }

    void appendTimeConductance(std::string name, double time)
    {
        if (timesConductance_.count(name) > 0)
        {
        #ifdef _DEBUG_PROFILER
            std::cout<<"'"<<name<<"' time "<< time<< " ms."<< std::endl;
        #endif
            timesConductance_[name].push_back(time);
        }
        else
        {
            std::cout << name << " is not registered (append time)."<< std::endl;
        }
    }

    /*
     * exported to python
     */
    double getAvgTimeConductance(std::string name, int begin, int end)
    {
        if (timesConductance_.count(name) > 0)
        {
        	return mean(timesConductance_[name], begin, end, name);
        }
		else
		{
			std::cout << name << " is not registered."<< std::endl;
			return 0.0;
		}
    }

    /*
     * exported to python
     */
    double lastRecordedTimeConductance(std::string name)
    {
        if (timesConductance_.count(name) > 0)
        {
            if (timesConductance_[name].size() > 0)
            {
                return timesConductance_[name].back();
            }
            else
            {
                std::cout << "no data in list."<< std::endl;
            }
        }
        else
        {
            std::cout << name << " is not registered."<< std::endl;
        }

        return 0.0;
    }

    void appendTimeSpikeDelivery(std::string name, double time)
    {
        if (timesSpikeDelivery_.count(name) > 0)
        {
        #ifdef _DEBUG_PROFILER
            std::cout<<"'"<<name<<"' time "<< time<< " ms."<< std::endl;
        #endif
            timesSpikeDelivery_[name].push_back(time);
        }
        else
        {
            std::cout << name << " is not registered (append time)."<< std::endl;
        }
    }

    /*
     * exported to python
     */
    double getAvgTimeSpikeDelivery(std::string name, int begin, int end)
    {
        if (timesSpikeDelivery_.count(name) > 0)
        {
        	return mean(timesSpikeDelivery_[name], begin, end, name);
        }
		else
		{
			std::cout << name << " is not registered."<< std::endl;
			return 0.0;
		}
    }

    /*
     * exported to python
     */
    double lastRecordedTimeSpikeDelivery(std::string name)
    {
        if (timesSpikeDelivery_.count(name) > 0)
        {
            if (timesSpikeDelivery_[name].size() > 0)
            {
                return timesSpikeDelivery_[name].back();
            }
            else
            {
                std::cout << "no data in list."<< std::endl;
            }
        }
        else
        {
            std::cout << name << " is not registered."<< std::endl;
        }

        return 0.0;
    }

    void appendTimePreEvent(std::string name, double time)
    {
        if (timesPreEvent_.count(name) > 0)
        {
        #ifdef _DEBUG_PROFILER
            std::cout<<"'"<<name<<"' time "<< time<< " ms."<< std::endl;
        #endif
            timesPreEvent_[name].push_back(time);
        }
        else
        {
            std::cout << name << " is not registered (append time)."<< std::endl;
        }
    }

    /*
     * exported to python
     */
    double getAvgTimePreEvent(std::string name, int begin, int end)
    {
        if (timesPreEvent_.count(name) > 0)
        {
        	return mean(timesPreEvent_[name], begin, end, name);
        }
		else
		{
			std::cout << name << " is not registered."<< std::endl;
			return 0.0;
		}
    }

    /*
     * exported to python
     */
    double lastRecordedTimePreEvent(std::string name)
    {
        if (timesPreEvent_.count(name) > 0)
        {
            if (timesPreEvent_[name].size() > 0)
            {
                return timesPreEvent_[name].back();
            }
            else
            {
                std::cout << "no data in list."<< std::endl;
            }
        }
        else
        {
            std::cout << name << " is not registered."<< std::endl;
        }

        return 0.0;
    }

    void appendTimePostEvent(std::string name, double time)
    {
        if (timesPostEvent_.count(name) > 0)
        {
        #ifdef _DEBUG_PROFILER
            std::cout<<"'"<<name<<"' time "<< time<< " ms."<< std::endl;
        #endif
            timesPostEvent_[name].push_back(time);
        }
        else
        {
            std::cout << name << " is not registered (append time)."<< std::endl;
        }
    }

    /*
     * exported to python
     */
    double getAvgTimePostEvent(std::string name, int begin, int end)
    {
        if (timesPostEvent_.count(name) > 0)
        {
        	return mean(timesPostEvent_[name], begin, end, name);
        }
		else
		{
			std::cout << name << " is not registered."<< std::endl;
			return 0.0;
		}
    }

    /*
     * exported to python
     */
    double lastRecordedTimePostEvent(std::string name)
    {
        if (timesPostEvent_.count(name) > 0)
        {
            if (timesPostEvent_[name].size() > 0)
            {
                return timesPostEvent_[name].back();
            }
            else
            {
                std::cout << "no data in list."<< std::endl;
            }
        }
        else
        {
            std::cout << name << " is not registered."<< std::endl;
        }

        return 0.0;
    }

    void set_std_dev_scale(double scale_factor)
    {
    	std_dev_scale_ = scale_factor;
    }
protected:
    Profile()
	{
    	std_dev_scale_ = 3.0;

    	timesNetwork_ = std::vector<double> ();

    	// rate coded variables
        timesSum_ = std::map< std::string, std::vector<double> >();
        timesStep_ = std::map< std::string, std::vector<double> >();
        timesGlobal_ = std::map< std::string, std::vector<double> >();
        timesLocal_ = std::map< std::string, std::vector<double> >();

        // spiking variables
        timesConductance_ = std::map< std::string, std::vector<double> >();
        timesSpikeDelivery_ = std::map< std::string, std::vector<double> >();
        timesPreEvent_ = std::map< std::string, std::vector<double> >();
        timesPostEvent_ = std::map< std::string, std::vector<double> >();
    }

    inline double mean(std::vector<double> data, int begin, int end, std::string name)
    {
        if(data.size()==0)
        {
        	std::cout << name << ":"<< std::endl;
        	std::cout << "   No recorded data ..." << std::endl;
        	return 0.0;
        }
        else
        {
			double mean = 0.0;
			for(auto it=data.begin()+begin; it!= data.begin()+end;it++)
				mean += *it;

			return mean/(double)(end-begin);
        }
    }

    inline double variance(std::vector<double> data, int begin, int end, std::string name)
    {
        if(data.size()==0)
        {
        	std::cout << name << ":"<< std::endl;
        	std::cout << "   No recorded data ..." << std::endl;
        	return 0.0;
        }
        else
        {
        	double mean_value = mean(data, begin, end, name);
        	double tmp = 0.0;
        	for(auto it=data.begin()+begin; it!= data.begin()+end;it++)
        		tmp += ( (*it) - mean_value ) * ( (*it) - mean_value );

        	return tmp / (double)(end - begin -1);
        }
    }

    inline double standard_deviation(std::vector<double> data, int begin, int end, std::string name)
    {
        if(data.size()==0)
        {
        	std::cout << name << ":"<< std::endl;
        	std::cout << "   No recorded data ..." << std::endl;
        	return 0.0;
        }
        else
        {
        	return std::sqrt(variance(data, begin, end, name));
        }
    }

    inline std::vector<double> remove_outlier(std::vector<double> data, int begin, int end, std::string name)
    {
    	auto cp_data = std::vector<double>(data.begin()+begin, data.begin()+end);

    	double std_dev = standard_deviation(data, 0, cp_data.size(), name);
    	double mean_value = mean(data, 0, cp_data.size(), name);

    	auto it = cp_data.begin();
    	while ( it != cp_data.end() )
    	{
    		if ( (*it) > (mean_value + std_dev_scale_ * std_dev) || (*it) < (mean_value - std_dev_scale_ * std_dev) )
    		{
    			it = cp_data.erase(it);
    		}
    		else
    		{
    			it++;
    		}
    	}

    	return cp_data;
    }

    inline double mean_without_outlier(std::vector<double> data, int begin, int end, std::string name)
    {
    	if(data.size()==0)
		{
			std::cout << name << ":"<< std::endl;
			std::cout << "   No recorded data ..." << std::endl;
			return 0.0;
		}
		else
		{
			auto cleaned_data = remove_outlier(data, begin, end, name);

			if ( cleaned_data.size() > 0 )
			{
				//std::cout << "Removed " << (end-begin) - cleaned_data.size() << " items from data (" << ( 1 - cleaned_data.size()/(double)(end-begin)) * 100.0 << " % )." << std::endl;
				return mean(cleaned_data, 0, cleaned_data.size(), name);
			}
			else
			{
				std::cout << "NOTICE: bad relation between mean and standard deviation." << std::endl;
				return mean(data, begin, end, name);
			}
		}
    }

    inline double standard_deviation_without_outlier(std::vector<double> data, int begin, int end, std::string name)
    {
    	if(data.size()==0)
		{
			std::cout << name << ":"<< std::endl;
			std::cout << "   No recorded data ..." << std::endl;
			return 0.0;
		}
		else
		{
	    	auto cleaned_data = remove_outlier(data, begin, end, name);

			if ( cleaned_data.size() > 0 )
			{
				//std::cout << "Removed " << (end-begin) - cleaned_data.size() << " items from data (" << ( 1 - cleaned_data.size() / (double)(end-begin)) * 100.0 << " % )." << std::endl;
				return std::sqrt(variance(cleaned_data, 0, cleaned_data.size(), name));
			}
			else
			{
				//std::cout << "bad relation between mean and standard deviation." << std::endl;
				return std::sqrt(variance(data, begin, end, name));
			}
		}
    }
private:
    static Profile* instance_;
    std::vector < double > timesNetwork_;
    double std_dev_scale_;

    // rate coded measurements
    std::map< std::string, std::vector<double> > timesSum_; 	/// [ pop_name, [times] ]
    std::map< std::string, std::vector<double> > timesStep_; 	/// [ pop_name, [times] ]
    std::map< std::string, std::vector<double> > timesGlobal_; 	/// [ pop_name, [times] ]
    std::map< std::string, std::vector<double> > timesLocal_; 	/// [ pop_name, [times] ]

    // spiking measurements
    std::map< std::string, std::vector<double> > timesConductance_; 	/// [ pop_name, [times] ]
    std::map< std::string, std::vector<double> > timesSpikeDelivery_; 	/// [ pop_name, [times] ]
    std::map< std::string, std::vector<double> > timesPreEvent_;	 	/// [ pop_name, [times] ]
    std::map< std::string, std::vector<double> > timesPostEvent_; 		/// [ pop_name, [times] ]
};

#endif
