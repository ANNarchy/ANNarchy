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
    static Profile *profileInstance() {
        if(instance_==NULL){
        #ifdef _DEBUG
            std::cout<<"Profile instance created"<<std::endl;
        #endif
            instance_ = new Profile();
        }

        return instance_;
    }

    ~Profile() {

    }

    void resetTimer() {
        for (auto it = timesSum_.begin(); it != timesSum_.end(); it++)
                it->second.clear();
        for (auto it = timesStep_.begin(); it != timesStep_.end(); it++)
                it->second.clear();
        for (auto it = timesLocal_.begin(); it != timesLocal_.end(); it++)
                it->second.clear();
        for (auto it = timesGlobal_.begin(); it != timesGlobal_.end(); it++)
                it->second.clear();
    }

    void addLayer(std::string name) {
    #ifdef _DEBUG
        std::cout<<"Population '"<<name<<"' added to profiler."<<std::endl;
    #endif
        timesSum_.insert(std::pair<std::string, std::vector<double> >(name, std::vector<double>()));
        timesStep_.insert(std::pair<std::string, std::vector<double> >(name, std::vector<double>()));
        timesLocal_.insert(std::pair<std::string, std::vector<double> >(name, std::vector<double>()));
        timesGlobal_.insert(std::pair<std::string, std::vector<double> >(name, std::vector<double>()));
    }

    void appendTimeSum(std::string name, double time) {
        if (timesSum_.count(name) > 0) {
        #ifdef _DEBUG
            std::cout<<"'"<<name<<"' time "<< time<< " ms."<< std::endl;
        #endif
            timesSum_[name].push_back(time);
        } else {
            std::cout << name << " is not registered (append time)."<< std::endl;
        }
    }

    /*
     * exported to python
     */
    double getAvgTimeSum(std::string name, int begin, int end) {
        return mean(timesSum_[name], begin, end);
    }

    /*
     * exported to python
     */
    double lastRecordedTimeSum(std::string name) {
        if (timesSum_.count(name) > 0) {
            if (timesSum_[name].size() > 0) {
                return timesSum_[name].back();
            } else {
                std::cout << "no data in list."<< std::endl;
            }
        } else {
            std::cout << name << " is not registered."<< std::endl;
        }

        return 0.0;
    }

    void appendTimeStep(std::string name, double time) {
        if (timesStep_.count(name) > 0) {
        #ifdef _DEBUG
            std::cout<<"'"<<name<<"' time "<< time<< " ms."<< std::endl;
        #endif
            timesStep_[name].push_back(time);
        } else {
            std::cout << name << " is not registered (append time)."<< std::endl;
        }
    }

    /*
     * exported to python
     */
    double getAvgTimeStep(std::string name, int begin, int end) {
        return mean(timesStep_[name], begin, end);
    }

    /*
     * exported to python
     */
    double lastRecordedTimeStep(std::string name) {
        if (timesStep_.count(name) > 0) {
            if (timesStep_[name].size() > 0) {
                return timesStep_[name].back();
            } else {
                std::cout << "no data in list."<< std::endl;
            }
        } else {
            std::cout << name << " is not registered."<< std::endl;
        }

        return 0.0;
    }

    void appendTimeLocal(std::string name, double time) {
        if (timesLocal_.count(name) > 0) {
        #ifdef _DEBUG
            std::cout<<"'"<<name<<"' time "<< time<< " ms."<< std::endl;
        #endif
            timesLocal_[name].push_back(time);
        } else {
            std::cout << name << " is not registered (append time)."<< std::endl;
        }
    }

    /*
     * exported to python
     */
    double getAvgTimeLocal(std::string name, int begin, int end) {
        return mean(timesLocal_[name], begin, end);
    }

    /*
     * exported to python
     */
    double lastRecordedTimeLocal(std::string name) {
        if (timesLocal_.count(name) > 0) {
            if (timesLocal_[name].size() > 0) {
                return timesLocal_[name].back();
            } else {
                std::cout << "no data in list."<< std::endl;
            }
        } else {
            std::cout << name << " is not registered."<< std::endl;
        }

        return 0.0;
    }

    void appendTimeGlobal(std::string name, double time) {
        if (timesGlobal_.count(name) > 0) {
        #ifdef _DEBUG
            std::cout<<"'"<<name<<"' time "<< time<< " ms."<< std::endl;
        #endif
            timesGlobal_[name].push_back(time);
        } else {
            std::cout << name << " is not registered (append time)."<< std::endl;
        }
    }

    /*
     * exported to python
     */
    double getAvgTimeGlobal(std::string name, int begin, int end) {
        return mean(timesGlobal_[name], begin, end);
    }

    /*
     * exported to python
     */
    double lastRecordedTimeGlobal(std::string name) {
        if (timesGlobal_.count(name) > 0) {
            if (timesGlobal_[name].size() > 0) {
                return timesGlobal_[name].back();
            } else {
                std::cout << "no data in list."<< std::endl;
            }
        } else {
            std::cout << name << " is not registered."<< std::endl;
        }

        return 0.0;
    }
protected:
    Profile() {
        timesSum_ = std::map< std::string, std::vector<double> >();
        timesStep_ = std::map< std::string, std::vector<double> >();
        timesGlobal_ = std::map< std::string, std::vector<double> >();
        timesLocal_ = std::map< std::string, std::vector<double> >();
    }

    inline double mean(std::vector<double> data, int begin, int end) {
        if(data.size()==0)
            return 0.0;

        double mean = 0.0;
        for(auto it=data.begin()+begin; it!= data.begin()+end;it++)
            mean += *it;

        return mean/(double)(end-begin);
    }
private:
    static Profile* instance_;
    std::map< std::string, std::vector<double> > timesSum_; /// [ pop_name, [times] ]
    std::map< std::string, std::vector<double> > timesStep_; /// [ pop_name, [times] ]
    std::map< std::string, std::vector<double> > timesGlobal_; /// [ pop_name, [times] ]
    std::map< std::string, std::vector<double> > timesLocal_; /// [ pop_name, [times] ]
};

#endif
