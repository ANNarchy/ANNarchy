cuda_profile_header =\
"""
#ifndef __PROFILING_H__
#define __PROFILING_H__

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <float.h>
#include <omp.h>
#include <papi.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#define pos(x) (x>0.0? x : 0.0)
#define sqr(x) (x*x)

class Profiling {
    
    public:
        Profiling() ;

        ~Profiling() {
        }
    
    /*Set/get Anzahl,Namen*/
    void set_CPU_time_number( int number){Profiling_time_CPU_count=number;}
    void set_GPU_time_number( int number){Profiling_time_count=number;}
    void set_Init_time_number(int number){Profiling_time_init_count=number;}
    void set_memcopy_number(  int number){Profiling_memcopy_count=number;}

    //Aufruf direkt nach allen set Number's
    void init(int extended=1);
    void init_GPU_prof(void);

    //Funktionen um alle Profiling Aktionen an(Standart)/ab zuschalten(betrifft !ALLE! start/stop/evaluate)->Messung Gesammtzeit ohne Profiling Wartepausen
    void set_profiling_off(){Profil=0;}
    void set_profiling_on( ){Profil=1;}

    
    //Folgende Funktionen erst nach init nutzbar
    void set_CPU_time_name( int number,std::string name){Prof_time_CPU[number].name=name;}
    void set_GPU_time_name( int number,std::string name){Prof_time[number].name=name;}
    void set_Init_time_name(int number,std::string name){Prof_time_init[number].name=name;}
    void set_memcopy_name(  int number,std::string name){Prof_memcopy[number].name=name;}

    void start_CPU_time_prof( int number);
    void start_GPU_time_prof( int number);
    void start_Init_time_prof(int number);
    void start_memcopy_prof(  int number,int bytesize);
    void start_overall_time_prof();

    void stop_CPU_time_prof( int number,int directevaluate=1);
    void stop_GPU_time_prof( int number,int directevaluate=1);
    void stop_Init_time_prof(int number,int directevaluate=1);
    void stop_memcopy_prof(  int number,int directevaluate=1);
    void stop_overall_time_prof();

    void evaluate_CPU_time_prof( int number);
    void evaluate_GPU_time_prof( int number);
    void evaluate_Init_time_prof(int number);
    void evaluate_memcopy_prof(  int number);
    void evaluate_overall_time_prof();

    void evaluate(int disp, int file,const char * filename="Profiling.log");

    private:

    struct Profiling_unit{
        long count=0;
        double min=FLT_MAX;
        double max=FLT_MIN;
        double avg;
        double standard;//Standard deviation
        double prozent_CPU;
        double prozent_GPU;
        double summ=0;
        double summsqr=0;
    };

    struct Profiling_time{
           std::string name="";
        Profiling_unit time;

        cudaEvent_t startevent, stopevent;
        long_long start,stop;
    };
    struct Profiling_memcopy{
           std::string name="";
        Profiling_unit time;
        Profiling_unit memorysize;
        Profiling_unit memorythroughput;

        cudaEvent_t startevent, stopevent;
        long_long start,stop;
        int memory;
    };

    struct Profiling_general{
        double CPU_summ=0;
        double GPU_summ=0;

        long_long start,stop;
    };

    int Profiling_time_count=0;
    int Profiling_time_CPU_count=0;
    int Profiling_time_init_count=0;
    int Profiling_memcopy_count=0;

    int Profil=1;

    //Profiling
        Profiling_time *Prof_time;
        Profiling_time *Prof_time_CPU;
        Profiling_time *Prof_time_init;
        Profiling_memcopy *Prof_memcopy;
        Profiling_general Prof_general;

    void evaluate_calc();
    void evaluate_disp();
    int evaluate_file(const char * filename="Profiling.log");
};

#endif
"""

openmp_profile_header=\
"""
#ifndef __PROFILING_CPU_H__
#define __PROFILING_CPU_H__

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <float.h>
#include <omp.h>
#include <papi.h>
#include <sched.h>  // sched_getcpu
#include <algorithm>

#define pos(x) (x>0.0? x : 0.0)
#define sqr(x) (x*x)
#define checkstring(x) ((x.find(":")!=std::string::npos)||(x.find("#")!=std::string::npos)||(x.find("\\n")!=std::string::npos))

class Profiling {
    
    public:
        Profiling() ;

        ~Profiling() {
        }

/*
Thread statistic: gruppen: summenbildung-> anz threads[kerne]: summ min avg max standart_devariation
*/
    
    /*Set/get Anzahl,Namen*/
    void set_CPU_time_number(  int number){Profiling_time_CPU_count=number;}
    void set_CPU_cycles_number(int number){Profiling_cycles_CPU_count=number;}
    void set_thread_statistic_number(int number){Profiling_thread_count=number;}
    void set_max_thread_number(int number){thread_count=number;}
    void set_max_core_number(int number){core_count=number;}//don't needed

    //Aufruf direkt nach allen set Number's
    void init(int extended=1);
    void init_thread();

    //Funktionen um alle Profiling Aktionen an(Standart)/ab zuschalten(betrifft !ALLE! start/stop/evaluate)->Messung Gesammtzeit ohne Profiling Wartepausen
    void set_profiling_off(){Profil=0;}
    void set_profiling_on( ){Profil=1;}

    //Folgende Funktionen erst nach init nutzbar
    void set_CPU_time_name(  int number,std::string name){Prof_time_CPU[number].name=name;}
    void set_CPU_cycles_name(int number,std::string name){Prof_cycles_CPU[number].name=name;}
    void set_thread_statistic_name(int number,std::string name){Prof_thread_statistic[number].name=name;}

    //additonal Syntax fuer Python Auswertung: Zahl(X-Ache);String
    void set_CPU_time_additonal(  int number,std::string additonal){Prof_time_CPU[number].additonal=additonal;}
    void set_CPU_cycles_additonal(int number,std::string additonal){Prof_cycles_CPU[number].additonal=additonal;}
    void set_thread_statistic_additonal(int number,std::string additonal){Prof_thread_statistic[number].additonal=additonal;}

    void start_CPU_time_prof( int number);
    void start_overall_time_prof();
    void start_CPU_cycles_prof( int number);

    void stop_CPU_time_prof( int number,int directevaluate=1);
    void stop_overall_time_prof();
    void stop_CPU_cycles_prof( int number,int directevaluate=1);

    void evaluate_CPU_time_prof( int number);
    void evaluate_overall_time_prof();
    void evaluate_CPU_cycles_prof( int number);

    void error_CPU_time_prof();
    void error_CPU_cycles_prof();

    //use at parallel loop
    void thread_statistic_run( int number);

    void evaluate(int disp, int file,const char * filename="Profiling.log");

    private:

    struct Profiling_unit{
        long count=0;
        double min=FLT_MAX;
        double max=FLT_MIN;
        double avg;
        double standard;//Standard deviation
        double prozent_CPU;
        double summ=0;
        double summsqr=0;

        // [hdin] sometimes needed for additional evaluation
        bool storeRawData=true;
        double adjustedMean=0.0;
        std::vector<double> rawData=std::vector<double>();
    };

    struct Profiling_thread_statistic_unit{
        volatile long count=0;
    };

    struct Profiling_time{
           std::string name="";
           std::string additonal="";
        Profiling_unit time;

        long_long start,stop;
    };

    struct Profiling_thread_statistic_core{
        Profiling_thread_statistic_unit *core;
    };

    struct Profiling_thread_statistic{
           std::string name="";
           std::string additonal="";
        Profiling_thread_statistic_core *thread;
        int used_threads=0;
    };

    struct Profiling_general{
        double CPU_summ=0;

        long_long start,stop;
    };

    int Profiling_time_CPU_count=0;
    int Profiling_cycles_CPU_count=0;
    int Profiling_thread_count=0;
    int thread_count=0;
    int core_count=0;

    int Profil=1;

    //Profiling
    Profiling_time *Prof_time_CPU;
    Profiling_time *Prof_cycles_CPU;
    Profiling_thread_statistic *Prof_thread_statistic;
    Profiling_general Prof_general;

    void evaluate_calc();
    void evaluate_disp();
    int evaluate_file(const char * filename="Profiling.log");
    
    /**
     *    \brief        Adjusting mean by removing items from dataset
     *    \details      removal of the 5 percent highest and lowest values from the dataset. This is only valid if the dataset is big enough
     *    \param[in]    data    vector containing the data to compute mean from
     *    \return       in case of either no data or less then 100 elements the function return 0.0
     *                  otherwise returns mean over remaining data
     */
    double compAdjustedMeanV1(std::vector<double> data) {
        if ( data.empty() )
            return 0.0;

        if ( data.size() < 100 )
            return 0.0;

        int toRemove = ceil(double(data.size()) / 100.0 * 5.0); // remove 10 percent of the items 
        std::vector<double> tmp = data; // need to copy vector, otherwise we would overwrite input
        std::sort( tmp.begin(), tmp.end() ); // default comparison is '<'
        
        tmp.erase( tmp.end()-toRemove, tmp.end());
        tmp.erase( tmp.begin(), tmp.begin()+toRemove);
        
        double sum = 0.0;
        for ( auto it = tmp.begin(); it != tmp.end(); it++ )
            sum+= *it;
            
        return sum / double(tmp.size());
    }
    
    /**
     *    \brief        compute mean
     *    \details      implemented as template function
     *    \param[in]    data    vector containing the data to compute standard deviation from
     *    \return       computed mean
     */
    template<typename T>
    T compMean(std::vector<T> data) {
        // compute mean
        T sum = 0.0;
        for ( auto it = data.begin(); it != data.end(); it++ )
            sum += *it;
        return T(sum / double(data.size()));
    }
    
    /**
     *    \brief        compute stdandard deviation
     *    \details      implemented as template function and use N-1 as divisor
     *    \param[in]    data    vector containing the data to compute standard deviation from
     *    \param[in]    mean    pre-computed mean over data
     *    \return       standard deviation
     */
    template<typename T>
    T compStandardDeviation(std::vector<T> data, T mean) {
        // compute varianz
        T var = 0.0;
        for ( auto it = data.begin(); it != data.end(); it++ )
            var += (*it-mean)*(*it-mean);
        
        // Notice: there are both variants in literature divide b N or N-1,
        //         I decided I take the second version, as it seems the more correct one.
        return sqrt( (var / double(data.size()-1) ) );
    }

    /**
     *    \brief        Adjusting mean by removing items dependant on stdandard deviation
     *    \details      removal of the items which are to far from mean ( 3 times standard deviation )
     *    \param[in]    data               vector containing the data to compute mean from
     *    \param[in]    thresholdFactor    determine size of window <=> mean +- thresholdFactor * std_dev 
     *    \return       in case of no data the function return 0.0
     *                  otherwise returns mean over adjusted data
     */
    double compAdjustedMeanV2(std::vector<double> data, double thresholdFactor = 3.0) {
        if ( data.empty() )
            return 0.0;

        // compute mean and standard deviation
        double mean = compMean<double>(data);
        double std_dev = compStandardDeviation<double>(data, mean);
        
        // remove all items higher then threshold
        std::vector<double> tmp = data;
        double pos_threshold = mean+thresholdFactor*std_dev;
        double neg_threshold = mean-thresholdFactor*std_dev;
        for ( auto it = tmp.begin(); it != tmp.end(); it++ )
            if ( *it < neg_threshold || *it > pos_threshold )
                it = tmp.erase(it);

        // compute and return adjusted mean
        return compMean<double>(tmp);
    }    
};
#endif
"""