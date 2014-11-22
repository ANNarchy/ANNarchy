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
// TODO:
"""