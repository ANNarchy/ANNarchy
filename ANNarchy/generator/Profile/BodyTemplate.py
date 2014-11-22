cuda_profile_body=\
"""
#include "Profiling.h"

Profiling::Profiling() {
}

/*
    Init
*/
void Profiling::init(int extended)
{
 //Profiling Array allocate
        Prof_time=     new Profiling_time[Profiling_time_count];
        Prof_time_CPU= new Profiling_time[Profiling_time_CPU_count];
        Prof_time_init=new Profiling_time[Profiling_time_init_count];
        Prof_memcopy=  new Profiling_memcopy[Profiling_memcopy_count];

 //additional initialisations
    if (extended){
        init_GPU_prof();
    }
}

void Profiling::init_GPU_prof(void)
{
 //Initial Profiling Dummy
    cudaEvent_t event1, event2;
    long_long start,stop;

    //create events
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    //record events around kernel launch
    cudaEventRecord(event1, 0); //where 0 is the default stream

    start = PAPI_get_real_usec();    
    
    stop = PAPI_get_real_usec();
    cudaEventRecord(event2, 0);

    //synchronize
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!
    float dt_ms;
    //calculate time
    cudaEventElapsedTime(&dt_ms, event1, event2);
}

/*
    GPU Time
*/
void Profiling::start_GPU_time_prof( int number)
{
  if(Profil){
    //create events
    cudaEventCreate(&Prof_time[number].startevent);
    cudaEventCreate(&Prof_time[number].stopevent);

    //record events around kernel launch
    cudaEventRecord(Prof_time[number].startevent, 0); //where 0 is the default stream

    Prof_time[number].start = PAPI_get_real_usec();
  }
}
    
void Profiling::stop_GPU_time_prof( int number,int directevaluate)
{
  if(Profil){
    Prof_time[number].stop = PAPI_get_real_usec();
    cudaEventRecord(Prof_time[number].stopevent, 0);

    //synchronize
    cudaEventSynchronize(Prof_time[number].startevent);
    cudaEventSynchronize(Prof_time[number].stopevent); //wait for the event to be executed!
    
    if (directevaluate){
            evaluate_GPU_time_prof(number);
    }
  }
}

void Profiling::evaluate_GPU_time_prof( int number)
{
  if(Profil){
    float dt_ms;
    //calculate time
    cudaEventElapsedTime(&dt_ms, Prof_time[number].startevent, Prof_time[number].stopevent);

    //pre-evaluate GPU time
    dt_ms-=((float)(Prof_time[number].stop-Prof_time[number].start))/1000;//Launch time
    dt_ms/=1000;//ms => s

    Prof_time[number].time.summ+=dt_ms;
    Prof_general.GPU_summ+=dt_ms;
    Prof_time[number].time.count++;
    Prof_time[number].time.summsqr+=sqr(dt_ms);
    (dt_ms<Prof_time[number].time.min)?(Prof_time[number].time.min=dt_ms):1;
    (Prof_time[number].time.max<dt_ms)?(Prof_time[number].time.max=dt_ms):1;//min/max
  }
}

/*
    memcopy
*/
void Profiling::start_memcopy_prof( int number,int bytesize)
{
  if(Profil){
    Prof_memcopy[number].memory=bytesize;

    //create events
    cudaEventCreate(&Prof_memcopy[number].startevent);
    cudaEventCreate(&Prof_memcopy[number].stopevent);

    //record events around kernel launch
    cudaEventRecord(Prof_memcopy[number].startevent, 0); //where 0 is the default stream

    Prof_memcopy[number].start = PAPI_get_real_usec();    
  }
}
    
void Profiling::stop_memcopy_prof( int number,int directevaluate)
{
  if(Profil){
    Prof_memcopy[number].stop = PAPI_get_real_usec();
    cudaEventRecord(Prof_memcopy[number].stopevent, 0);

    //synchronize
    cudaEventSynchronize(Prof_memcopy[number].startevent);
    cudaEventSynchronize(Prof_memcopy[number].stopevent); //wait for the event to be executed!
    
    if (directevaluate){
            evaluate_memcopy_prof(number);
    }
  }
}

void Profiling::evaluate_memcopy_prof( int number)
{
  if(Profil){
    float dt_ms;
    //calculate time
    cudaEventElapsedTime(&dt_ms, Prof_memcopy[number].startevent, Prof_memcopy[number].stopevent);

    //pre-evaluate GPU time
    dt_ms-=((float)(Prof_memcopy[number].stop-Prof_memcopy[number].start))/1000;//Launch time
    dt_ms/=1000;//ms => s
//calculate time
    Prof_memcopy[number].time.summ+=dt_ms;
    Prof_general.GPU_summ+=dt_ms;//memcopy is part of GPU
    Prof_memcopy[number].time.count++;
    Prof_memcopy[number].time.summsqr+=sqr(dt_ms);
    (dt_ms<Prof_memcopy[number].time.min)?(Prof_memcopy[number].time.min=dt_ms):1;
    (Prof_memcopy[number].time.max<dt_ms)?(Prof_memcopy[number].time.max=dt_ms):1;//min/max
//calculate memory
    double bytesize=Prof_memcopy[number].memory;
    Prof_memcopy[number].memorysize.summ+=bytesize;
    Prof_memcopy[number].memorysize.count++;
    Prof_memcopy[number].memorysize.summsqr+=sqr(bytesize);
    (bytesize<Prof_memcopy[number].memorysize.min)?(Prof_memcopy[number].memorysize.min=bytesize):1;
    (Prof_memcopy[number].memorysize.max<bytesize)?(Prof_memcopy[number].memorysize.max=bytesize):1;//min/max
//calculate Throughput
    double memorythroughput=(bytesize)/dt_ms;
    Prof_memcopy[number].memorythroughput.summ+=memorythroughput;
    Prof_memcopy[number].memorythroughput.count++;
    Prof_memcopy[number].memorythroughput.summsqr+=sqr(memorythroughput);
    (memorythroughput<Prof_memcopy[number].memorythroughput.min)?(Prof_memcopy[number].memorythroughput.min=memorythroughput):1;
    (Prof_memcopy[number].memorythroughput.max<memorythroughput)?(Prof_memcopy[number].memorythroughput.max=memorythroughput):1;//min/max

  }
}

/*
    CPU Time
*/
void Profiling::start_CPU_time_prof( int number)
{
  if(Profil){
    Prof_time_CPU[number].start = PAPI_get_real_usec();    
  }
}
    
void Profiling::stop_CPU_time_prof( int number,int directevaluate)
{
  if(Profil){
    Prof_time_CPU[number].stop = PAPI_get_real_usec();

    if (directevaluate){
            evaluate_CPU_time_prof(number);
     }
  }
}

void Profiling::evaluate_CPU_time_prof( int number)
{
  if(Profil){
    double dt_ms;

    //pre-evaluate CPU time
    dt_ms=((double)(Prof_time_CPU[number].stop-Prof_time_CPU[number].start))/1000;//Launch time
    dt_ms/=1000;//ms => s

    Prof_time_CPU[number].time.summ+=dt_ms;
    Prof_time_CPU[number].time.count++;
    Prof_time_CPU[number].time.summsqr+=sqr(dt_ms);
    (dt_ms<Prof_time_CPU[number].time.min)?(Prof_time_CPU[number].time.min=dt_ms):1;
    (Prof_time_CPU[number].time.max<dt_ms)?(Prof_time_CPU[number].time.max=dt_ms):1;//min/max
  }
}

/*
    Init Time
*/
void Profiling::start_Init_time_prof( int number)
{
  if(Profil){
    Prof_time_init[number].start = PAPI_get_real_usec();    
  }
}
    
void Profiling::stop_Init_time_prof( int number,int directevaluate)
{
  if(Profil){
    Prof_time_init[number].stop = PAPI_get_real_usec();

    if (directevaluate){
        evaluate_Init_time_prof(number);
    }
  }
}

void Profiling::evaluate_Init_time_prof( int number)
{
  if(Profil){
    double dt_ms;

    //pre-evaluate CPU time
    dt_ms=((double)(Prof_time_init[number].stop-Prof_time_init[number].start))/1000;//Launch time
    dt_ms/=1000;//ms => s

    Prof_time_init[number].time.summ+=dt_ms;
    Prof_time_init[number].time.count++;
    Prof_time_init[number].time.summsqr+=sqr(dt_ms);
    (dt_ms<Prof_time_init[number].time.min)?(Prof_time_init[number].time.min=dt_ms):1;
    (Prof_time_init[number].time.max<dt_ms)?(Prof_time_init[number].time.max=dt_ms):1;//min/max
  }
}

/*
    Overall Time
*/
void Profiling::start_overall_time_prof()
{
  if(Profil){
    cudaDeviceSynchronize();
    Prof_general.start = PAPI_get_real_usec();    
  }
}
    
void Profiling::stop_overall_time_prof()
{
  if(Profil){
    cudaDeviceSynchronize();

    Prof_general.stop = PAPI_get_real_usec();

    evaluate_overall_time_prof();
  }
}

void Profiling::evaluate_overall_time_prof()
{
  if(Profil){
    Prof_general.CPU_summ=((double)(Prof_general.stop-Prof_general.start))/1000/1000;//s
  }
}


/*
    Evaluation
*/
void Profiling::evaluate(int disp, int file,const char * filename)
{
    evaluate_calc();
    if (disp)evaluate_disp();
    if (file)evaluate_file(filename);
}

void Profiling::evaluate_calc()
{
    for(int i=0;i<Profiling_time_count;i++){
        Prof_time[i].time.avg=Prof_time[i].time.summ/Prof_time[i].time.count;
        Prof_time[i].time.standard=sqrt(Prof_time[i].time.summsqr/Prof_time[i].time.count-sqr(Prof_time[i].time.avg));
        Prof_time[i].time.prozent_CPU=100*Prof_time[i].time.summ/Prof_general.CPU_summ;
        Prof_time[i].time.prozent_GPU=100*Prof_time[i].time.summ/Prof_general.GPU_summ;
    }
    for(int i=0;i<Profiling_time_CPU_count;i++){
        Prof_time_CPU[i].time.avg=Prof_time_CPU[i].time.summ/Prof_time_CPU[i].time.count;
        Prof_time_CPU[i].time.standard=sqrt(Prof_time_CPU[i].time.summsqr/Prof_time_CPU[i].time.count-sqr(Prof_time_CPU[i].time.avg));
        Prof_time_CPU[i].time.prozent_CPU=100*Prof_time_CPU[i].time.summ/Prof_general.CPU_summ;
        Prof_time_CPU[i].time.prozent_GPU=0;
    }
    for(int i=0;i<Profiling_time_init_count;i++){
        Prof_time_init[i].time.prozent_CPU=100*Prof_time_init[i].time.summ/Prof_general.CPU_summ;
    }
    for(int i=0;i<Profiling_memcopy_count;i++){
        Prof_memcopy[i].time.avg=Prof_memcopy[i].time.summ/Prof_memcopy[i].time.count;
        Prof_memcopy[i].time.standard=sqrt(Prof_memcopy[i].time.summsqr/Prof_memcopy[i].time.count-sqr(Prof_memcopy[i].time.avg));
        Prof_memcopy[i].time.prozent_CPU=100*Prof_memcopy[i].time.summ/Prof_general.CPU_summ;
        Prof_memcopy[i].time.prozent_GPU=100*Prof_memcopy[i].time.summ/Prof_general.GPU_summ;

        Prof_memcopy[i].memorysize.avg=Prof_memcopy[i].memorysize.summ/Prof_memcopy[i].memorysize.count;
        Prof_memcopy[i].memorysize.standard=sqrt(Prof_memcopy[i].memorysize.summsqr/Prof_memcopy[i].memorysize.count-sqr(Prof_memcopy[i].memorysize.avg));

        Prof_memcopy[i].memorythroughput.avg=Prof_memcopy[i].memorythroughput.summ/Prof_memcopy[i].memorythroughput.count;
        Prof_memcopy[i].memorythroughput.standard=sqrt(Prof_memcopy[i].memorythroughput.summsqr/Prof_memcopy[i].memorythroughput.count-sqr(Prof_memcopy[i].memorythroughput.avg));
    }
}
void Profiling::evaluate_disp()
{
    std::cout.precision(8);
    std::cout << "Overall time: "<< std::fixed << Prof_general.CPU_summ     << "s " <<"On GPU only: "<< std::fixed << Prof_general.GPU_summ     << "s "<< std::endl;
    for(int i=0;i<Profiling_time_init_count;i++){
        int found = (Prof_time_init[i].name.find(":")!=std::string::npos)||(Prof_time_CPU[i].name.find("#")!=std::string::npos);//1 wenn Prof_time_init[i].name ungueltig 
        if ((Prof_time_CPU[i].name=="")||(found))
            std::cout << "Initialisation_Time "<<i;
        else     
            std::cout << Prof_time_init[i].name;
        std::cout     <<" time: "          << std::fixed << Prof_time_init[i].time.summ     << "s "
                << "Faktor CPU time: " << std::fixed << Prof_time_init[i].time.prozent_CPU/100<< std::endl;
    }
        std::cout     << std::endl;
    for(int i=0;i<Profiling_memcopy_count;i++){
        int found = (Prof_memcopy[i].name.find(":")!=std::string::npos)||(Prof_memcopy[i].name.find("#")!=std::string::npos);//1 wenn Prof_memcopy[i].name ungueltig 
        if ((Prof_memcopy[i].name=="")||(found))
            std::cout << "Memcopy_Time "<<i;
        else     
            std::cout << Prof_memcopy[i].name;
        std::cout     <<" time: "          << std::fixed << Prof_memcopy[i].time.summ     << "s "
                << "Relative to CPU time: " << std::fixed << Prof_memcopy[i].time.prozent_CPU<< "% "
                << "Relative to GPU time: " << std::fixed << Prof_memcopy[i].time.prozent_GPU<< "% "
                << "Average time: "      << std::fixed << Prof_memcopy[i].time.avg     << "s "
                << "Minimum time: "      << std::fixed << Prof_memcopy[i].time.min     << "s "
                << "Maximum time: "      << std::fixed << Prof_memcopy[i].time.max     << "s "
                << "Standard deviation: "<< std::fixed << Prof_memcopy[i].time.standard     << "s "<< std::endl;

        std::cout     <<" \tMemory: "      << std::fixed << Prof_memcopy[i].memorysize.summ     << "Byte "
                << "Average Memory: "      << std::fixed << Prof_memcopy[i].memorysize.avg     << "Byte "
                << "Minimum Memory: "      << std::fixed << Prof_memcopy[i].memorysize.min     << "Byte "
                << "Maximum Memory: "      << std::fixed << Prof_memcopy[i].memorysize.max     << "Byte "
                << "Standard deviation: "<< std::fixed << Prof_memcopy[i].memorysize.standard     << "Byte "<< std::endl;

        std::cout     <<" \tThroughput: "      << std::fixed << Prof_memcopy[i].memorythroughput.summ     << "Byte/s "
                << "Average Memory: "      << std::fixed << Prof_memcopy[i].memorythroughput.avg     << "Byte/s "
                << "Minimum Memory: "      << std::fixed << Prof_memcopy[i].memorythroughput.min     << "Byte/s "
                << "Maximum Memory: "      << std::fixed << Prof_memcopy[i].memorythroughput.max     << "Byte/s "
                << "Standard deviation: "<< std::fixed << Prof_memcopy[i].memorythroughput.standard     << "Byte/s "<< std::endl;
    }
        std::cout     << std::endl;
    for(int i=0;i<Profiling_time_CPU_count;i++){
        int found = (Prof_time_CPU[i].name.find(":")!=std::string::npos)||(Prof_time_CPU[i].name.find("#")!=std::string::npos);//1 wenn Prof_time_CPU[i].name ungueltig 
        if ((Prof_time_CPU[i].name=="")||(found))
            std::cout << "CPU_Time "<<i;
        else     
            std::cout << Prof_time_CPU[i].name;
        std::cout     <<" time: "          << std::fixed << Prof_time_CPU[i].time.summ     << "s "
                << "Relative to CPU time: " << std::fixed << Prof_time_CPU[i].time.prozent_CPU<< "% "
                << "Average time: "      << std::fixed << Prof_time_CPU[i].time.avg     << "s "
                << "Minimum time: "      << std::fixed << Prof_time_CPU[i].time.min     << "s "
                << "Maximum time: "      << std::fixed << Prof_time_CPU[i].time.max     << "s "
                << "Standard deviation: "<< std::fixed << Prof_time_CPU[i].time.standard     << "s "<< std::endl;
    }
        std::cout     << std::endl;
    for(int i=0;i<Profiling_time_count;i++){
        int found = (Prof_time[i].name.find(":")!=std::string::npos)||(Prof_time[i].name.find("#")!=std::string::npos);//1 wenn Prof_time[i].name ungueltig 
        if ((Prof_time[i].name=="")||(found))
            std::cout << "GPU_Time "<<i;
        else     
            std::cout << Prof_time[i].name;
        std::cout     <<" time: "          << std::fixed << Prof_time[i].time.summ     << "s "
                << "Relative to CPU time: " << std::fixed << Prof_time[i].time.prozent_CPU<< "% "
                << "Relative to GPU time: " << std::fixed << Prof_time[i].time.prozent_GPU<< "% "
                << "Average time: "      << std::fixed << Prof_time[i].time.avg     << "s "
                << "Minimum time: "      << std::fixed << Prof_time[i].time.min     << "s "
                << "Maximum time: "      << std::fixed << Prof_time[i].time.max     << "s "
                << "Standard deviation: "<< std::fixed << Prof_time[i].time.standard     << "s "<< std::endl;
    }
}
int Profiling::evaluate_file(const char * filename)
{
    std::ofstream fp;
    fp.open(filename,std::ios::out|std::ios::trunc);
    if (!(fp.is_open()))return 0;
    fp.precision(8);
    fp <<"#"                                                //Trennzeile 1:CPU Gesammtzeit
     << "Overall time: in s"<< std::endl;
    fp << std::fixed << Prof_general.CPU_summ << std::endl;
    fp <<"#"                                                //Trennzeile 2:GPU Gesammtzeit
     << "On GPU only: in s"<< std::endl;
    fp << std::fixed << Prof_general.GPU_summ << std::endl;
    fp <<"#"                                                //Trennzeile 3:CPU Zeiten
     << "Name:Summe(s):Relative(%):Calls(1):Average(s):Minimum(s):Maximum(s):Standard deviation(s)"<< std::endl;
        for(int i=0;i<Profiling_time_CPU_count;i++){
            int found = (Prof_time_CPU[i].name.find(":")!=std::string::npos)||(Prof_time_CPU[i].name.find("#")!=std::string::npos);//1 wenn Prof_time_CPU[i].name ungueltig 
            if ((Prof_time_CPU[i].name=="")||(found))
                fp << "CPU_Time "<<i;
            else     
                fp << Prof_time_CPU[i].name;

            fp <<           ":" << std::fixed << Prof_time_CPU[i].time.summ 
                    << ":" << std::fixed << Prof_time_CPU[i].time.prozent_CPU
                    << ":" << std::fixed << Prof_time_CPU[i].time.count     
                    << ":" << std::fixed << Prof_time_CPU[i].time.avg     
                    << ":" << std::fixed << Prof_time_CPU[i].time.min     
                    << ":" << std::fixed << Prof_time_CPU[i].time.max     
                    << ":" << std::fixed << Prof_time_CPU[i].time.standard << std::endl;
        }
    fp <<"#"                                                //Trennzeile 4:GPU Zeiten
     << "Name:Summe(s):Relative_CPU(%):Relative_GPU(%):Calls(1):Average(s):Minimum(s):Maximum(s):Standard deviation(s)"<< std::endl;
        for(int i=0;i<Profiling_time_count;i++){
            int found = (Prof_time[i].name.find(":")!=std::string::npos)||(Prof_time[i].name.find("#")!=std::string::npos);//1 wenn Prof_time[i].name ungueltig 
            if ((Prof_time[i].name=="")||(found))
                fp << "GPU_Time "<<i;
            else     
                fp << Prof_time[i].name;

            fp <<           ":" << std::fixed << Prof_time[i].time.summ 
                    << ":" << std::fixed << Prof_time[i].time.prozent_CPU
                    << ":" << std::fixed << Prof_time[i].time.prozent_GPU
                    << ":" << std::fixed << Prof_time[i].time.count     
                    << ":" << std::fixed << Prof_time[i].time.avg     
                    << ":" << std::fixed << Prof_time[i].time.min     
                    << ":" << std::fixed << Prof_time[i].time.max     
                    << ":" << std::fixed << Prof_time[i].time.standard << std::endl;
        }
    fp <<"#"                                                //Trennzeile 5:Initialisierungs Zeiten
     << "Name:Summe(s):Faktor_CPU(1)"<< std::endl;
        for(int i=0;i<Profiling_time_init_count;i++){
            int found = (Prof_time_init[i].name.find(":")!=std::string::npos)||(Prof_time_CPU[i].name.find("#")!=std::string::npos);//1 wenn Prof_time_init[i].name ungueltig 
            if ((Prof_time_CPU[i].name=="")||(found))
                fp << "Initialisation_Time "<<i;
            else     
                fp << Prof_time_init[i].name;

            fp<<        ":"<< std::fixed << Prof_time_init[i].time.summ
                    <<":" << std::fixed << (Prof_time_init[i].time.prozent_CPU/100.0)<< std::endl;
        }
    fp <<"#"                                                //Trennzeile 6:Memcopy
     << "Name:Time summ(s):Relative_CPU(%):Relative_GPU(%):Calls(1):Average(s):Minimum(s):Maximum(s):Standard deviation(s):(Memory)summ(Byte):Average(Byte):Minimum(Byte):Maximum(Byte):Standard deviation(Byte):(Throughput)Average(Byte/s):Minimum(Byte/s):Maximum(Byte/s):Standard deviation(Byte/s)"<< std::endl;
        for(int i=0;i<Profiling_memcopy_count;i++){
            int found = (Prof_memcopy[i].name.find(":")!=std::string::npos)||(Prof_memcopy[i].name.find("#")!=std::string::npos);//1 wenn Prof_time_CPU[i].name ungueltig 
            if ((Prof_memcopy[i].name=="")||(found))
                fp << "Memcopy_Time "<<i;
            else     
                fp << Prof_memcopy[i].name;

            fp <<           ":" << std::fixed << Prof_memcopy[i].time.summ 
                    << ":" << std::fixed << Prof_memcopy[i].time.prozent_CPU
                    << ":" << std::fixed << Prof_memcopy[i].time.prozent_GPU
                    << ":" << std::fixed << Prof_memcopy[i].time.count     
                    << ":" << std::fixed << Prof_memcopy[i].time.avg     
                    << ":" << std::fixed << Prof_memcopy[i].time.min     
                    << ":" << std::fixed << Prof_memcopy[i].time.max     
                    << ":" << std::fixed << Prof_memcopy[i].time.standard
                    << ":" << std::fixed << Prof_memcopy[i].memorysize.summ     
                    << ":" << std::fixed << Prof_memcopy[i].memorysize.avg     
                    << ":" << std::fixed << Prof_memcopy[i].memorysize.min     
                    << ":" << std::fixed << Prof_memcopy[i].memorysize.max     
                    << ":" << std::fixed << Prof_memcopy[i].memorysize.standard
                    << ":" << std::fixed << Prof_memcopy[i].memorythroughput.avg     
                    << ":" << std::fixed << Prof_memcopy[i].memorythroughput.min     
                    << ":" << std::fixed << Prof_memcopy[i].memorythroughput.max     
                    << ":" << std::fixed << Prof_memcopy[i].memorythroughput.standard << std::endl;
        }

    return 1;
}
"""

openmp_profile_body=\
"""
#include "Profiling.h"

// TODO:
"""