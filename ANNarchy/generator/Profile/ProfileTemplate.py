"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

profile_base_template = """#pragma once

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <ostream>
#include <fstream>
#include <iomanip>
#include <math.h>
%(timer_import)s

#ifdef _DEBUG
    #define debug_cout(x) (std::cout << x << std::endl)
#else
    #define debug_cout(x)
#endif

%(measurement_class)s

/**
 *  @brief      Profiling class
 *  @details    Creation and initialization of the class are not thread-safe. 
 *              The class is implemented as singleton ensure uniqueness during
 *              runtime.
 */
extern class Profiling* prof_ptr;
class Profiling {
    std::vector<Measurement*> _datasets;                                ///< Instances of measurement objects. The index for correct access is retrieved from Profiling::_identifier
    std::map<std::pair<std::string, std::string>, int > _identifier;    ///< maps a (obj, func) descriptor to index for Profiling::_datasets
    std::map<std::string, int> _obj_to_ids;                             ///< needed for profile application

    std::ofstream _out_file;
%(timer_start_decl)s

public:

    /**
     *  @brief  Constructor
     */
    Profiling() {
        debug_cout("Create Profiling instance.");

        // HACK: the object constructor is now called by nanobind, need to update reference in C++ library
        prof_ptr = this;

%(timer_init)s

        if (false) {
            _out_file.open("results_%(config)s.xml", std::ofstream::out | std::ofstream::trunc);
            _out_file << "<root>" << std::endl;

        %(config_xml)s
        }
    }

    /**
     *  @brief  Destructor
     */
    ~Profiling() {
        debug_cout("Destroy Profiling instance.");

        for(auto it = _datasets.begin(); it != _datasets.end(); it++ )
            delete *it;

        _out_file << "</root>" << std::endl;
        _out_file.close();
    }

    /**
     *  @brief      Add a function to measurement dataset
     *  @details    A measurment is uniquely described by an object (either population or projection name)
     *              and a function name. The provided items will be used as key for the internal data map.
     *  @param[IN]  obj type as string (either pop, proj or net)
     *  @param[IN]  obj object name as string
     *  @param[IN]  func function name as string
     *  @return     Instance of measurement class. If the function is called multiple times no additional 
     *              object will be created.
     */
    Measurement* register_function(std::string type, std::string obj, int obj_id, std::string func, std::string label) {
        auto pair = std::pair<std::string, std::string>(obj, func);

        if ( _identifier.count(pair) == 0) { // not in list
            _identifier.insert(std::pair< std::pair<std::string, std::string>, int >(pair, _datasets.size()));
            _datasets.push_back(new Measurement(type, label));

            debug_cout( "(" + pair.first + ", " + pair.second + ") added to dataset." );
        }

        _obj_to_ids[obj] = obj_id;
        return get_measurement(obj, func);
    }

    /**
     *  @brief      Retrieve registered function from measurement dataset
     *  @details    A measurment is uniquely described by an object (either population or projection name)
     *              and a function name. The provided items will be used as key for the internal data map.
     *  @param[IN]  obj object name as string
     *  @param[IN]  func function name as string
     *  @return     Instance of measurement class. If the measurment were not registered before, no additional 
     *              object will be created. The function returns a 'nullptr' in this case.
     */
    Measurement* get_measurement(const std::string &obj, const std::string &func) {

        auto idx = _identifier.find(std::pair<std::string, std::string>(obj, func));
        if ( idx == _identifier.end() )
            return nullptr;

        return _datasets[idx->second];
    }

    /**
     *  @brief      Clear performance data of registered functions.
     *  @details    A measurment is uniquely described by an object (either population or projection name)
     *              and a function name. We iterate over all datasets and clear them seperatly.
     */
    void reset() {
        for( auto it = _datasets.begin(); it != _datasets.end(); it++ )
            (*it)->reset();
    }

    /**
     *  @brief      Evaluate registered functions.
     *  @details    A measurment is uniquely described by an object (either population or projection name)
     *              and a function name. We iterate over all datasets and evaluate them seperatly.
     */
    void evaluate() {
        for( auto it = _datasets.begin(); it != _datasets.end(); it++ )
            (*it)->evaluate();
    }

    void store() {
        //
        // To ensure, that the network related datasets are written first, we run two times
        // across the list ...
        //
        for( auto it = _identifier.begin(); it != _identifier.end(); it++ ) {
            if ( _datasets[it->second]->_raw_data.empty() || (_datasets[it->second]->_type.compare("net")!=0 ) )
                continue;    // nothing recorded, omit dataset

            _out_file << "  <dataset>" << std::endl;
            _out_file << "    <obj_type>" << _datasets[it->second]->_type << "</obj_type>" << std::endl;
            _out_file << "    <name>" << it->first.first << "</name>" << std::endl;
            _out_file << "    <func>" << it->first.second << "</func>" << std::endl;
            _out_file << "    <label>" << it->first.second << "</label>" << std::endl; // for top level objects "label" is equal to "func"
            _out_file << "    <mean>" << std::fixed << std::setprecision(4) << _datasets[it->second]->_mean << "</mean>"<< std::endl;
            _out_file << "    <std>" << std::fixed << std::setprecision(4) << _datasets[it->second]->_std << "</std>"<< std::endl;

            _out_file << "    <raw_data>";
            for(auto it2 = _datasets[it->second]->_raw_data.begin(); it2 != _datasets[it->second]->_raw_data.end(); it2++)
                _out_file << std::fixed << std::setprecision(4) << *it2 << " ";
            _out_file << "</raw_data>" << std::endl;

            _out_file << "  </dataset>" << std::endl;
        }

        for( auto it = _identifier.begin(); it != _identifier.end(); it++ ) {
            if ( _datasets[it->second]->_raw_data.empty() || (_datasets[it->second]->_type.compare("net")==0 ) )
                continue;    // nothing recorded, omit dataset

            _out_file << "  <dataset>" << std::endl;
            _out_file << "    <obj_type>" << _datasets[it->second]->_type << "</obj_type>" << std::endl;
            _out_file << "    <name>" << it->first.first << "</name>" << std::endl;
            _out_file << "    <func>" << it->first.second << "</func>" << std::endl;
            _out_file << "    <label>" << _datasets[it->second]->_label << "</label>" << std::endl;
            _out_file << "    <mean>" << std::fixed << std::setprecision(4) << _datasets[it->second]->_mean << "</mean>"<< std::endl;
            _out_file << "    <std>" << std::fixed << std::setprecision(4) << _datasets[it->second]->_std << "</std>"<< std::endl;

            _out_file << "    <raw_data>";
            for(auto it2 = _datasets[it->second]->_raw_data.begin(); it2 != _datasets[it->second]->_raw_data.end(); it2++)
                _out_file << std::fixed << std::setprecision(4) << *it2 << " ";
            _out_file << "</raw_data>" << std::endl;

            _out_file << "  </dataset>" << std::endl;
        }
    }

    /*
     *  Exported to Python
     */
    double get_avg_time(std::string obj, std::string func) {
        auto dataset = get_measurement(obj, func);
        if (dataset == nullptr)
            return 0.0;

        return dataset->_mean;
    }

    double get_std_time(std::string obj, std::string func) {
        auto dataset = get_measurement(obj, func);
        if (dataset == nullptr)
            return 0.0;

        return dataset->_std;
    }

    friend std::ostream& operator << (std::ostream& stream, const Profiling& profiling);
};

// out stream operators
inline std::ostream& operator << (std::ostream& stream, const Measurement& measure) {
    if ( measure._raw_data.empty() )
        return stream;

    stream << "mean: " << measure._mean << " ms, "
           << "std: " << measure._std << " ( over " << measure._raw_data.size() << " measurements )";

    return stream;
}

inline std::ostream& operator << (std::ostream& stream, const Profiling& profiling) {
    if ( profiling._datasets.empty() )
        return stream;

    for ( auto it = profiling._identifier.begin(); it != profiling._identifier.end(); it++ ) {
        stream << "  (" << it->first.first << ", " << it->first.second << ") - " << *(profiling._datasets[ it->second ]) << std::endl;
    }

    return stream;
}
"""

cpp11_profile_header = """
class Measurement{
    std::string _type;
    std::string _label;
    std::vector<double> _raw_data;

    double _mean;
    double _std;
    std::chrono::time_point<std::chrono::steady_clock> _start;
    std::chrono::time_point<std::chrono::steady_clock> _stop;

public:
    Measurement(std::string type, std::string label) {
        debug_cout("Create Measurement object");
        _type = type;
        _label = label;
        _mean = 0.0;
        _std = 0.0;
        _raw_data = std::vector<double>();
    }

    ~Measurement() {
        debug_cout("Destroy Measurement object");
        _raw_data.clear();
    }

    inline void start_wall_time() {
        _start = std::chrono::steady_clock::now();
    }

    inline void stop_wall_time() {
        _stop = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> dur = _stop - _start;
        // The measured times are stored in seconds
        _raw_data.push_back(std::chrono::duration<double>(dur).count());
    }

    void reset() {
        debug_cout("Reset Measurement object");
        _raw_data.clear();
        _mean = 0.0;
        _std = 0.0;
    }

    void evaluate() {
        if (_raw_data.empty())
            return;

        double sum = 0.0;
        int num_elem = _raw_data.size();

        // mean over all times
        for( auto it = _raw_data.begin(); it != _raw_data.end(); it++ ) {
            sum += *it;
        }

        _mean = double(sum) / double(num_elem);

        // variance over all times
        double var = 0;
        for( auto it = _raw_data.begin(); it != _raw_data.end(); it++ ) {
            var += (*it-_mean)*(*it-_mean);
        }

        _std = sqrt(var/double(num_elem));
    }

    friend std::ostream& operator << (std::ostream& stream, const Measurement& measure);
    friend class Profiling;
};
"""

cpp11_profile_template = {
    'include': """// Profiling
#include "Profiling.hpp"
extern class Profiling* prof_ptr;
""",
    'init': """
    //initialize profiler, create singleton instance
    prof_ptr->register_function("net", "network", 0, "step", "overall");
    prof_ptr->register_function("net", "network", 0, "psp", "overall");
    prof_ptr->register_function("net", "network", 0, "proj_step", "overall");
    prof_ptr->register_function("net", "network", 0, "proj_post_event", "overall");
    prof_ptr->register_function("net", "network", 0, "neur_step", "overall");
    prof_ptr->register_function("net", "network", 0, "global_op", "overall");
    prof_ptr->register_function("net", "network", 0, "record", "overall");
    prof_ptr->register_function("net", "network", 0, "rng", "overall");
    """,
    # Operations
    'proj_psp_pre': """// measure synaptic transmission
    auto measure_psp = prof_ptr->get_measurement("network", "psp");
    measure_psp->start_wall_time();
    """,
    'proj_psp_post': """// done
    measure_psp->stop_wall_time();
    """,
    'proj_step_pre': """// measure synaptic update
    auto measure_proj_step = prof_ptr->get_measurement("network", "proj_step");
    measure_proj_step->start_wall_time();
    """,
    'proj_step_post': """// done
    measure_proj_step->stop_wall_time();
    """,
    'proj_post_event_pre': """// measure post-event update
    auto measure_proj_post_event_step = prof_ptr->get_measurement("network", "proj_post_event");
    measure_proj_step->start_wall_time();
    """,
    'proj_post_event_post': """// done
    measure_proj_post_event_step->stop_wall_time();
    """,
    'neur_step_pre': """// measure population update
    auto measure_neur_step = prof_ptr->get_measurement("network", "neur_step");
    measure_neur_step->start_wall_time();
    """,
    'neur_step_post': """// done
    measure_neur_step->stop_wall_time();
    """,
    # Global operations
    'global_op_pre': """// measure global operations
    auto measure_global_op = prof_ptr->get_measurement("network", "global_op");
    measure_global_op->start_wall_time();
    """,
    'global_op_post': """// done
    measure_global_op->stop_wall_time();
    """,
    # Record
    'record_pre': """// measure record
    auto measure_rec = prof_ptr->get_measurement("network", "record");
    measure_rec->start_wall_time();
    """,
    'record_post': """// done
    measure_rec->stop_wall_time();
    """,
    # RNG
    'rng_pre': """// measure update rng
    auto measure_rng = prof_ptr->get_measurement("network", "rng");
    measure_rng->start_wall_time();
    """,
    'rng_post': """// done
    measure_rng->stop_wall_time();
    """,
    # Overall and setup
    'step_pre': """// before
    auto measure = prof_ptr->get_measurement("network", "step");
    measure->start_wall_time();
    """,
    'step_post': """// after
    measure->stop_wall_time();
    """,
    'run_pre': """
    prof_ptr->reset();
    """,
    'run_post': """
    prof_ptr->evaluate();
    if (false) {
        prof_ptr->store();
        std::cout << "profiling results: " << std::endl;
        std::cout << *prof_ptr << std::endl;
    }
    """,

    #
    # Execute the profile in each Object (i. e. populations, projections)
    'compute_psp': {
        'before' : "measure_psp->start_wall_time();",
        'after' : "measure_psp->stop_wall_time();"
    },
    'update_synapse': {
        'before' : "measure_step->start_wall_time();",
        'after' : "measure_step->stop_wall_time();"
    },
    'post_event': {
        'before' : "measure_pe->start_wall_time();",
        'after' : "measure_pe->stop_wall_time();"
    },
    'update_neuron': {
        'before' : "measure_step->start_wall_time();",
        'after' : "measure_step->stop_wall_time();"
    },
    'update_rng':{
        'before' : "measure_rng->start_wall_time();",
        'after' : "measure_rng->stop_wall_time();"
    },
    'update_delay':{
        'before' : "measure_delay->start_wall_time();",
        'after' : "measure_delay->stop_wall_time();"
    },
    'spike_gather': {
        'before' : "measure_sc->start_wall_time();",
        'after' : "measure_sc->stop_wall_time();"
    },
    'spike_prop': {
        'before' : "measure_prop->start_wall_time();",
        'after' : "measure_prop->stop_wall_time();"
    }
}

cpp11_omp_profile_template = {
    'include': """// Profiling
#include "Profiling.hpp"
extern class Profiling* prof_ptr;
""",
    'init': """
    //initialize profiler, create singleton instance
    prof_ptr->register_function("net", "network", 0, "step", "overall");
    prof_ptr->register_function("net", "network", 0, "psp", "overall");
    prof_ptr->register_function("net", "network", 0, "proj_step", "overall");
    prof_ptr->register_function("net", "network", 0, "proj_post_event", "overall");    
    prof_ptr->register_function("net", "network", 0, "neur_step", "overall");
    prof_ptr->register_function("net", "network", 0, "global_op", "overall");
    prof_ptr->register_function("net", "network", 0, "record", "overall");
    prof_ptr->register_function("net", "network", 0, "rng", "overall");
    """,
    # Operations
    'proj_psp_pre': """// measure synaptic transmission
    auto measure_psp = prof_ptr->get_measurement("network", "psp");
    #pragma omp barrier
    #pragma omp master
    measure_psp->start_wall_time();
    """,
    'proj_psp_post': """// done
    #pragma omp barrier
    #pragma omp master
    measure_psp->stop_wall_time();
    """,
    'proj_step_pre': """// measure synaptic update
    auto measure_proj_step = prof_ptr->get_measurement("network", "proj_step");
    #pragma omp barrier
    #pragma omp master
    measure_proj_step->start_wall_time();
    """,
    'proj_step_post': """// done
    #pragma omp barrier
    #pragma omp master
    measure_proj_step->stop_wall_time();
    """,
    'proj_post_event_pre': """// measure post-event update
    auto measure_proj_post_event_step = prof_ptr->get_measurement("network", "proj_post_event");
    #pragma omp barrier
    #pragma omp master
    measure_proj_step->start_wall_time();
    """,
    'proj_post_event_post': """// done
    #pragma omp barrier
    #pragma omp master
    measure_proj_post_event_step->stop_wall_time();
    """,
    'neur_step_pre': """// measure population update
    auto measure_neur_step = prof_ptr->get_measurement("network", "neur_step");
    #pragma omp barrier
    #pragma omp master
    measure_neur_step->start_wall_time();
    """,
    'neur_step_post': """// done
    #pragma omp barrier
    #pragma omp master
    measure_neur_step->stop_wall_time();
    """,
    # Global operations
    'global_op_pre': """// measure global operations
    auto measure_global_op = prof_ptr->get_measurement("network", "global_op");
    #pragma omp barrier
    #pragma omp master
    measure_global_op->start_wall_time();
    """,
    'global_op_post': """// done
    #pragma omp barrier
    #pragma omp master
    measure_global_op->stop_wall_time();
    """,
    # Record
    'record_pre': """// measure record
    auto measure_rec = prof_ptr->get_measurement("network", "record");
    #pragma omp barrier
    #pragma omp master
    measure_rec->start_wall_time();
    """,
    'record_post': """// done
    #pragma omp barrier
    #pragma omp master
    measure_rec->stop_wall_time();
    """,
    # RNG
    'rng_pre': """// measure update rng
    auto measure_rng = prof_ptr->get_measurement("network", "rng");
    #pragma omp master
    measure_rng->start_wall_time();
    """,
    'rng_post': """// done
    #pragma omp master
    measure_rng->stop_wall_time();
    """,

    # Overall and setup
    'step_pre': """// before
    auto measure = prof_ptr->get_measurement("network", "step");
    #pragma omp master
    measure->start_wall_time();
    """,
    'step_post': """// after
    #pragma omp master
    measure->stop_wall_time();
    """,
    'run_pre': """
    prof_ptr->reset();
    """,
    'run_post': """
    prof_ptr->evaluate();
    if (false) {
        prof_ptr->store();
        std::cout << "profiling results: " << std::endl;
        std::cout << *prof_ptr << std::endl;
    }
    """,

    #
    # Execute the profile in each Object (i. e. populations, projections)
    'compute_psp': {
        'before' : "#pragma omp barrier\n#pragma omp master\nmeasure_psp->start_wall_time();",
        'after' : "#pragma omp barrier\n#pragma omp master\nmeasure_psp->stop_wall_time();"
    },
    'update_synapse': {
        'before' : "#pragma omp barrier\n#pragma omp master\nmeasure_step->start_wall_time();",
        'after' : "#pragma omp barrier\n#pragma omp master\nmeasure_step->stop_wall_time();"
    },
    'post_event': {
        'before' : "#pragma omp barrier\n#pragma omp master\nmeasure_pe->start_wall_time();",
        'after' : "#pragma omp barrier\n#pragma omp master\nmeasure_pe->stop_wall_time();"
    },
    'update_neuron': {
        'before' : "#pragma omp barrier\n#pragma omp master\nmeasure_step->start_wall_time();",
        'after' : "#pragma omp barrier\n#pragma omp master\nmeasure_step->stop_wall_time();"
    },
    'update_rng':{
        'before' : "#pragma omp barrier\n#pragma omp master\nmeasure_rng->start_wall_time();",
        'after' : "#pragma omp barrier\n#pragma omp master\nmeasure_rng->stop_wall_time();"
    },
    'update_delay':{
        'before' : "#pragma omp barrier\n#pragma omp master\nmeasure_delay->start_wall_time();",
        'after' : "#pragma omp barrier\n#pragma omp master\nmeasure_delay->stop_wall_time();"
    },
    'spike_gather': {
        'before' : "#pragma omp barrier\n#pragma omp master\nmeasure_sc->start_wall_time();",
        'after' : "#pragma omp barrier\n#pragma omp master\nmeasure_sc->stop_wall_time();"
    },
    'spike_prop': {
        'before' : "#pragma omp barrier\n#pragma omp master\nmeasure_prop->start_wall_time();",
        'after' : "#pragma omp barrier\n#pragma omp master\nmeasure_prop->stop_wall_time();"
    }
}

cuda_profile_header = \
"""
class Measurement{
    std::string _type;
    std::string _label;
    std::vector<double> _raw_data;

    double _mean;
    double _std;
    cudaEvent_t _start;
    cudaEvent_t _stop;

public:
    Measurement(std::string type, std::string label) {
        debug_cout("Create Measurement object");
        _type = type;
        _label = label;
        _mean = 0.0;
        _std = 0.0;
        _raw_data = std::vector<double>();

        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
    }

    ~Measurement() {
        debug_cout("Destroy Measurement object");
        _raw_data.clear();

        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
    }

    inline void start_wall_time() {
        cudaEventRecord(_start);
    }

    inline void stop_wall_time() {
        cudaEventRecord(_stop);

        cudaEventSynchronize(_stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, _start, _stop);

         // The measured times are stored in seconds
        _raw_data.push_back(double(milliseconds/1000.0));
    }

    void reset() {
        debug_cout("Reset Measurement object");
        _raw_data.clear();
        _mean = 0.0;
        _std = 0.0;
    }

    void evaluate() {
        if (_raw_data.empty())
            return;

        double sum = 0.0;
        int num_elem = _raw_data.size();

        // mean over all times
        for( auto it = _raw_data.begin(); it != _raw_data.end(); it++ ) {
            sum += *it;
        }

        _mean = double(sum) / double(num_elem);

        // variance over all times
        double var = 0;
        for( auto it = _raw_data.begin(); it != _raw_data.end(); it++ ) {
            var += (*it-_mean)*(*it-_mean);
        }

        _std = sqrt(var/double(num_elem));
    }

    friend std::ostream& operator << (std::ostream& stream, const Measurement& measure);
    friend class Profiling;
};
"""

cuda_profile_template = {
    'include': """// Profiling
#include "Profiling.hpp"
extern class Profiling* prof_ptr;
""",
    'init': """
    //initialize profiler, create singleton instance
    prof_ptr->register_function("net", "network", 0, "step", "overall");
    prof_ptr->register_function("net", "network", 0, "psp", "overall");
    prof_ptr->register_function("net", "network", 0, "proj_step", "overall");
    prof_ptr->register_function("net", "network", 0, "proj_post_event", "overall");
    prof_ptr->register_function("net", "network", 0, "neur_step", "overall");
    prof_ptr->register_function("net", "network", 0, "record", "overall");
    """,
    # Operations
    'proj_psp_pre': """// measure synaptic transmission
    auto measure_psp = prof_ptr->get_measurement("network", "psp");
    measure_psp->start_wall_time();
    """,
    'proj_psp_post': """// done
    measure_psp->stop_wall_time();
    """,
    'proj_step_pre': """// measure synaptic transmission
    auto measure_proj_step = prof_ptr->get_measurement("network", "proj_step");
    measure_proj_step->start_wall_time();
    """,
    'proj_step_post': """// done
    measure_proj_step->stop_wall_time();
    """,
    'neur_step_pre': """// measure population update
    auto measure_neur_step = prof_ptr->get_measurement("network", "neur_step");
    measure_neur_step->start_wall_time();
    """,
    'neur_step_post': """// done
    measure_neur_step->stop_wall_time();
    """,

    # Overall and setup
    'step_pre': """// before
    auto measure = prof_ptr->get_measurement("network", "step");
    measure->start_wall_time();
    """,
    'step_post': """// after
    measure->stop_wall_time();
    """,
    'run_pre': """
    prof_ptr->reset();
    """,
    'run_post': """
    prof_ptr->evaluate();
    if (false) {
        prof_ptr->store();
        std::cout << "profiling results: " << std::endl;
        std::cout << *prof_ptr << std::endl;
    }
    """,
    # Record
    'record_pre': """// measure record
    auto measure_rec = prof_ptr->get_measurement("network", "record");
    measure_rec->start_wall_time();
    """,
    'record_post': """// done
    measure_rec->stop_wall_time();
    """,
    #
    # Operations
    'compute_psp': {
        'before' : "proj%(id)s->measure_psp->start_wall_time();",
        'after' : "proj%(id)s->measure_psp->stop_wall_time();"
    },
    'update_synapse': {
        'before' : "proj%(id)s->measure_step->start_wall_time();",
        'after' : "proj%(id)s->measure_step->stop_wall_time();"
    },
    'post_event': {
        'before' : "proj%(id)s->measure_pe->start_wall_time();",
        'after' : "proj%(id)s->measure_pe->stop_wall_time();"
    },
    'update_neuron': {
        'before' : "pop%(id)s->measure_step->start_wall_time();",
        'after' : "pop%(id)s->measure_step->stop_wall_time();"
    },
    'spike_gather': {
        'before' : "pop%(id)s->measure_gather->start_wall_time();",
        'after' : "pop%(id)s->measure_gather->stop_wall_time();"
    },
    'spike_prop': {
        'before' : "measure_prop->start_wall_time();",
        'after' : "measure_prop->stop_wall_time();"
    },
}
