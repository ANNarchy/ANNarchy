#===============================================================================
#
#     ProfileTemplate.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2016-2018  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================papi_profile_header=\
papi_profile_header = """
#pragma once

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <ostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <papi.h>

#ifdef _DEBUG
    #define debug_cout(x) (std::cout << x << std::endl)
#else
    #define debug_cout(x)
#endif

class Measurement{
    std::string _type;
    std::vector<double> _raw_data;

    double _mean;
    double _std;
    long_long _start;
    long_long _stop;

public:
    Measurement(std::string type) {
        debug_cout("Create Measurement object");
        _type = type;
        _mean = 0.0;
        _std = 0.0;
        _start = 0.0;
        _stop = 0.0;
        _raw_data = std::vector<double>();
    }

    ~Measurement() {
        debug_cout("Destroy Measurement object");
        _raw_data.clear();
    }

    inline void start_wall_time() {
        _start = PAPI_get_real_usec();
    }

    inline void stop_wall_time() {
        _stop = PAPI_get_real_usec();
        _raw_data.push_back(double(_stop-_start));
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

        long_long sum = 0.0;
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

/**
 *  @brief      Profiling class
 *  @details    Creation and initialization of the class are not thread-safe. 
 *              The class is implemented as singleton ensure uniqueness during
 *              runtime.
 */
class Profiling {
    static std::unique_ptr<Profiling> _instance;    ///< reference to this class, created on first call of Profiling::get_instance()
    std::vector<Measurement*> _datasets;    ///< Instances of measurement objects. The index for correct access is retrieved from Profiling::_identifier
    std::map<std::pair<std::string, std::string>, int > _identifier;    ///< maps a (obj, func) descriptor to index for Profiling::_datasets
    std::ofstream _out_file;
    long_long _profiler_start;

    /**
     *  @brief  Constructor
     */
    Profiling() {
        debug_cout("Create Profiling instance.");

        // initialize PAPI
        if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
            exit(1);

        _profiler_start = PAPI_get_real_usec();
        _out_file.open("%(result_file)s", std::ofstream::out | std::ofstream::trunc);
        _out_file << "<root>" << std::endl;

        %(config_xml)s
    }

public:
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
     *  @brief      Retrieve Profiling class instance
     *  @details    First call initialize the class
     *  @return     Reference to Profiling class for further interaction
     */
    static Profiling* get_instance() {
        if ( _instance.get() == nullptr )
            _instance = std::unique_ptr<Profiling>(new Profiling());

        return _instance.get();
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
    Measurement* register_function(std::string type, std::string obj, std::string func) {
        auto pair = std::pair<std::string, std::string>(obj, func);

        if ( _identifier.count(pair) == 0) { // not in list
            _identifier.insert(std::pair< std::pair<std::string, std::string>, int >(pair, _datasets.size()));
            _datasets.push_back(new Measurement(type));

            debug_cout( "(" + pair.first + ", " + pair.second + ") added to dataset." );
        }

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
    Measurement* get_measurement(std::string obj, std::string func) {

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
            _out_file << "    <mean>" << std::fixed << std::setprecision(4) << _datasets[it->second]->_mean << "</mean>"<< std::endl;
            _out_file << "    <std>" << std::fixed << std::setprecision(4) << _datasets[it->second]->_std << "</std>"<< std::endl;

            _out_file << "    <raw_data>";
            for(auto it2 = _datasets[it->second]->_raw_data.begin(); it2 != _datasets[it->second]->_raw_data.end(); it2++)
                _out_file << std::fixed << std::setprecision(4) << *it2 << " ";
            _out_file << "</raw_data>" << std::endl;

            _out_file << "  </dataset>" << std::endl;
        }
    }

    friend std::ostream& operator << (std::ostream& stream, const Profiling& profiling);
};

// out stream operators
inline std::ostream& operator << (std::ostream& stream, const Measurement& measure) {
    if ( measure._raw_data.empty() )
        return stream;

    stream << "mean: " << measure._mean << " us, "
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

papi_profile_template = {
    'include': """// Profiling
#include "Profiling.h"
std::unique_ptr<Profiling> Profiling::_instance(nullptr);
""",
    'init': """
    //initialize profiler, create singleton instance
    auto profiler = Profiling::get_instance();
    profiler->register_function("net", "network", "step");
    profiler->register_function("net", "network", "psp");
    profiler->register_function("net", "network", "neur_step");
    """,
    # Operations
    'proj_psp_pre': """// measure synaptic transmission
    auto measure_psp = Profiling::get_instance()->get_measurement("network", "psp");
    measure_psp->start_wall_time();
    """,
    'proj_psp_post': """// done
    measure_psp->stop_wall_time();
    """,
    'neur_step_pre': """// measure population update
    auto measure_neur_step = Profiling::get_instance()->get_measurement("network", "neur_step");
    measure_neur_step->start_wall_time();
    """,
    'neur_step_post': """// done
    measure_neur_step->stop_wall_time();
    """,

    # Overall and setup
    'step_pre': """// before
    auto measure = Profiling::get_instance()->get_measurement("network", "step");
    measure->start_wall_time();
    """,
    'step_post': """// after
    measure->stop_wall_time();
    """,
    'run_pre': """
    Profiling::get_instance()->reset();
    """,
    'run_post': """
    Profiling::get_instance()->evaluate();
    Profiling::get_instance()->store();
    std::cout << "profiling results: " << std::endl;
    std::cout << *Profiling::get_instance() << std::endl;
    """,
    #
    # Operations
    'compute_psp': {
        'before' : "measure_psp->start_wall_time();",
        'after' : "measure_psp->stop_wall_time();"
    },
    'update_synapse': {
        'before' : "measure_step->start_wall_time();",
        'after' : "measure_step->stop_wall_time();"
    },
    'update_neuron': {
        'before' : "measure_step->start_wall_time();",
        'after' : "measure_step->stop_wall_time();"
    },
    'spike_prop': {
        'before' : "measure_prop->start_wall_time();",
        'after' : "measure_prop->stop_wall_time();"
    }
}

cpp11_profile_header = """
#pragma once

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <ostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <chrono>

#ifdef _DEBUG
    #define debug_cout(x) (std::cout << x << std::endl)
#else
    #define debug_cout(x)
#endif

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
        std::chrono::duration<float> dur = _stop - _start;
        _raw_data.push_back( dur.count() * 1000 * 1000); // duration is in sec
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

/**
 *  @brief      Profiling class
 *  @details    Creation and initialization of the class are not thread-safe. 
 *              The class is implemented as singleton ensure uniqueness during
 *              runtime.
 */
class Profiling {
    static std::unique_ptr<Profiling> _instance;    ///< reference to this class, created on first call of Profiling::get_instance()
    std::vector<Measurement*> _datasets;    ///< Instances of measurement objects. The index for correct access is retrieved from Profiling::_identifier
    std::map<std::pair<std::string, std::string>, int > _identifier;    ///< maps a (obj, func) descriptor to index for Profiling::_datasets
    std::map<std::string, int> _obj_to_ids; ///< needed for profile application

    std::ofstream _out_file;
    std::chrono::time_point<std::chrono::steady_clock> _profiler_start;

    /**
     *  @brief  Constructor
     */
    Profiling() {
        debug_cout("Create Profiling instance.");

        _profiler_start = std::chrono::steady_clock::now();
        _out_file.open("%(result_file)s", std::ofstream::out | std::ofstream::trunc);
        _out_file << "<root>" << std::endl;

        %(config_xml)s
    }

public:
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
     *  @brief      Retrieve Profiling class instance
     *  @details    First call initialize the class
     *  @return     Reference to Profiling class for further interaction
     */
    static Profiling* get_instance() {
        if ( _instance.get() == nullptr )
            _instance = std::unique_ptr<Profiling>(new Profiling());

        return _instance.get();
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
    Measurement* get_measurement(std::string obj, std::string func) {

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

    friend std::ostream& operator << (std::ostream& stream, const Profiling& profiling);
};

// out stream operators
inline std::ostream& operator << (std::ostream& stream, const Measurement& measure) {
    if ( measure._raw_data.empty() )
        return stream;

    stream << "mean: " << measure._mean << " us, "
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

cpp11_profile_template = {
    'include': """// Profiling
#include "Profiling.h"
std::unique_ptr<Profiling> Profiling::_instance(nullptr);
""",
    'init': """
    //initialize profiler, create singleton instance
    auto profiler = Profiling::get_instance();
    profiler->register_function("net", "network", 0, "step", "overall");
    profiler->register_function("net", "network", 0, "psp", "overall");
    profiler->register_function("net", "network", 0, "proj_step", "overall");
    profiler->register_function("net", "network", 0, "neur_step", "overall");
    profiler->register_function("net", "network", 0, "record", "overall");
    profiler->register_function("net", "network", 0, "rng", "overall");
    """,
    # Operations
    'proj_psp_pre': """// measure synaptic transmission
    auto measure_psp = Profiling::get_instance()->get_measurement("network", "psp");
    measure_psp->start_wall_time();
    """,
    'proj_psp_post': """// done
    measure_psp->stop_wall_time();
    """,
    'proj_step_pre': """// measure synaptic update
    auto measure_proj_step = Profiling::get_instance()->get_measurement("network", "proj_step");
    measure_proj_step->start_wall_time();
    """,
    'proj_step_post': """// done
    measure_proj_step->stop_wall_time();
    """,
    'neur_step_pre': """// measure population update
    auto measure_neur_step = Profiling::get_instance()->get_measurement("network", "neur_step");
    measure_neur_step->start_wall_time();
    """,
    'neur_step_post': """// done
    measure_neur_step->stop_wall_time();
    """,
    # Record
    'record_pre': """// measure record
    auto measure_rec = Profiling::get_instance()->get_measurement("network", "record");
    measure_rec->start_wall_time();
    """,
    'record_post': """// done
    measure_rec->stop_wall_time();
    """,
    # RNG
    'rng_pre': """// measure update rng
    auto measure_rng = Profiling::get_instance()->get_measurement("network", "rng");
    measure_rng->start_wall_time();
    """,
    'rng_post': """// done
    measure_rng->stop_wall_time();
    """,

    # Overall and setup
    'step_pre': """// before
    auto measure = Profiling::get_instance()->get_measurement("network", "step");
    measure->start_wall_time();
    """,
    'step_post': """// after
    measure->stop_wall_time();
    """,
    'run_pre': """
    Profiling::get_instance()->reset();
    """,
    'run_post': """
    Profiling::get_instance()->evaluate();
    Profiling::get_instance()->store();
    std::cout << "profiling results: " << std::endl;
    std::cout << *Profiling::get_instance() << std::endl;
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
    'update_neuron': {
        'before' : "measure_step->start_wall_time();",
        'after' : "measure_step->stop_wall_time();"
    },
    'update_rng':{
        'before' : "measure_rng->start_wall_time();",
        'after' : "measure_rng->stop_wall_time();"
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

cuda_profile_header = \
"""
#pragma once

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <ostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <cuda_runtime_api.h>

#ifdef _DEBUG
    #define debug_cout(x) (std::cout << x << std::endl)
#else
    #define debug_cout(x)
#endif

class Measurement{
    std::string _type;
    std::vector<double> _raw_data;

    double _mean;
    double _std;
    cudaEvent_t _start;
    cudaEvent_t _stop;

public:
    Measurement(std::string type) {
        debug_cout("Create Measurement object");
        _type = type;
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

        _raw_data.push_back(double(milliseconds*1000.0)); // storage in us
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

/**
 *  @brief      Profiling class
 *  @details    Creation and initialization of the class are not thread-safe. 
 *              The class is implemented as singleton ensure uniqueness during
 *              runtime.
 */
class Profiling {
    static std::unique_ptr<Profiling> _instance;    ///< reference to this class, created on first call of Profiling::get_instance()
    std::vector<Measurement*> _datasets;    ///< Instances of measurement objects. The index for correct access is retrieved from Profiling::_identifier
    std::map<std::pair<std::string, std::string>, int > _identifier;    ///< maps a (obj, func) descriptor to index for Profiling::_datasets
    std::ofstream _out_file;
    cudaEvent_t _profiler_start;

    /**
     *  @brief  Constructor
     */
    Profiling() {
        debug_cout("Create Profiling instance.");

        cudaEventCreate(&_profiler_start);
        cudaEventRecord(_profiler_start);

        _out_file.open("results_%(config)s.xml", std::ofstream::out | std::ofstream::trunc);
        _out_file << "<root>" << std::endl;

        %(config_xml)s
    }

public:
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
     *  @brief      Retrieve Profiling class instance
     *  @details    First call initialize the class
     *  @return     Reference to Profiling class for further interaction
     */
    static Profiling* get_instance() {
        if ( _instance.get() == nullptr )
            _instance = std::unique_ptr<Profiling>(new Profiling());

        return _instance.get();
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
    Measurement* register_function(std::string type, std::string obj, std::string func) {
        auto pair = std::pair<std::string, std::string>(obj, func);

        if ( _identifier.count(pair) == 0) { // not in list
            _identifier.insert(std::pair< std::pair<std::string, std::string>, int >(pair, _datasets.size()));
            _datasets.push_back(new Measurement(type));

            debug_cout( "(" + pair.first + ", " + pair.second + ") added to dataset." );
        }

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
    Measurement* get_measurement(std::string obj, std::string func) {

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
            _out_file << "    <mean>" << std::fixed << std::setprecision(4) << _datasets[it->second]->_mean << "</mean>"<< std::endl;
            _out_file << "    <std>" << std::fixed << std::setprecision(4) << _datasets[it->second]->_std << "</std>"<< std::endl;

            _out_file << "    <raw_data>";
            for(auto it2 = _datasets[it->second]->_raw_data.begin(); it2 != _datasets[it->second]->_raw_data.end(); it2++)
                _out_file << std::fixed << std::setprecision(4) << *it2 << " ";
            _out_file << "</raw_data>" << std::endl;

            _out_file << "  </dataset>" << std::endl;
        }
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

cuda_profile_template = {
    'include': """// Profiling
#include "Profiling.h"
std::unique_ptr<Profiling> Profiling::_instance(nullptr);
""",
    'init': """
    //initialize profiler, create singleton instance
    auto profiler = Profiling::get_instance();
    profiler->register_function("net", "network", "step");
    profiler->register_function("net", "network", "psp");
    profiler->register_function("net", "network", "proj_step");
    profiler->register_function("net", "network", "neur_step");
    """,
    # Operations
    'proj_psp_pre': """// measure synaptic transmission
    auto measure_psp = Profiling::get_instance()->get_measurement("network", "psp");
    measure_psp->start_wall_time();
    """,
    'proj_psp_post': """// done
    measure_psp->stop_wall_time();
    """,
    'proj_step_pre': """// measure synaptic transmission
    auto measure_proj_step = Profiling::get_instance()->get_measurement("network", "proj_step");
    measure_proj_step->start_wall_time();
    """,
    'proj_step_post': """// done
    measure_proj_step->stop_wall_time();
    """,
    'neur_step_pre': """// measure population update
    auto measure_neur_step = Profiling::get_instance()->get_measurement("network", "neur_step");
    measure_neur_step->start_wall_time();
    """,
    'neur_step_post': """// done
    measure_neur_step->stop_wall_time();
    """,

    # Overall and setup
    'step_pre': """// before
    auto measure = Profiling::get_instance()->get_measurement("network", "step");
    measure->start_wall_time();
    """,
    'step_post': """// after
    measure->stop_wall_time();
    """,
    'run_pre': """
    Profiling::get_instance()->reset();
    """,
    'run_post': """
    Profiling::get_instance()->evaluate();
    Profiling::get_instance()->store();
    std::cout << "profiling results: " << std::endl;
    std::cout << *Profiling::get_instance() << std::endl;
    """,
    #
    # Operations
    'compute_psp': {
        'before' : "proj%(id)s.measure_psp->start_wall_time();",
        'after' : "proj%(id)s.measure_psp->stop_wall_time();"
    },
    'update_synapse': {
        'before' : "proj%(id)s.measure_step->start_wall_time();",
        'after' : "proj%(id)s.measure_step->stop_wall_time();"
    },
    'update_neuron': {
        'before' : "pop%(id)s.measure_step->start_wall_time();",
        'after' : "pop%(id)s.measure_step->stop_wall_time();"
    },
    'spike_gather': {
        'before' : "pop%(id)s.measure_gather->start_wall_time();",
        'after' : "pop%(id)s.measure_gather->stop_wall_time();"
    },
    'spike_prop': {
        'before' : "measure_prop->start_wall_time();",
        'after' : "measure_prop->stop_wall_time();"
    }
}
