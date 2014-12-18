import ANNarchy.core.Global as Global

class ProfileGenerator(object):

    def __init__(self, populations, projections):
        
        self.populations = populations
        self.projections = projections

    def generate(self):
        # Generate header for profiling
        with open(Global.annarchy_dir+'/generate/Profiling.h', 'w') as ofile:
            ofile.write(self._generate_header())

        # Generate cpp for profiling
        with open(Global.annarchy_dir+'/generate/Profiling.cpp', 'w') as ofile:
            ofile.write(self._generate_body())

    def calculate_num_ops(self):
        num_ops= 0
        for proj in Global._projections:
            num_ops += 1
        for pop in Global._populations:
            num_ops += 1
        return num_ops

    def _generate_header(self):
        from .HeaderTemplate import openmp_profile_header, cuda_profile_header
        if Global.config["paradigm"] == "openmp":
            return openmp_profile_header
        else:
            return cuda_profile_header
    
    def _generate_body(self):
        from .BodyTemplate import openmp_profile_body, cuda_profile_body

        num_op = self.calculate_num_ops()
        num_threads = 6
        
        # count initialization
        count = """    set_CPU_time_number( %(num_op)s * %(num_thread)s );
""" % { 'num_op': num_op,
        'num_thread': num_threads,
       }

        name = ""
        add = ""
        c = 0
        for proj in Global._projections:
            name += """        set_CPU_time_name( i*%(num_op)s+%(off)s,"Proj%(id)s - ws");
""" % { 'id': proj.id, 'num_op': num_op, 'off': c }
            c+= 1
        for pop in Global._populations:
            name += """        set_CPU_time_name( i*%(num_op)s+%(off)s,"%(name)s-step()");
""" % { 'name': pop.name, 'num_op': num_op, 'off': c }
            c+= 1

        c = 0
        for proj in Global._projections:
            add += """        set_CPU_time_additional( i*%(num_op)s+%(off)s, s);
""" % { 'id': proj.id, 'num_op': num_op, 'off': c }
            c+= 1
        for pop in Global._populations:
            add += """        set_CPU_time_additional( i*%(num_op)s+%(off)s, s);
""" % { 'id': pop.id, 'num_op': num_op, 'off': c }
            c+= 1

        init = """
    // setup counter
%(count)s
""" % { 'count': count }

        init2 = """
    for ( int i = 0; i < %(num_threads)s; i++ )
    {
        // set names
%(name)s

        // set additonal
        std::string s = std::to_string(i+1);
        s+=";Threads";
%(add)s
    }
""" % { 'num_threads': num_threads,
        'name': name,
        'add': add 
       }
         
        if Global.config["paradigm"] == "openmp":
            code = openmp_profile_body % { 'init': init, 'init2': init2  }
            return code
        else:
            return cuda_profile_body 
    