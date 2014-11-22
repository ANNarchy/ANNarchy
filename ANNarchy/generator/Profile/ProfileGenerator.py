import ANNarchy.core.Global as Global

class ProfileGenerator(object):

    def __init__(self, populations, projections):
        
        self.populations = populations
        self.projections = projections

    def generate(self):

        # Generate header for profiling
        with open(Global.annarchy_dir+'/generate/Profiling.h', 'w') as ofile:
            ofile.write(self.generate_header())

        # Generate cpp for profiling
        with open(Global.annarchy_dir+'/generate/Profiling.cpp', 'w') as ofile:
            ofile.write(self.generate_body())

    def generate_header(self):
        from .HeaderTemplate import openmp_profile_header, cuda_profile_header
        if Global.config["paradigm"] == "openmp":
            return openmp_profile_header
        else:
            return cuda_profile_header
    
    def generate_body(self):
        from .BodyTemplate import openmp_profile_body, cuda_profile_body
        if Global.config["paradigm"] == "openmp":
            return openmp_profile_body
        else:
            return cuda_profile_body
    