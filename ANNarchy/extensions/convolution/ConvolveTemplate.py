# =============================================================================
#
#     ConvolutionTemplate.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2019-2020  Julien Vitay <julien.vitay@gmail.com>,
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
# =============================================================================
convole_template_omp = {
    # Declare the connectivity matrix
    'declare_connectivity_matrix': """
    // Connectivity data
    std::vector<int> post_rank;
    std::vector< std::vector<int> > pre_coords;
    """,

    # Accessors for the connectivity matrix
    'access_connectivity_matrix': """
    // Accessor to connectivity data
    std::vector<int> get_post_rank() { return post_rank; }
    void set_post_rank(std::vector<int> ranks) { post_rank = ranks; }
    std::vector<std::vector<int>> get_pre_coords() { return pre_coords; }
    void set_pre_coords(std::vector<std::vector<int>> coords) { pre_coords = coords; }
    int nb_synapses() { return 0; } // TODO: filter-dim * number of filters?
    int dendrite_size(int n) { return 0; } // TODO: filter-dim?
    int nb_dendrites() { return post_rank.size(); }
""" ,

    # Export the connectivity matrix
    'export_connectivity': """
        # Connectivity
        vector[int] get_post_rank()
        void set_post_rank(vector[int])
        vector[vector[int]] get_pre_coords()
        void set_pre_coords(vector[vector[int]])
        int nb_dendrites()
""",

    # Arguments to the wrapper constructor
    'wrapper_args': "weights, coords",

    # Initialize the wrapper connectivity matrix
    'wrapper_init_connectivity': """
        proj%(id_proj)s.set_post_rank(list(range(%(size_post)s)))
        proj%(id_proj)s.set_pre_coords(coords)
""",

    # Delays
    'wrapper_init_delay': "",

    # Something like init_from_lil?
    'wrapper_connector_call': "",

    # Wrapper access to connectivity matrix
    'wrapper_access_connectivity': """
    # Connectivity
    def post_rank(self):
        return proj%(id_proj)s.get_post_rank()
    def pre_coords(self):
        return proj%(id_proj)s.get_pre_coords()
""",

    # Wrapper access to variables
    'wrapper_access_parameters_variables' : "",

    # Variables for the psp code
    'psp_prefix': """
        int rk_pre;
        %(float_prec)s sum=0.0;
"""
}