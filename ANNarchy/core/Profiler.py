#===============================================================================
#
#     Profiler.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2018 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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
#===============================================================================
import time
import csv
import matplotlib.pylab as plt

import ANNarchy.core.Global as Global

class Profiler(object):
    """
    The Profiler module should help to understand the performance of a simulation
    using the ANNarchy neural simulator.

    Therefore are functions to investigate memory consumption and timeline
    information provided.
    """
    _color_code = {
        "default": "blue",
        "compile": "green",
        "simulate": "red",
        "instantiate": "orange",
        # will be ignored in image
        "cpp core": "black"
    }

    def __init__(self):
        """
        Initialize profiler instance and register it in Global.
        """
        Global.config["profiling"] = True   # enable c++ profiling

        if Global._profiler is not None:
            Global._warning("Profiling already initialized ...")

        Global._profiler = self
        self._basetime = time.time()
        self._entries = []
        self._cpp_profiler = None       # set during Compiler._instantiate()
        self.add_entry( self._basetime, self._basetime, "initialized" )
    
    def add_entry( self, t_entry, t_escape, label, group="default" ):
        """
        Add a function to timeline.

        :param t_entry: entry time point of the function
        :param t_escape: escape time point of the function
        :param label: label of the function
        *:param group: which group does the function belong to (determines color code, default="default")
        """
        if group not in self._color_code.keys():
            # unknown group will be set to default values
            group = "default"

        self._entries.append( (t_entry, t_escape, label, group) )

    def update_entry( self, t_entry, t_escape, label, group ):
        """
        The profile entries are a list of tuples. Therefore such an entry can
        not modified easily.
        """
        if group not in self._color_code.keys():
            # unknown group will be set to default values
            group = "default"

        found = False
        for idx_t, (_,_,it_label, it_group) in enumerate(self._entries):
            if label == it_label and group == it_group:
                tmp = list(self._entries[idx_t])
                tmp[0] = t_entry
                tmp[1] = t_escape
                self._entries[idx_t] = tuple(tmp)
                found = True

        if not found:
            Global._warning("Profiler.update_entry(): the entry was not found ...")

    def clear(self):
        """
        Clear all recorded time points.
        """
        self._entries.clear()

    def print_profile(self):
        """
        Print the content to console.
        """
        divided = ["cpp core", "instantiate", "compile"]

        for t_start, t_end, label, group in self._entries:
            if group not in divided: # Python functions
                print(label, ":", t_end-t_start, "seconds")

            if group == "compile":
                if label == "overall":
                    print("compile:", t_end-t_start, "seconds")
                else:
                    print("-", label, t_end-t_start, "seconds")

            if group == "instantiate":
                if label == "overall":
                    print("instantiate:", t_end-t_start, "seconds")
                else:
                    print("-", label, t_end-t_start, "seconds")

            if group == "cpp core": # CPP functions
                if t_start == 0.0:
                    continue

                if label == "overall":
                    print("-", label,":", t_start, "seconds (", t_end, "% )")
                else:
                    print("  -", label,":", t_start, "seconds (", t_end, "% )")

    def store_cpp_time_as_csv(self):
        """
        Store the measured timings on the C++ core as .csv to
        be further processed e. g. using pandas.
        """
        if Global._check_paradigm("cuda"):
            fname = "profile_cuda.csv"
        else:
            fname = "profile_omp_"+str(Global.config["num_threads"])+"threads.csv"

        with open(Global.config["profile_out"]+'/'+fname, mode='w') as Datafile:
            csv_writer = csv.writer(Datafile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

            for t_start, t_end, label, group in self._entries:
                # skip Python functions
                if group != "cpp core":
                    continue

                # non-defined function
                if t_start == 0.0:
                    continue

                # CPP functions
                if group == "cpp core":
                    csv_writer.writerow( (label, t_start,) )

    def show_timeline(self, store_graph=False):
        """
        Visualize the timeline.
        """
        f, ax = plt.subplots()

        scale_param = 1.0 # origin data is in second

        for ts in self._entries:
            # Overview
            ax.barh(4, (ts[1]-ts[0])*scale_param, left=(ts[0]-self._basetime)*scale_param, color=self._color_code["default"])
        
            if ( ts[3] == "compile"):
                bar = ax.barh(3, (ts[1]-ts[0])*scale_param, left=(ts[0]-self._basetime)*scale_param, color=self._color_code[ts[3]], edgecolor="k", linewidth=2.0)
                patch = bar.patches[-1]
                
                bl = patch.get_xy()
                x = 0.5*patch.get_width() + bl[0]
                y = bl[1] - 0.1
                ax.text(x, y, ts[2], ha='center',va='center')

            if ( ts[3] == "simulate"):
                ax.barh(2, (ts[1]-ts[0])*scale_param, left=(ts[0]-self._basetime)*scale_param, color=self._color_code[ts[3]])

        ax.set_yticks([4,3,2])
        ax.set_yticklabels(["all", "compile()", "simulate()"])
        ax.set_xlabel("clock time [s]")
        
        if not store_graph:
            plt.show()
        else:
            f.savefig("timeline.png")