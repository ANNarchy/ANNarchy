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
        "simulate": "red"
    }

    def __init__(self):
        """
        Initialize profiler instance and register it in Global.
        """
        if Global._profiler is None:
            Global._profiler = self
            self._basetime = time.time()
            self._entries = []
            self.add_entry( self._basetime, self._basetime, "initialized" )
        else:
            print("Profiling already initialized ...")
    
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

        import pprint
        pprint.pprint(self._entries)

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