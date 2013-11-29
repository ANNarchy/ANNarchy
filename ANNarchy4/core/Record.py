"""

    Global.py
    
    This file is part of ANNarchy.
    
    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from ANNarchy4.core import Global

class Record:
    """
    A simple helper class used by the ANNarchy4.core.Population class.
    """
    INITED = 0
    RUNNING = 1
    PAUSED = 2
    
    def __init__(self, name):
        """
        Constructor.
        
        Parameter:
        
        * *name*: variable name
        """
        self._start = []
        self._stop = []
        self._name = name
        self._state = self.INITED
        
    def start(self):
        """
        either starts or resume the recording, additional the current network time is stored.
        """
        self._start.append(Global.current_step())
        self._state = self.RUNNING
        
    def pause(self):
        """
        pause the recording, additional the current network time is stored.
        """
        self._stop.append(Global.current_step())
        self._state = self.RUNNING

    def reset(self):
        """
        reset the recording.
        """
        self._start = []
        self._stop = []
        self._state = self.INITED

    @property
    def start_time(self):
        """
        Returns the network time(s) where data was recorded.
        """
        if len(self._start)==1:
            return self._start[0]
        else:
            return self._start

    @property
    def stop_time(self):
        """
        Returns the network time(s) where data record was paused/stopped.
        """
        if len(self._stop)==1:
            return self._stop[0]
        else:
            return self._stop
        
    @property
    def is_inited(self):
        """
        Returns if record is inited.
        """
        return self._state == self.INITED

    @property
    def is_paused(self):
        """
        Returns if record is paused.
        """
        return self._state == self.PAUSED
        
    @property
    def is_running(self):
        """
        Returns if record is paused.
        """
        return self._state == self.RUNNING