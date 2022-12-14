#===============================================================================
#
#     ANNtoSNNConverter.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2022    Abdul Rehaman Kampli <>
#                           Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#                           René Larisch <>
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
from ANNarchy.core import Global
from ANNarchy.core.Neuron import Neuron
from ANNarchy.core.Population import Population
from ANNarchy.core.Projection import Projection
from ANNarchy.core.Monitor import Monitor
from ANNarchy.core.SpecificPopulation import PoissonPopulation
from ANNarchy import compile, simulate, reset

from tqdm import tqdm
import numpy as np
import h5py

from .InputEncoding import *

IF = Neuron(
    parameters = """
        vt = 1 : population
        vr = 0 : population
    """,
    equations = """
        dv/dt = g_exc : init = 0
    """,
    spike = """
        v > vt
    """,
    reset = """
        v = vr
    """
)

class ANNtoSNNConverter(object):
    """
    Implements a conversion of a pre-trained fully-connected Keras model into a spiking model. The procedure is
    based on the implementation of Diehl et al. (TODO: reference)

    Parameters:

    * neuron_model:     neuron model for hidden and output layer. Either the default integrate-and-fire (IF) or an ANNarchy Neuron object
    * input_encoding:   a string which input incoding should be used: poisson, PSO, IB and CH (for more details see InputEncoding)
    """

    def __init__(self, neuron_model=IF, input_encoding='poisson', **kwargs):

        self._neuron_model = neuron_model
        self._input_encoding = input_encoding

        if input_encoding == "poisson":
            self._input_model = None
        elif input_encoding == 'PSO':
            self._input_model=PSO
        elif input_encoding=='IB':
            self._input_model=IB
        elif input_encoding=='CH':
            self._input_model=CH
        else:
            raise ValueError("Unknown input encoding:", input_encoding)

        self._max_f = 1000      # scale factor used for poisson encoding

        # TODO: sanity check on key-value args
        for key, value in kwargs:
            if key == "max_f":
                self._max_f = value


    def init_from_keras_model(self, model_as_h5py):
        """
        Read out the pre-trained model provided as .h5

        TODO:   it might be a better approach to copy the layer names stored in the .h5 file
                and assign them to the ANNarchy populations too ... (HD: 20th July 2022)
        """
        #
        # 1st step: extract weights from model file
        #
        dims, weight_matrices = self._extract_weight_matrices(model_as_h5py)

        #
        # 2nd step: normalize weights
        #
        norm_weight_matrices = self._normalize_weights(weight_matrices)

        #
        # 3rd step: build up ANNarchy network
        #
        pop = [None] * len(dims)

        if self._input_encoding=='poisson':
            pop[0] = PoissonPopulation(name = 'Input', geometry=dims[0], rates=0)
            print('The Selected Encoding Method is Rate Coded')

        else:
            pop[0] = Population(name = 'Input', geometry=dims[0], neuron=self._input_model)
            print('The Selected Encoding Method is Temporal Coding')

        # populations
        for i in range(1, len(dims)):
            pop[i]=Population(geometry=dims[i], neuron=self._neuron_model, name=f"pop{i}")

            # ARK:  scaling the threshold as number of layers increases divide
            #       the value 1/half of the number of the network
            pop[i].vt=1.0/float(len(dims))

        # projections
        proj=[None]*(len(dims)-1)
        for i in range(len(dims)-1):
            proj[i] = Projection(pre = pop[i], post = pop[i+1], target = "exc")
            # will be overwritten after compile
            proj[i].connect_all_to_all(weights=0.0, force_multiple_weights=True)

        # First layer is the input
        self._input_pop = pop[0]

        # Last layer is the output
        self._output_pop = pop[-1]

        # we use the spike count in the last layer as read-out
        self._pop_class_mon = Monitor(pop[-1],['spike'])

        compile()

        # Set the pre-trained weights. Please note, that the last projection
        # (connection to output) is not normalized!
        for i in range(len(dims)-2):
            proj[i].w=norm_weight_matrices[i]
        proj[-1].w=weight_matrices[-1]

    def get_annarchy_network(self):
        """
        returns an ANNarchy.core.Network instance
        """
        pass

    def predict(self, samples, duration_per_sample=1000, measure_time=False, **kwargs):
        """
        returns class label for a given input series. 

        Parameters:

        * samples: set of inputs to present to the network. The function expects a 2-dimensional array (num_samples, input_size).
        * duration_per_sample: the number of simulation steps for one input sample (default: 1000, 1 second biological time)
        * measure_time: print out the computation time spent for one input sample (default: False)
        """
        predictions = []

        # Iterate over all samples
        for i in tqdm(range(samples.shape[0]),ncols=80):
            # Reset state variables
            reset(populations=True, monitors=True, projections=False)
            
            # transform input
            self._set_input(samples[i,:], self._input_encoding)
            
            # simulate 1s and record spikes in output layer
            simulate(duration_per_sample, measure_time=measure_time)

            # count the number of spikes each output neuron emitted.
            # The predicted label is the neuron index with the highest
            # number of spikes.
            spk_class = self._pop_class_mon.get('spike')
            act_pred = np.zeros(self._output_pop.size)
            for c in range(self._output_pop.size):
                act_pred[c] = len(spk_class[c])
            predictions.append(np.argmax(act_pred))
        
        return predictions

    def _set_input(self, sample, encoding):
        """
        We will support different input encodings in the future.
        """
        if encoding == "poisson":
            self._input_pop.rates = sample*self._max_f
        else:
            self._input_pop.rates = sample*self._max_f

    def _extract_weight_matrices(self, filename):
        """
        Read the .h5 file and extract layer sizes as well as the
        pre-trained weights.
        """
        f=h5py.File(filename,'r')
        
        if not 'model_weights' in f.keys():
            Global._error("could not find weight matrices")

        model_weights = (f['model_weights'])
        layer_names = list(model_weights.keys())

        Global._debug("ANNtoSNNConverter: detected", len(layer_names), "layers.")
        weight_matrices=[]
        dimension_list=[]

        for layer_name in layer_names:
            # Skip input
            if layer_name == 'input_1':
                continue 

            # Input -> 1st layer
            if layer_name == 'dense':
                w1 = np.asarray(model_weights[layer_name][layer_name]['kernel:0'])
                weight_matrices.append(np.asarray(w1).T) 
                # store both dimensions, input and 1st hidden layer
                dimension_list.extend(w1.shape)

            # dense_x are the other hidden layers
            else:
                w1 = np.asarray(model_weights[layer_name][layer_name]['kernel:0'])
                weight_matrices.append(np.asarray(w1).T)
                # store only the 2nd dimension which is the next hidden layer
                dimension_list.append(w1.shape[1])

        return dimension_list, weight_matrices

    def _normalize_weights(self, weight_matrices):
        """

        TODO: documentation
        """
        norm_wlist=[]

        for a in weight_matrices:
            max_pos_input = 0
        
            for row in range (a.shape[0]):
                input_sum = 0
            
                input_sum = np.sum(a[row,np.where(a[row,:]>0)])
            
                max_pos_input = max(max_pos_input, input_sum) 
            
            for row in range (a.shape[0]):
                a[row]=a[row]/max_pos_input

            norm_wlist.append(a)

        return norm_wlist
