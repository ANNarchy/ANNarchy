#===============================================================================
#
#     ANNtoSNNConverter.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2022    Abdul Rehaman Kampli <>
#                           René Larisch <renelarischif@gmail.com>
#                           Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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
from ANNarchy.core.Network import Network
from ANNarchy.core.Neuron import Neuron
from ANNarchy.core.Population import Population
from ANNarchy.core.Projection import Projection
from ANNarchy.core.Monitor import Monitor
from ANNarchy.core.SpecificPopulation import PoissonPopulation
from ANNarchy.core.Random import Uniform
from ANNarchy.extensions.convolution import Convolution, Pooling
from ANNarchy import compile, simulate, reset

from tqdm import tqdm
import numpy as np
import h5py
import json
from copy import copy

from .InputEncoding import *

IaF = Neuron(
    parameters = """
        vt = 1          : population
        vr = 0          : population
        mask_tau = 20.0 : population
    """,
    equations = """
        dv/dt    = g_exc          : init = 0.0 , min=-2.0
        dmask/dt = -mask/mask_tau : init = 0.0
    """,
    spike = """
        v > vt
    """,
    reset = """
        v = vr
        mask += 1/mask_tau
    """
)

class ANNtoSNNConverter(object):
    """
    Implements a conversion of a pre-trained fully-connected Keras model into a spiking model. The procedure is
    based on the implementation of Diehl et al. (2015) "Fast-classifying, high-accuracy spiking deep networks
    through weight and threshold balancing".

    Parameters:

    * neuron_model:     neuron model for hidden and output layer. Either the default integrate-and-fire (IaF) or an ANNarchy Neuron object
    * input_encoding:   a string which input incoding should be used: custom poisson, PSO, IB and CH (for more details see InputEncoding)
    """

    def __init__(self, neuron_model=IaF, input_encoding='poisson', **kwargs):

        self._neuron_model = neuron_model
        self._input_encoding = input_encoding

        if input_encoding == "poisson":
            self._input_model = 'CPN'
        elif input_encoding == 'PSO':
            self._input_model=PSO
        elif input_encoding=='IB':
            self._input_model=IB
        elif input_encoding=='CH':
            self._input_model=CH
        else:
            raise ValueError("Unknown input encoding:", input_encoding)

        self._max_f = 100      # scale factor used for poisson encoding

        # TODO: sanity check on key-value args
        for key, value in kwargs:
            if key == "max_f":
                self._max_f = value

        self.snn_network = None

    def init_from_keras_model(self, model_as_h5py, show_info=True):
        """
        Read out the pre-trained model provided as .h5
        """
        #
        # 1st step: extract weights from model file
        #
        weight_matrices, layer_order, input_dim = self._extract_weight_matrices(model_as_h5py)

        #
        # 2nd step: normalize weights
        #
        norm_weight_matrices = self._normalize_weights(weight_matrices)

        #
        # 3rd step: build up ANNarchy network
        #
        snn_network = Network(everything = False)
        input_pop = Population(name = layer_order[0], geometry=input_dim, neuron=self._input_model)
        snn_network.add(input_pop)

        if show_info:
            print()
            print('Show populations/layer')
            print('----------------------')

        ### create Populations ###
        for layer in range(len(layer_order)):
            if 'conv' in layer_order[layer]:

                l_weights = norm_weight_matrices[layer] # get the weights
                dim_0 = np.shape(l_weights)[0]

                geometry = input_dim # add it to the geometry
                geometry = geometry + (dim_0,)
                conv_pop = Population(geometry = geometry, neuron=IaF , name=layer_order[layer] ) # create convolution population
                conv_pop.vt = conv_pop.vt - (0.05*layer) # reduce the threshold for deeper layers
                snn_network.add(conv_pop)

                if show_info:
                    print(layer_order[layer], 'geometry = ', geometry)

            elif 'pool' in layer_order[layer]:

                input_dim = (int(input_dim[0]/ 2), int(input_dim[1]/ 2))
                l_weights = norm_weight_matrices[layer-1] # get the weights of the previous layer (should be a conv-layer, or not?)
                dim_0 = np.shape(l_weights)[0]

                geometry = input_dim # add it to the geometry
                geometry = geometry + (dim_0,)
                pool_pop = Population(geometry = geometry, neuron=IaF , name=layer_order[layer])
                pool_pop.vt = pool_pop.vt - (0.05*layer) # reduce the threshold for deeper layers
                snn_network.add(pool_pop)

                if show_info:
                    print(layer_order[layer], 'geometry = ', geometry)

            elif 'dense' in layer_order[layer]:

                l_weights = norm_weight_matrices[layer]
                dim_0 = np.shape(l_weights)[0]

                geometry = dim_0
                dense_pop = Population(geometry = geometry, neuron=IaF , name=layer_order[layer])
                # ARK:  scaling the threshold as number of layers increases divide
                #       the value 1/half of the number of the network
                dense_pop.vt = dense_pop.vt - (0.05*layer)
                # HD (20th Feb. 2023): we want to generate this firing vector for a
                #                      single time step
                dense_pop.compute_firing_rate(Global.dt())
                snn_network.add(dense_pop)

                if show_info:
                    print(layer_order[layer], 'geometry = ', geometry)


        ### create Projections ###
        if show_info:
            print()
            print('Show Connections/Projections')
            print('----------------------')
        for p in range(1,len(layer_order)):
            if show_info:
                print('--------')
            if 'conv' in layer_order[p]:

                post_pop = snn_network.get_population(layer_order[p])
                pre_pop = snn_network.get_population(layer_order[p-1])

                weight_m = np.squeeze(norm_weight_matrices[p])

                conv_proj = Convolution(pre = pre_pop, post=post_pop, target='exc', psp="pre.mask * w", name='conv_proj_%i'%p)
                conv_proj.connect_filters(weights=weight_m)
                snn_network.add(conv_proj)

                if show_info:
                    print(layer_order[p-1],' -> ' ,layer_order[p])
                    print(pre_pop.geometry, post_pop.geometry)
                    print('weight_m :', np.shape(weight_m))


            elif 'pool' in layer_order[p]:

                post_pop = snn_network.get_population(layer_order[p])
                pre_pop = snn_network.get_population(layer_order[p-1])

                pool_proj = Pooling(pre = pre_pop, post=post_pop, target='exc', operation='max', psp="pre.mask", name='pool_proj_%i'%p)
                pool_proj.connect_pooling(extent=(2,2,1))
                snn_network.add(pool_proj)
                if show_info:
                    print(layer_order[p-1],' -> ' ,layer_order[p])

            elif 'dense' in layer_order[p]:

                weight_m = norm_weight_matrices[p]

                post_pop = snn_network.get_population(layer_order[p])
                pre_pop = snn_network.get_population(layer_order[p-1])

                dense_proj = Projection(pre = pre_pop, post = post_pop, target = "exc", name='dense_proj_%i'%p)
                if pre_pop.neuron_type.type=="rate":
                    dense_proj.connect_all_to_all(weights=Uniform(0,1), storage_format="dense")
                else:
                    dense_proj.connect_all_to_all(weights=Uniform(0,1), storage_format="dense", storage_order="pre_to_post")

                snn_network.add(dense_proj)
                if show_info:
                    print(layer_order[p-1],' -> ' ,layer_order[p])
                    print(pre_pop.geometry, post_pop.geometry)
                    print('weight_m :', np.shape(weight_m))

        #snn_network.compile(directory='ann/annarchy_'+str(trial_number))
        snn_network.compile()

        ## go again over all dense projections to load the weight matrices ##
        for proj in snn_network.get_projections():
            if 'dense' in proj.name: # find the dense projection
                proj_name = proj.name.split('_')
                proj_idx = int(proj_name[-1]) # get the index of the dense layer in relation to all other layers

                ## use the not normed weights to the classification layer
                proj.w = norm_weight_matrices[proj_idx]

        self.snn_network = snn_network
        #return(snn_network)

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
        m_popClass = Monitor(self.snn_network.get_population(self.snn_network.get_populations()[-1].name), ['spike']) # record always the last layer
        self.snn_network.add(m_popClass)

        class_pop_size = self.snn_network.get_population(self.snn_network.get_populations()[-1].name).size

        # Iterate over all samples
        for i in tqdm(range(samples.shape[0]),ncols=80):
            # Reset state variables
            self.snn_network.reset(populations=True, monitors=True, projections=False)

            # transform input
            #self._set_input(samples[i,:], self._input_encoding)
            self.snn_network.get_population('input_1').rates =  samples[i,:]*self._max_f

            # simulate 1s and record spikes in output layer
            self.snn_network.simulate(duration_per_sample, measure_time=measure_time)

            # count the number of spikes each output neuron emitted.
            # The predicted label is the neuron index with the highest
            # number of spikes.
            spk_class = self.snn_network.get(m_popClass).get('spike')

            act_pred = np.zeros(class_pop_size)
            for c in range(class_pop_size):
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

        ## get the configuration of the Keras model
        model_config = f.attrs.get("model_config")
        model_config = model_config.decode("utf-8")
        model_config = json.loads(model_config)

        ## get the list with all layer names
        model_layers = (model_config['config']['layers'])
        model_weights = (f['model_weights'])

        Global._debug("ANNtoSNNConverter: detected", len(model_layers), "layers.")

        weight_matrices=[] # array to save the weight matrices
        layer_order = [] # additional array to save the order of the layers to know it later


        for layer in model_layers:
            layer_name = layer['config']['name']

            if 'conv2d' in layer_name:
                layer_w = model_weights[layer_name][layer_name]['kernel:0']
                ## if it is a convolutional layer, reshape it to fitt to annarchy
                dim_h, dim_w, dim_pre, dim_post = np.shape(layer_w)
                new_w = np.zeros((dim_post, dim_h, dim_w, dim_pre))
                for i in range(dim_post):
                    new_w[i,:,:] = layer_w[:,:,:,i]
                weight_matrices.append(new_w)
                layer_order.append(layer_name)

            elif 'dense' in layer_name:
                layer_w = model_weights[layer_name][layer_name]['kernel:0']
                weight_matrices.append(np.transpose(layer_w))
                layer_order.append(layer_name)

            elif 'pool' in layer_name:
                layer_order.append(layer_name)
                weight_matrices.append([]) # add an empty weight matrix to pad the array

            elif 'input' in layer_name:
                layer_order.append(layer_name)
                input_dim = layer['config']['batch_input_shape']
                if len(input_dim) >2 : #probably a conv. if >2
                    input_dim = tuple(input_dim[1:3])
                else:           # probably a MLP
                    input_dim = input_dim[1]
                weight_matrices.append([]) # add an empty weight matrix to pad the array

        return weight_matrices, layer_order, input_dim

    def _normalize_weights(self, weight_matrices):
        """
        Weight normalization based on the "model based normalization" from Diehl et al. (2015)
        """
        norm_wlist=[]

        ## iterate over all weight matrices
        for level in range(len(weight_matrices)):
            max_pos_input = 0
            w_matrix = copy(weight_matrices[level])
            if len(w_matrix)> 0: # Empty weight matrix ?
                ## each row correspnds to one post-synaptic neuron
                for row in range (w_matrix.shape[0]):
                    w_matrix_flat=w_matrix[row].flatten()
                    idx=np.where(w_matrix_flat>0)
                    input_sum=np.sum(w_matrix_flat[idx])
                    ## save the maximum input current over all post neurons in this connection
                    max_pos_input = max(max_pos_input, input_sum)

                for row in range (w_matrix.shape[0]):
                    ## normalize the incoming weights for each neuron, based on the maximum input
                    ## for the complete connection
                    ## and multiply it with the deepth of the connection to boost the input current
                    w_matrix[row]=(level+1)* w_matrix[row]/max_pos_input

            norm_wlist.append(w_matrix)

        return norm_wlist
