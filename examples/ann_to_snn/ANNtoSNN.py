# ANN-to-SNN conversion
# 
# This script demonstrates how to transform a neural network trained using tensorflow/keras into an SNN network usable in ANNarchy.
# 
# The models are adapted from the original models used in:
# 
# > Diehl et al. (2015) "Fast-classifying, high-accuracy spiking deep networks through weight and threshold balancing" Proceedings of IJCNN. doi: 10.1109/IJCNN.2015.7280696


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# First we need to download and process the MNIST dataset provided by tensorflow.
# Download data
(X_train, t_train), (X_test, t_test) = tf.keras.datasets.mnist.load_data()

# Normalize inputs
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255.
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255.

# One-hot output vectors
T_train = tf.keras.utils.to_categorical(t_train, 10)
T_test = tf.keras.utils.to_categorical(t_test, 10)


def create_mlp():
    # Model
    inputs = tf.keras.layers.Input(shape=(784,))
    x= tf.keras.layers.Dense(128, use_bias=False, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x= tf.keras.layers.Dense(128, use_bias=False, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x=tf.keras.layers.Dense(10, use_bias=False, activation='softmax')(x)

    model= tf.keras.Model(inputs, x)

    # Optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

    # Loss function
    model.compile(
        loss='categorical_crossentropy', # loss function
        optimizer=optimizer, # learning rule
        metrics=['accuracy'] # show accuracy
    )
    print(model.summary())

    return model


# The script should be passed "--train" in order to train the ANN
import argparse
parser = argparse.ArgumentParser(description='ANN2SNN demo.')
parser.add_argument('--train', dest='train', action='store_const',
                    const=True, default=False,
                    help='defines whether to train the ANN.')
args = parser.parse_args()

if args.train:
    # Create model
    model = create_mlp()

    # Train model
    history = model.fit(
        X_train, T_train,       # training data
        batch_size=100,          # batch size
        epochs=20,              # Maximum number of epochs
        validation_split=0.1,   # Percentage of training data used for validation
    )
    model.save("runs/mlp.h5")

    # Test model
    predictions_keras = model.predict(X_test, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, T_test, verbose=0)
    print(f"Test accuracy of the ANN: {test_accuracy}")


# Initialize the ANN-to-SNN converter
# 
# We first create an instance of the ANN-to-SNN conversion object. The function receives the *input_encoding* parameter, which is the type of input encoding we want to use. 
# 
# By default, there are *intrinsically bursting* (`IB`), *phase shift oscillation* (`PSO`) and *Poisson* (`poisson`) available.
from ANNarchy.extensions.ann_to_snn_conversion import ANNtoSNNConverter

snn_converter = ANNtoSNNConverter(
    input_encoding='IB',
    hidden_neuron='IaF', 
    read_out='spike_count',
)

net = snn_converter.init_from_keras_model("runs/mlp.h5")

predictions_snn = snn_converter.predict(X_test, duration_per_sample=100)

# Depending on the selected read-out method, it can happen that multiple neurons/classes are selected as a winner for an example. For example, if `duration_per_sample` is too low, several output neurons might output the same number of spikes. 
# 
# In the following cell, we force the predictions to keep only one of the winning neurons by using `np.random.choice`.

predictions_snn = [ [np.random.choice(p)] for p in predictions_snn ]

from sklearn.metrics import classification_report, accuracy_score

print(classification_report(t_test, predictions_snn))
print("Test accuracy of the SNN:", accuracy_score(t_test, predictions_snn))

