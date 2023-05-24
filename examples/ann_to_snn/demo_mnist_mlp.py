#
#   Simple demonstration of the ANNtoSNNConverter
#
#   Authors:    Abdul Rehaman Kampli, Helge Uelo Dinkelbach and René Larisch
#
from ANNarchy.extensions.ann_to_snn_conversion import ANNtoSNNConverter
import numpy as np
import os

try:
    from mnist.loader import MNIST
    from sklearn.metrics import classification_report, accuracy_score
except:
    print('Not all necessary python packages are installed.')
    print("Please install 'python-mnist' and 'scikit-learn'.")

## if mnist_test set is not there, load it
if not os.path.exists('./mnist_testset'):
    os.mkdir('./mnist_testset/')
    os.system("wget --recursive --level=1 --cut-dirs=3 --no-host-directories --directory-prefix=mnist_testset --accept '*.gz' http://yann.lecun.com/exdb/mnist/")
    os.system("gunzip ./mnist_testset/* ./mnist_testset/")

# Load and prepare MNIST input
mndata = MNIST("./mnist_testset")
images, labels = mndata.load_testing()

ab_img=np.array(images)
ab_labl=np.array(labels)

xt=ab_img.reshape(10000, 784) / 255

# Initialize ANNarchy SNN network with pre-trained weights
# Implemented Encoding Technichs IB,CH and PSO

snn_converter = ANNtoSNNConverter(input_encoding='IB')
snn_converter.init_from_keras_model("model_MLP.h5")

# test the first n samples
n_samples = 200
predictions = snn_converter.predict(xt[:n_samples,:], duration_per_sample=100)

# resolve the case that multiple neurons were selected for one sample
predictions = [ [np.random.choice(p)] for p in predictions ]

#Classification Report and Accuracy of the network
print(classification_report(ab_labl[:n_samples], predictions))
print(accuracy_score(ab_labl[:n_samples], predictions))

