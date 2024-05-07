"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import ANNarchy.core.Global as Global

from ANNarchy.intern import Messages

try:
    from tensorboardX import SummaryWriter
except Exception as e:
    print(e)
    Messages._error("tensorboard extension: please install tensorboardX (pip install tensorboardX).")

import os
import socket
from datetime import datetime
import numpy as np


class Logger(object):
    """
    Logger class to use tensorboard to visualize ANNarchy simulations. Requires the `tensorboardX` package (pip install tensorboardX). 

    The Logger class is a thin wrapper around tensorboardX.SummaryWriter, which you could also use directly. The doc is available at <https://tensorboardx.readthedocs.io/>. Tensorboard can read any logging data, as long as they are saved in the right format (tfevents), so it is not limited to tensorflow. TensorboardX has been developed to allow the use of tensorboard with pytorch.

    The extension has to be imported explictly:

    ```python
    from ANNarchy.extensions.tensorboard import Logger
    ```

    The ``Logger`` class has to be closed properly at the end of the script, so it is advised to use a context:

    ```python
    with Logger() as logger:
        logger.add_scalar("Accuracy", acc, trial)
    ```

    You can also make sure to close it:

    ```python
    logger = Logger()
    logger.add_scalar("Accuracy", acc, trial)
    logger.close()
    ```

    By default, the logs will be written in a subfolder of ``./runs/`` (which will be created in the current directory). 
    The subfolder is a combination of the current datetime and of the hostname, e.g. ``./runs/Apr22_12-11-22_machine``. 
    You can control these two elements by passing arguments to ``Logger()``:

    ```python
    with Logger(logdir="/tmp/annarchy", experiment="trial1"): # logs in /tmp/annarchy/trial1
    ```

    The ``add_*`` methods allow you to log various structures, such as scalars, images, histograms, figures, etc.

    A tag should be given to each plot. In the example above, the figure with the accuracy will be labelled "Accuracy" in tensorboard. 
    You can also group plots together with tags such as "Global performance/Accuracy", "Global performance/Error rate", "Neural activity/Population 1", etc.

    After (or while) logging data within your simulation, run `tensorboard` in the terminal by specifying the log directory:

    ```bash
    tensorboard --logdir runs
    ```

    TensorboardX enqueues the data in memory before writing to disk. You can force flushing with:

    ```python
    logger.flush()
    ```

    :param logdir: path (absolute or relative) to the logging directory. Subfolders will be created for each individual run. The default is "runs/"
    :param experiment: name of the subfolder for the current run. By default, it is a combination of the current time and the hostname (e.g. Apr22_12-11-22_machine). If you reuse an experiment name, the data will be appended.
    """
    
    def __init__(self, logdir:str="runs/", experiment:str=None):

        self.logdir = logdir
        self.experiment = experiment

        # Create the logdir if it does not exist
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        if not experiment:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.currentlogdir=os.path.join(
                self.logdir, current_time + '_' + socket.gethostname())
        else:
            self.currentlogdir = self.logdir + "/" + self.experiment

        print("Logging in", self.currentlogdir)
    
        self._create_summary_writer()

    def _create_summary_writer(self):

         self._summary = SummaryWriter(self.currentlogdir, comment="", purge_step=None, max_queue=10, flush_secs=10, filename_suffix='', write_to_disk=True)

    # Logging methods
        
    def add_scalar(self, tag:str, value:float, step:int=None):
        """
        Logs a single scalar value, e.g. a success rate at various stages of learning.

        Example:

        ```python
        with Logger() as logger:
            for trial in range(100):
                simulate(1000.0)
                accuracy = ...
                logger.add_scalar("Accuracy", accuracy, trial)
        ```

        :param tag: name of the figure in tensorboard.
        :param value: value.
        :param step: time index.
        """
        
        self._summary.add_scalar(tag=tag, scalar_value=value, global_step=step, walltime=None)
        
    def add_scalars(self, tag:str, value:dict, step:int=None):
        """
        Logs multiple scalar values to be displayed in the same figure, e.g. several metrics or neural activities.

        Example:

        ```python
        with Logger() as logger:
            for trial in range(100):
                simulate(1000.0)
                act1 = pop.r[0]
                act2 = pop.r[1]
                logger.add_scalars(
                    "Accuracy", 
                    {'First neuron': act1, 'Second neuron': act2}, 
                    trial)
        ```

        :param tag: name of the figure in tensorboard.
        :param value: dictionary of values.
        :param step: time index.
        """
        
        self._summary.add_scalars(main_tag=tag, tag_scalar_dict=value, global_step=step, walltime=None)
        
    def add_image(self, tag:str, img: np.ndarray, step:int=None, equalize:bool=False):
        """
        Logs an image.
        
        The image must be a numpy array of size (height, width) for monochrome images or (height, width, 3) for colored images. The values should either be integers between 0 and 255 or floats between 0 and 1. The parameter ``equalize`` forces the values to be between 0 and 1 by equalizing using the min/max values.

        Example::

        ```python
        with Logger() as logger:
            for trial in range(100):
                simulate(1000.0)
                img = pop.r.reshape((10, 10))
                logger.add_image("Population / Firing rate", img, trial, equalize=True)
        ```

        :param tag: name of the figure in tensorboard.
        :param img: array for the image.
        :param step: time index.
        :param equalize: rescales the pixels between 0 and 1 using the min and max values of the array.
        """
        if img.ndim ==2:
            if equalize:  
                img = img.astype(np.float)              
                img = (img - img.min())/(img.max() - img.min())

            self._summary.add_image(tag=tag, img_tensor=img, global_step=step, walltime=None, dataformats='HW')
        
        elif img.ndim == 3:
            if not img.shape[2] == 3:
                Messages._error("Logger.add_image: color images must be of shape (H, W, 3).")
            
            if equalize:   
                img = np.array(img).astype(np.float)         
                img = (img - img.min())/(img.max() - img.min())

            self._summary.add_image(tag=tag, img_tensor=img, global_step=step, walltime=None, dataformats='HWC')

        else:
            Messages._error("Logger.add_image: images must be of shape (H, W) or (H, W, 3).")
        
    def add_images(self, tag:str, img:np.array, step:int=None, equalize:bool=False, equalize_per_image:bool=False):
        """
        Logs a set of images (e.g. receptive fields).
       
        The numpy array must be of size (number, height, width) for monochrome images or (number, height, width, 3) for colored images. The values should either be integers between 0 and 255 or floats between 0 and 1. The parameter ``equalize`` forces the values to be between 0 and 1 by equalizing using the min/max values.

        Example:

        ```python
        with Logger() as logger:
            for trial in range(100):
                simulate(1000.0)
                weights= proj.w.reshape(100, 10, 10) # 100 post neurons, 10*10 pre neurons
                logger.add_images("Projection/Receptive fields", weights, trial, equalize=True)
        ```

        :param tag: name of the figure in tensorboard.
        :param img: array for the images.
        :param step: time index.
        :param equalize: rescales the pixels between 0 and 1 using the min and max values of the array.
        :param equalize_per_image: whether the rescaling should be using the global min/max values of the array, or per image. Has no effect if equalize of False.
 
        """
        if img.ndim == 3:
            img = np.expand_dims(img, axis=3)
        
        if equalize:   
            img = np.array(img).astype(np.float) 
            if not equalize_per_image:        
                img = (img - img.min())/(img.max() - img.min())
            else:
                for i in range(img.shape[0]):
                    img[i,...] = (img[i,...] - img[i,...].min())/(img[i,...].max() - img[i,...].min())
        
        self._summary.add_images(tag=tag, img_tensor=img, global_step=step, walltime=None, dataformats='NHWC')
        
    def add_parameters(self, params:dict, metrics:dict):
        """
        Logs parameters of a simulation.

        This should be run only once per simulation, generally at the end. 
        This allows to compare different runs of the same network using 
        different parameter values and study how they influence the global output metrics, 
        such as accuracy, error rate, reaction speed, etc.

        Example:

        ```python
        with Logger() as logger:
            # ...
            logger.add_parameters({'learning_rate': lr, 'tau': tau}, {'accuracy': accuracy})
        ```

        :param params: dictionary of parameters.
        :param metrics: dictionary of metrics.
        """
        
        self._summary.add_hparams(params, metrics)

    def add_histogram(self, tag:str, hist: list | np.ndarray, step:int=None):
        """
        Logs an histogram.

        Example:

        ```python
        with Logger() as logger:
            for trial in range(100):
                simulate(1000.0)
                weights= proj.w.flatten()
                logger.add_histogram("Weight distribution", weights, trial)
        ```


        :param tag: name of the figure in tensorboard.
        :param hist: a list or 1D numpy array of values.
        :param step: time index.
        """

        self._summary.add_histogram(tag, hist, step)

    def add_figure(self, tag:str, figure: list | np.ndarray, step:int=None, close:bool=True):
        """
        Logs a Matplotlib figure.

        Example:

        ```python
        with Logger() as logger:
            for trial in range(100):
                simulate(1000.0)
                fig = plt.figure()
                plt.plot(pop.r)
                logger.add_figure("Activity", fig, trial)
        ```

        :param tag: name of the image in tensorboard.
        :param figure: a list or 1D numpy array of values.
        :param step: time index.
        :param close: whether the logger will close the figure when done (default: True).
        """

        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as plt_backend_agg
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        if close:
            plt.close(figure)
        self._summary.add_image(tag, image_chw, step)
    
    # Resource management
    def flush(self):
        "Forces the logged data to be flushed to disk."
        self._summary.flush()
        
    def close(self):
        "Closes the logger."
        self._summary.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()