"""

A library to perform feature visualization of most layers in most networks.


Attempts to re-implement experiments performed and referenced in:
- https://distill.pub/2017/feature-visualization/
- https://raghakot.github.io/keras-vis/
- https://www.auduno.com/2015/07/29/visualizing-googlenet-classes/

Author: Simon Thomas
Email: simon.thomas@uq.edu.au
Date: 12th June, 2019

"""


import keras
from keras.models import Model
import keras.backend as K

from skimage.transform import rescale, resize
from skimage.filters import gaussian

import numpy as np
import matplotlib.pyplot as plt

from sys import maxsize as infinity


class Utils(object):
    """
    A class that provides useful functions
    """
    @staticmethod
    def deprocess_image(x):
        """
        Converts the (-inf, inf) image to have range 0-255.

        Input:
            x - image as numpy array

        Output:
            x - image a numpy array
        """
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        x += 0.5  # Clips to [0, 1]
        x = np.clip(x, 0, 1)

        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')  # Converts to RGB array
        return x

    # Regularization Functions
    @staticmethod
    def L1(tensor, _lambda):
        """
        Returns the L1 norm of the input tensor weighted by lambda
        """
        return _lambda * K.sum(K.abs(tensor))

    @staticmethod
    def L2(tensor, _lambda):
        """
        Returns the L2 norm of the input tensor weighted by lambda
        """
        return _lambda * K.sum(K.square(tensor))


class Jitter(object):
    def __init__(self, size, frequency = 2):
        """
        Initialized with a jitter size and frequency to
        apply it
        """
        self.size = size
        self.frequency = frequency

    def __call__(self, image, step):
        """
        Jitters the input image
        """
        if step % self.frequency == 0:
            ox, oy = np.random.randint(-self.size, self.size + 1, 2)
            return np.roll(np.roll(image, ox, -1), oy, -2)
        # Don't do anything
        return image


class Blur(object):
    def __init__(self, sigma, frequency=5):
        """
        Initialized with a sigma and frequency to apply it
        """
        self.sigma = sigma
        self.frequency = frequency

    def __call__(self, image, step):
        """
        Blurs the input image
        """
        if step % self.frequency == 0:
            return gaussian(image[0], self.sigma)[np.newaxis, ::]
        # Don't do anything
        return image


class Scale(object):
    def __init__(self, frequency=10, end=infinity):
        """Initialized with a frequency and when to stop applying it"""
        self.frequency = frequency
        self.end = end

    def __call__(self, image, step):
        """
        Randomly scales the input image
        """
        if step % self.frequency == 0 and step < self.end:
            ref_shape = image.shape
            pos = np.random.randint(0, 3)
            scales = [1, 1.0025, 1.001]
            image = rescale(image[0], scales[pos])[0:dim, 0:dim][np.newaxis, ::]
            if image.shape != ref_shape:
                image = resize(image[0], (224, 224))[np.newaxis, ...]
            return image
        # Don't do anything
        return image


class Visualizer(object):
    """
    Main class to run a feature visualisation
    """
    def __init__(self, model):
        """
        Creates a visualisation object for the given model.
        """
        self.model = model
        self._loss_is_set = False
        self.feeder = None

    def __repr__(self):
        return "Feature Visualizer - model: " + str(self.model.name)

    def set_layer_filter(self, layer_names, filter_idxs, weights=[1.], regularizer=(Utils.L1, 0.1), activation=False):
        """
        Sets the loss function to optimize

        ATTENTION - CURRENTLY ONLY WORKS FOR FINAL PREDICTIONS LAYER and MULTI-LOSS NOT TESTED

        Input:
            layer_name - layer name or list of layer name of model to measure
            filter - filter or list of filtto check activation
            regularizer - Either L1 or L2 function with lambda as tuple. default (L1, 0.1)

        """
        # Check for single term loss
        if type(layer_names) != list:
            layer_names = [layer_names]
        if type(filter_idxs) != list:
            filter_idxs = [filter_idxs]

        # Create loss
        loss = 0
        for i,layer in enumerate(layer_names):
            if activation:
                layer_output = self.model.get_layer(layer)
            else:
                # Get layer output before activation function
                layer_output = self.model.get_layer(layer).output.op.inputs[0]

            # Weighted loss defaults to equal weights
            if i > len(weights)-1:
                w = 1
            else:
                w = weights[i]
            loss += K.mean(layer_output[0, filter_idxs[i]]) * w

            # Add a regularization term to the loss
            if regularizer:
                func = regularizer[0]
                _lambda = regularizer[1]

                loss -= func(layer_output[0, filter_idxs[i]], -_lambda)

        # Create backend function to get grads
        gradients = K.gradients(loss, self.model.get_input_at(0))[0]
        # Normalize gradients (a trick that makes it work!)
        gradients /= (K.sqrt(K.mean(K.square(gradients))) + 1e-5)  # add 1e-5 to avoid dividing by zero
        #
        # Fetching Numpy output values given Numpy input values
        self.feeder = K.function(inputs=[model.get_input_at(0)], outputs=[loss, gradients])

        # Signal loss
        self._loss_is_set = True

    def optimize(self, max_iter=200, transforms=[], learning_rate=6000, verbose=True, maximize=True,
                 l2_lambda=0.0001, gradient_blur_sigma=0.1):
        """
        Optimize for the pre-set loss. Raises exception if no loss function is set.

        Inputs:
            max_iter - the number of iterations in the optimization
            maximize - the step direction e.g. minimize versus maximise
            learning_rate - the learning rate / gradient step size. default is 6000
                    though it is not quite understood how this effects the learning
                    when it is sufficiently high. It is slow when small.

        Output:
            image_out - optimized input image as numpy array as type uint8
        """
        if not self._loss_is_set:
            raise Exception("Failed to Optimize: You need to set a loss objective")

        # Generate random input image
        dim = self.model.input.shape[1]
        input_img_data = np.random.random((1, dim, dim, 3))*20

        print("Optimizing on loss objective...")

        # Optimize
        for step in range(max_iter):

            # Compute gradients
            loss_value, gradients = self.feeder([input_img_data])

            # Change the grads until signal found
            if np.sum(gradients) == 0:
                gradients = np.random.random((1, dim, dim, 3)) * 5

            # Remove small gradients with L2
            gradients = gradients - l2_lambda * np.sqrt(np.sum(np.square(gradients)))

            # Blur Gradients - remove checker board effects
            gradients = gaussian(gradients[0], gradient_blur_sigma)[np.newaxis, ::]

            # Update input image
            if maximize:
                input_img_data += gradients * (learning_rate / np.mean(np.abs(gradients)))
            else:
                input_img_data -= gradients * (learning_rate / np.mean(np.abs(gradients)))

            if verbose:
                # Print progress
                print("Step {0} of {1} - Loss = {2}".format(step+1, max_iter, loss_value))

            # Perform robustness transforms
            for transform in transforms:
                input_img_data = transform(input_img_data, step)

        # De-process image
        optimised_image = Utils.deprocess_image(input_img_data[0])

        return optimised_image


if __name__ == "__main__":

    from keras.applications.vgg16 import VGG16

    # Load the model to visualize
    dim = 224
    model = VGG16(input_shape=(dim, dim, 3), include_top=True, weights='imagenet')

    # Create visualizer
    visualizer = Visualizer(model)

    # Set which layer / filters to maximize
    visualizer.set_layer_filter("predictions", 20,
                                weights=[1.],
                                regularizer=(Utils.L1, 0.1),
                                activation=False
                                )

    # Optimize for 300 steps and apply transformations
    image = visualizer.optimize(max_iter=300,
                                transforms=[Blur(1), Jitter(2), Scale(10)]
                                )
    plt.imshow(image)
    plt.show()













