"""
A module containing the transformations
that can be used to remove high frequency noise

Author: Simon Thomas
Email: simon.thomas@uq.edu.au
Date: 14th June, 2019

"""
from sys import maxsize as infinity

import numpy as np

from skimage.transform import rescale, resize
from skimage.filters import gaussian

import keras.backend as K





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
def L1(tensor, _lambda):
    """
    Returns the L1 norm of the input tensor weighted by lambda
    """
    return _lambda * K.sum(K.abs(tensor))


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
            image = rescale(image[0], scales[pos])[np.newaxis, ::]
            if image.shape != ref_shape:
                image = resize(image[0], (ref_shape[1], ref_shape[1]))[np.newaxis, ...]
            return image
        # Don't do anything
        return image
