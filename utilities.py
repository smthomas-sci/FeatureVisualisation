"""
Utilites for feature visualisation

Author: Simon Thomas
Email: simon.thomas@uq.edu.au
Date: 14th June 2019

"""

import numpy as np


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


