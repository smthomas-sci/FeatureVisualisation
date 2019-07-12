"""
Utilites for feature visualisation

Author: Simon Thomas
Email: simon.thomas@uq.edu.au
Date: 5th July 2019

"""

import numpy as np
import sklearn.decomposition


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


"""Helper for using sklearn.decomposition on high-dimensional tensors.
Provides ChannelReducer, a wrapper around sklearn.decomposition to help them
apply to arbitrary rank tensors. It saves lots of annoying reshaping.
"""

class ChannelReducer(object):
  """Helper for dimensionality reduction to the innermost dimension of a tensor.
  This class wraps sklearn.decomposition classes to help them apply to arbitrary
  rank tensors. It saves lots of annoying reshaping.
  See the original sklearn.decomposition documentation:
  http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
  """

  def __init__(self, n_components=3, reduction_alg="NMF", **kwargs):
    """Constructor for ChannelReducer.
    Inputs:
      n_components: Numer of dimensions to reduce inner most dimension to.
      reduction_alg: A string or sklearn.decomposition class. Defaults to
        "NMF" (non-negative matrix facotrization). Other options include:
        "PCA", "FastICA", and "MiniBatchDictionaryLearning". The name of any of
        the sklearn.decomposition classes will work, though.
      kwargs: Additional kwargs to be passed on to the reducer.
    """

    if not isinstance(n_components, int):
      raise ValueError("n_components must be an int, not '%s'." % n_components)

    # Defensively look up reduction_alg if it is a string and give useful errors.
    algorithm_map = {}
    for name in dir(sklearn.decomposition):
      obj = sklearn.decomposition.__getattribute__(name)
      if isinstance(obj, type) and issubclass(obj, sklearn.decomposition.base.BaseEstimator):
        algorithm_map[name] = obj
    if isinstance(reduction_alg, str):
      if reduction_alg in algorithm_map:
        reduction_alg = algorithm_map[reduction_alg]
      else:
        raise ValueError("Unknown dimensionality reduction method '%s'." % reduction_alg)


    self.n_components = n_components
    self._reducer = reduction_alg(n_components=n_components, **kwargs)
    self._is_fit = False

  @classmethod
  def _apply_flat(cls, f, acts):
    """Utility for applying f to inner dimension of acts.
    Flattens acts into a 2D tensor, applies f, then unflattens so that all
    dimesnions except innermost are unchanged.
    """
    orig_shape = acts.shape
    acts_flat = acts.reshape([-1, acts.shape[-1]])
    new_flat = f(acts_flat)
    if not isinstance(new_flat, np.ndarray):
      return new_flat
    shape = list(orig_shape[:-1]) + [-1]
    return new_flat.reshape(shape)

  def fit(self, acts):
    self._is_fit = True
    return ChannelReducer._apply_flat(self._reducer.fit, acts)

  def fit_transform(self, acts):
    self._is_fit = True
    return ChannelReducer._apply_flat(self._reducer.fit_transform, acts)

  def transform(self, acts):
    return ChannelReducer._apply_flat(self._reducer.transform, acts)

  def __call__(self, acts):
    if self._is_fit:
      return self.transform(acts)
    else:
      return self.fit_transform(acts)

  def __getattr__(self, name):
    if name in self.__dict__:
      return self.__dict__[name]
    elif name + "_" in self._reducer.__dict__:
      return self._reducer.__dict__[name+"_"]

  def __dir__(self):
    dynamic_attrs = [name[:-1]
                     for name in dir(self._reducer)
                     if name[-1] == "_" and name[0] != "_"
                    ]

    return list(ChannelReducer.__dict__.keys()) + list(self.__dict__.keys()) + dynamic_attrs



def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[:]
    else:
        fx = np.fft.fftfreq(w)[:]
    return np.sqrt(fx * fx + fy * fy)

def fft_image(shape, sd=None, decay_power=1):
    """An image paramaterization using 2D Fourier coefficients."""

    sd = sd or 0.01
    batch, h, w, ch = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (2, ch) + freqs.shape

    # Randomise the values
    init_val = np.random.normal(size=init_val_size, scale=sd).astype(np.float32)
    # Compose of real and imaginary parts
    spectrum_t = init_val[0] + init_val[1]*1j

    # Scale the spectrum. First normalize energy, then scale by the square-root
    # of the number of pixels to get a unitary transformation.
    # This allows to use similar leanring rates to pixel-wise optimisation.
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale *= np.sqrt(w * h)
    scaled_spectrum_t = scale * spectrum_t

    image_t = np.fft.ifft2(np.transpose(scaled_spectrum_t, (1, 2, 0))).real

    return image_t # Magic constant

