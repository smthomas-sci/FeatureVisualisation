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

from .regularizers import *
from .utilities import *


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
        self.objective = None

    def __repr__(self):
        return "Feature Visualizer - model: " + str(self.model.name)

    def set_layer_filter(self, layer_names, filter_idxs, rows=None, cols=None, channels=None,  weights=[1.], regularizer=(L1, 0.1), activation=False):
        """
        Sets the loss function to optimize

        ATTENTION - MULTI-LOSS NOT TESTED

        Input:
            layer_name - layer name or list of layer name of model to measure
            filter_idxs - filter or list of filters to get activation
            rows - row start and end [5, 10] == tensor[5:10, :, :]
            cols - col start and end [5, 10] == tensor[:, 5:10, :]
            channels - channel start and end [5, 10] == tensor[:, :, 5:10]
            weights - list of weights to balance a multi layered loss
            regularizer - Either L1 or L2 function with lambda as tuple. default (L1, 0.1)
            activation - whether to include the activation function. By default False to use logits

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

            # Create the loss tensor
            # <= 3D layer
            if rows or cols:
                if channels:
                    loss_tensor = layer_output[0, rows[0]:rows[-1], cols[0]:cols[-1], channels[0]:channels[-1]]
                else:
                    # <= 2D layer
                    loss_tensor = layer_output[0, rows[0]:rows[-1], cols[0]:cols[-1], filter_idxs[i]]
            else:

                if len(layer_output.shape) == 2:
                    # = 1D layer - only for Dense layers such as "predictions"
                    if len(filter_idxs) > 0:
                        loss_tensor = layer_output[0, filter_idxs[i]]
                    else:
                        loss_tensor = layer_output[0, :]  # whole layer
                else:
                    # =2D layer - want the whole activation
                    if len(filter_idxs) > 0:
                        loss_tensor = layer_output[0, :, :, filter_idxs[i]]
                    else:
                        loss_tensor = layer_output[0, :, :, :]  # whole layer


            loss += K.mean(loss_tensor) * w

            # Add a regularization term to the loss
            if regularizer:
                func = regularizer[0]
                _lambda = regularizer[1]

                loss -= func(loss_tensor, -_lambda)

        # Create backend function to get grads
        gradients = K.gradients(loss, self.model.get_input_at(0))[0]
        # Normalize gradients (a trick that makes it work!)
        gradients /= (K.sqrt(K.mean(K.square(gradients))) + 1e-5)  # add 1e-5 to avoid dividing by zero
        #
        # Fetching Numpy output values given Numpy input values
        self.feeder = K.function(inputs=[self.model.get_input_at(0)], outputs=[loss, gradients])

        # Signal loss
        self._loss_is_set = True
        self.objective = "Objective - Layer(s) = " + str(layer_names) + " - neuron/filters(s) =  " + str(filter_idxs)

    def optimize(self, max_iter=200, transforms=[], learning_rate=6000, verbose=True, fourier=False,
                 l2_lambda=0.0001, gradient_blur_sigma=0.1, image_dim=None, input_image=None):
        """
        Optimize for the pre-set loss. Raises exception if no loss function is set.

        Inputs:
            max_iter - the number of iterations in the optimization
            maximize - the step direction e.g. minimize versus maximise
            learning_rate - the learning rate / gradient step size. default is 6000
                    though it is not quite understood how this effects the learning
                    when it is sufficiently high. It is slow when small.
            input_image - input image to optimize - (Deep Dream)

        Output:
            image_out - optimized input image as numpy array as type uint8
        """
        if not self._loss_is_set:
            raise Exception("Failed to Optimize: You need to set a loss objective")

        # Set input image
        if input_image is not None:
            input_img_data = input_image[np.newaxis, ::] / 255.
            dim = input_img_data.shape[1]
        else:
            # Generate random input image
            dim = self.model.get_input_at(0).shape[1].value
            if dim == None:
                dim = image_dim

            if fourier:
                input_img_data = fft_image((1, dim, dim, 3), sd=None, decay_power=1)[np.newaxis, ::]
            else:
                input_img_data = np.random.random((1, dim, dim, 3))*20

        print("Optimizing on loss objective...", self.objective)

        # Optimize
        for step in range(max_iter):

            # Compute gradients
            loss_value, gradients = self.feeder([input_img_data])

            # Change the grads until signal found
            #if np.sum(gradients) == 0:
            #    gradients += np.random.random((1, dim, dim, 3))*50

            # Remove small gradients with L2
            gradients = gradients - l2_lambda * np.sqrt(np.sum(np.square(gradients)))

            # Blur Gradients - remove checker board effects
            gradients = gaussian(gradients[0], gradient_blur_sigma)[np.newaxis, ::]

            # Update input image
            input_img_data += gradients * (learning_rate / np.mean(np.abs(gradients)))

            if verbose:
                # Print progress
                if step % 10 == 0:
                    print("Step {0} of {1} - Loss = {2}".format(step+1, max_iter, loss_value))

            # Perform robustness transforms
            for transform in transforms:
                input_img_data = transform(input_img_data, step)

        # Deep dream
        #if np.any(input_image):
        #    return input_img_data[0]

        # Random optimize - # De-process image
        optimised_image = deprocess_image(input_img_data[0])

        return optimised_image


if __name__ == "__main__":

    from keras.applications.vgg16 import VGG16

    import skimage.io as io

    # Load the model to visualize
    dim = 1024
    #model = VGG16(input_shape=(dim, dim, 3), include_top=True, weights='imagenet')
    model = VGG16(input_shape=(dim, dim, 3), include_top=False, weights='imagenet')
    # Create visualizer
    visualizer = Visualizer(model)

    # -------------- CLASS ACTIVATIONS -------------- #

    # # Set which layer / filters to maximize
    # visualizer.set_layer_filter("predictions", 20,
    #                             weights=[1.],
    #                             regularizer=(L1, 0.1),
    #                             activation=False
    #                             )
    #
    # # Optimize for 300 steps and apply transformations
    # image = visualizer.optimize(max_iter=300,
    #                             transforms=[
    #                                 Blur(1), Jitter(2), Scale(10)
    #                             ])
    # # ------------------------------------------------ #

    # -------------- FILTER ACTIVATIONS -------------- #

    for i in [351]:#range(512):
        # Set which layer / filters to maximize
        visualizer.set_layer_filter("block5_pool", [i],
                                    regularizer=(L1, 0.1),
                                    activation=False
                                    )

        # Optimize for 300 steps and apply transformations
        image = visualizer.optimize(max_iter=600,
                                    transforms=[
                                        Blur(1), Jitter(2), Scale(10)
                                    ],
                                    learning_rate=0.05)
        # Save image
        print("Saving...")
        #io.imsave("./block5_pool/block5_pool_filter_{0}.png".format(i), image)
        io.imsave("./block5_pool_filter_{0}.png".format(i), image)













