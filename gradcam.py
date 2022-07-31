#!/usr/bin/env python3
import argparse
import glob
import numpy as np
import os
import sys

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import keras as keras
from keras.models import Model
from keras.preprocessing import image
from keras.utils import load_img,img_to_array
import tensorflow as tf
# from classification_models.tfkeras import Classifiers
# from skimage.segmentation import mark_boundaries

from pdb import set_trace as bp

import config as cnfg
import compiled_model as c_model

# Copied from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)
        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        image = image.astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)


def process_sample(img_path, dest_dir, model):
    img =load_img(img_path, target_size=(32, 32))
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    x = img_to_array(img)
    # batched = np.expand_dims(x, axis=0)
    # batched = preproc_func(batched)
    image = x.reshape(-1, 32, 32, 3)
    image = image.astype('float32')
    image /= 255
    preds = model.predict(image)[0]
    label = np.argmax(preds)
    cam = GradCAM(model, label)
    heatmap = cam.compute_heatmap(image)
    (heatmap, output) = cam.overlay_heatmap(heatmap, x, alpha=0.5)
    output = np.vstack([x, heatmap, output])
    cv2.imwrite(dest_dir + '/' + base_name + '.png', output)
    return preds[label]


def main():
    model_path = cnfg.model_path
    dataset_path = cnfg.csv_path
    dest_dir = cnfg.dest_dir

    if not os.path.exists(model_path):
        print('Model {} doesn\'t exist'.format(model_path))
    if not os.path.exists(dataset_path):
        print('Can\'t find dataset at {}'.format(model_path))
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    # samples = glob.glob(os.path.join(dataset_path, '*.jpeg'))
    sample='output0.png'
    # print('Found {} samples in {}'.format(len(samples), dataset_path))
    results_dict = {'path': [], 'score': []}
    # model_name = 'resnet'
    # _, preprocess_input = Classifiers.get(model_name)
    model = keras.models.load_model(model_path)

    score = process_sample(sample, dest_dir, model)
    results_dict['path'].append(sample)
    results_dict['score'].append(score)
    df = pd.DataFrame(results_dict)
    df.describe()
    df = df.sort_values(by='score', ascending=False)

    df.to_csv(dest_dir + '/scores.csv')


if __name__ == '__main__':
    main()