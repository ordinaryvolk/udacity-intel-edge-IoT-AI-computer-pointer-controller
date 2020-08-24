'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import sys
import cv2
import time
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Models:
    '''
    Class for the Face Detection Model.

    This is a single model class that can handle 4 types of models
    used in this application:


    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.plugin = None
        self.network = None
        self.net_plugin = None

        self.input_blob = None
        self.output_blob = None
        self.input_shape = None
        self.image_width = 0
        self.image_height = 0

        self.model_type = None
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.prob_threshold = None


    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IECore()
        model_bin = os.path.splitext(self.model_name)[0] + ".bin"
        
        # Read network
        self.network = self.plugin.read_network(model=self.model_name, weights=model_bin)

        ### Check for supported layers ###
        self.check_model()

        ### Add any necessary extensions ###
        ### Not needed as my local openvino is 2020.R4

        ### Return the loaded inference plugin ###
        self.net_plugin = self.plugin.load_network(self.network, self.device)

        # Get the input &output layer blob
        self.input_blob = next(iter(self.network.input_info))
        self.output_blob = next(iter(self.network.outputs))

        # Get input shape
        self.get_input_shape()

        return self.net_plugin

    def get_input_shape(self):
        ### Set the shape of the input layer ###

        input_shapes = {}
        for network_input in self.network.input_info:
            input_shapes[network_input] = (self.network.input_info[network_input].input_data.shape)

        self.input_shape = input_shapes

    def predict(self, frame, threshold):
        '''
        This method is meant for running predictions on the input image.
        '''
        prepropossed_frame = self.preprocess_input(frame)
        
        start_time = time.time()

        inference_output = self.net_plugin.infer(prepropossed_frame)

        inference_time = (time.time() - start_time) * 1000

        return self.preprocess_output(frame, inference_output, threshold, inference_time) 
 
    def check_model(self):
        '''
        Check for supported layers
        '''

        # Check supported layers
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)

        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
   
    def preprocess_input(self, frame):
        '''
        Preprocess the raw input frame to prepare for inferencing
        '''

        self.image_height = frame.shape[0]
        self.image_width  = frame.shape[1]
        preprossed_frame = cv2.resize(frame, (self.input_shape[self.input_blob][3], self.input_shape[self.input_blob][2]))
        preprossed_frame = preprossed_frame.transpose((2,0,1))
        b  = 1
        c  = preprossed_frame.shape[0]
        h  = preprossed_frame.shape[1]
        w  = preprossed_frame.shape[2]
        preprossed_frame   = preprossed_frame.reshape(b, c, h, w)

        preprossed_network = {self.input_blob:preprossed_frame}

        return preprossed_network


    def preprocess_output(self, frame, outputs, threshold, inference_time):
        '''
        Note: placeholder. Overridden  in individual subclasses for their unique handlings 
        '''
        raise NotImplementedError
