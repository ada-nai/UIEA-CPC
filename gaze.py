'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import sys
import time
import logging as log
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore

class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''

        # TODO: Initialize attributes
        self.network = None
        self.core = None
        self.input = None
        self.output = None
        self.exec_net = None

        # TODO: Save path of .bin and .xml files of model
        gaze_xml = os.path.abspath('../intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml')
        gaze_bin = os.path.splitext(gaze_xml)[0]+'.bin'

        # TODO: Initialize IENetwork object
        self.network = IENetwork(gaze_xml, gaze_bin)

        gaze_iter = iter(self.network.inputs)
        self.head_pose_angles = next(gaze_iter)
        self.left_eye_image = next(gaze_iter)
        self.right_eye_image = next(gaze_iter)

        self.head_pose_angles_shape = self.network.inputs[self.head_pose_angles].shape
        self.left_eye_image_shape = self.network.inputs[self.left_eye_image].shape
        self.right_eye_image_shape = self.network.inputs[self.right_eye_image].shape

        self.output = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output].shape


    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        try:
            # TODO: Initialize IECore object and load the network as ExecutableNetwork object
            self.core = IECore()
            self.exec_net = self.core.load_network(network= self.network, device_name= 'CPU', num_requests= 1)
        except Exception as e:
            raise NotImplementedError('gaze Detection Model could not be initialized/loaded.', e)
        return



    def predict(self, axes_arr, left_eye, right_eye):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        gaze_input = {self.head_pose_angles: axes_arr, self.left_eye_image: left_eye, self.right_eye_image: right_eye}
        # try:
        gaze_result = self.exec_net.infer(gaze_input)
        # print(gaze_result)
        # except:
        #     print('Output could not be evaluated')

        # gaze_result = np.squeeze(gaze_result['95']) #['detection_out'] #CHECK THIS
        return gaze_result

    def check_model(self):
        log.info('gaze model inputs: ',self.head_pose_angles, self.left_eye_image, self.right_eye_image)
        log.info('head pose shape',self.head_pose_angles_shape)
        log.info('eye images shape:', self.left_eye_image_shape)
        log.info('gaze output: ', self.output)
        log.info('gaze output shape', self.output_shape)
        pass


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        try:
            temp = image.copy()
            # print('preprocess shape: ',  temp.shape)
            temp = cv2.resize(temp, (self.left_eye_image_shape[3], self.left_eye_image_shape[2]) ) # n,c,h,w
            temp = temp.transpose((2, 0, 1))
            temp = temp.reshape(1, *temp.shape)
            # print('post process shape: ',temp.shape)
        except:
            print('Frame ignored')
        return temp


    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        output = np.squeeze(outputs['gaze_vector'])
        x = output[0]
        y = output[1]
        z = output[2]
        return x, y, z
