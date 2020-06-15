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


class HeadPoseEstimation:
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
        head_pose_xml = os.path.abspath('../intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml')
        head_pose_bin = os.path.splitext(head_pose_xml)[0]+'.bin'

        # TODO: Initialize IENetwork object
        self.network = IENetwork(head_pose_xml, head_pose_bin)

        self.input = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input].shape
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
            raise NotImplementedError('head_pose Detection Model could not be initialized/loaded.', e)
        return


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        head_pose_input = {self.input: image}
        head_pose_result = self.exec_net.infer(head_pose_input)
        #print(head_pose_result)
        # head_pose_result = np.squeeze(head_pose_result['95']) #['detection_out'] #CHECK THIS
        return head_pose_result

    def check_model(self):
        print('head_pose Model Input shape: ', self.input_shape)
        print('head_pose Model Output shape: ', self.output_shape)


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        temp = image.copy()
        # print('preprocess shape: ',  temp.shape)
        temp = cv2.resize(temp, (self.input_shape[3], self.input_shape[2] ) ) # n,c,h,w
        temp = temp.transpose((2, 0, 1))
        temp = temp.reshape(1, *temp.shape)
        # print('post process shape: ',temp.shape)
        return temp

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        pitch = np.squeeze(outputs['angle_p_fc'])
        roll = np.squeeze(outputs['angle_r_fc'])
        yaw = np.squeeze(outputs['angle_y_fc'])
        axes_op = np.array([[pitch, roll, yaw]])


        return axes_op
