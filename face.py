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

class FaceDetection:
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
        face_xml = os.path.abspath('../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml')
        face_bin = os.path.splitext(face_xml)[0]+'.bin'

        # TODO: Initialize IENetwork object
        try:
            self.network = IENetwork(face_xml, face_bin)
        except Exception as e:
            log.info('Face Detection IENetwork object could not be initialized/loaded. Check if model files are stored in correct path.', e)

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


            # TODO: Initialize IECore object and load the network as ExecutableNetwork object
        try:
            self.core = IECore()
            self.exec_net = self.core.load_network(network= self.network, device_name= 'CPU', num_requests= 1)
        except Exception as e:
            log.info('Face Detection IECore object could not be initialized/loaded.', e)
        return

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        try:
            face_input = {self.input: image}
            face_result = self.exec_net.infer(face_input)
            face_result = face_result['detection_out']
        except Exception as e:
            log.info('Face Detection inference error: ', e)
        return face_result

    def check_model(self):
        log.info('Face Model Input shape: {0} '.format( str(self.input_shape) ))
        log.info('Face Model Output shape: {0}'.format( str(self.output_shape) ))
        pass

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        temp = image.copy()
        temp = cv2.resize(temp, (self.input_shape[3], self.input_shape[2] ) ) # n,c,h,w
        temp = temp.transpose((2, 0, 1))
        temp = temp.reshape(1, *temp.shape)
        return temp

    def preprocess_output(self, frame, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        width =  int(frame.shape[1]) #1920
        height = int(frame.shape[0]) #1080
        # width =  1920
        # height = 1080
        print('Post results', (width, height))
        output = np.squeeze(outputs)[0]
        # print('face op normalized: ', output[3], output[4], output[5], output[6])
        x_min = int(output[3] * width)
        y_min = int(output[4] * height)
        x_max = int(output[5] * width)
        y_max = int(output[6] * height)

        f_center = (x_min + width / 2, y_min + height / 2, 0)

        # print('face op denormalized',x_min, y_min, x_max, y_max)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        print((x_min, y_min), (x_max, y_max))
        # cv2.imshow('Computer Pointer Controller', frame)
        face = frame[y_min:y_max, x_min:x_max]
        return frame, face, f_center
