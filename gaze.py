'''
This class deals with the GazeEstimation model and the various operations associated with it.
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
    def __init__(self, path, device='CPU', extensions=None):
        '''
        TODO: Set instance variables.
        '''

        # TODO: Initialize attributes
        self.network = None
        self.core = None
        self.input = None
        self.output = None
        self.exec_net = None
        self.device = device
        self.extension = extensions
        self.count = 1

        # TODO: Save path of .bin and .xml files of model
        # gaze_xml = os.path.abspath('../intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml')
        gaze_xml = os.path.abspath(path)
        gaze_bin = os.path.splitext(gaze_xml)[0]+'.bin'

        # TODO: Initialize IENetwork object
        try:
            self.network = IENetwork(gaze_xml, gaze_bin)
        except Exception as e:
            print('Error occurred, refer `CPC.log` file for details')
            log.error('Gaze Estimation IENetwork object could not be initialized/loaded. Check if model files are stored in correct path.', e)

        # Initialize the three inputs that the model requires
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

            # TODO: Initialize IECore object and load the network as ExecutableNetwork object
        try:
            self.core = IECore()
            self.exec_net = self.core.load_network(network= self.network, device_name= self.device, num_requests= 1)
            if self.extension is not None:
                self.core.add_extension(extension_path= self.extension, device_name= self.device)
        except Exception as e:
            print('Error occurred, refer `CPC.log` file for details')
            log.error('Gaze Estimation IECore object could not be initialized/loaded.', e)
        return



    def predict(self, axes_arr, left_eye, right_eye, pflag):

        try:
            gaze_input = {self.head_pose_angles: axes_arr, self.left_eye_image: left_eye, self.right_eye_image: right_eye}
            gaze_result = self.exec_net.infer(gaze_input)

            if pflag == 1 and (self.count == 1 or self.count%5) == 0:
                perf_count = self.exec_net.requests[0].get_perf_counts()
                self.get_model_perf(perf_count, self.count)
                
            self.count += 1


        except Exception as e:
            print('Error occurred, refer `CPC.log` file for details')
            log.error('Gaze Estimation inference error: ', e)

        print(gaze_result)
        return gaze_result

    def get_model_perf(self, perf_count, count):
        with open('./model_perf/gaze.txt', 'a') as fh:
            fh.write('Frame: '+ str(count) + '\n\n')
        for layer in perf_count:
            if perf_count[layer]['status'] == 'EXECUTED':
                perf_dict = {'index': perf_count[layer]['execution_index'], 'layer_name': layer,  'exec_time': perf_count[layer]['cpu_time'] }
                with open('./model_perf/gaze.txt', 'a') as fh:
                    fh.write(str(perf_dict))
                    fh.write('\n')


    def check_model(self):
        log.info('gaze model inputs: {0}, {1}, {2}'.format(self.head_pose_angles, self.left_eye_image, self.right_eye_image))
        log.info('head pose shape: {0}'.format(self.head_pose_angles_shape))
        log.info('eye images shape: {0}'.format( self.left_eye_image_shape))
        log.info('gaze output: {0}'.format(self.output))
        log.info('gaze output shape: {0}'.format( self.output_shape))

        supported_layers = self.core.query_network(network= self.network, device_name= self.device)

        ### TODO: Check for any unsupported layers, and let the user
        ###       know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.warning("Unsupported layers found: {}".format(unsupported_layers))
            sys.exit("Add necessary extension for given hardware")


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function does that.
        '''
        ## TODO: Ignore frame if input frame has invalid/dubious dimensions
        try:
            temp = image.copy()
            temp = cv2.resize(temp, (self.left_eye_image_shape[3], self.left_eye_image_shape[2]) ) # n,c,h,w
            temp = temp.transpose((2, 0, 1))
            temp = temp.reshape(1, *temp.shape)
        except:
            log.warning('Frame ignored')
        return temp


    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function does that.
        '''
        output = np.squeeze(outputs['gaze_vector'])
        x = output[0]
        y = output[1]
        z = output[2]
        return x, y, z
