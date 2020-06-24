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


class LandmarkDetection:
    '''
    Class for the landmark Detection Model.
    '''
    def __init__(self, path, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        # TODO: Initialize attributes
        self.network = None
        self.core = None
        self.input = None
        self.output = None
        self.exec_net = None
        self.count = 1

        # TODO: Save path of .bin and .xml files of model
        # landmark_xml = os.path.abspath('../intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml')
        landmark_xml = os.path.abspath(path)
        print(landmark_xml)
        landmark_bin = os.path.splitext(landmark_xml)[0]+'.bin'

        # TODO: Initialize IENetwork object
        try:
            self.network = IENetwork(landmark_xml, landmark_bin)
        except Exception as e:
            print('Error occurred, refer `CPC.log` file for details')
            log.error('Landmark Detection IENetwork object could not be initialized/loaded. Check if model files are stored in correct path.', e)

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
            print('Error occurred, refer `CPC.log` file for details')
            log.error('landmark Detection IECore object could not be initialized/loaded.', e)
        return

    def predict(self, image, pflag):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        try:
            landmark_input = {self.input: image}
            landmark_result = self.exec_net.infer(landmark_input)
            landmark_result = np.squeeze(landmark_result['95']) #['detection_out'] #CHECK THIS

            if pflag == 1 and (self.count == 1 or self.count%5) == 0:
                perf_count = self.exec_net.requests[0].get_perf_counts()
                self.get_model_perf(perf_count, self.count)
                # print(perf_count)

            self.count += 1

        except Exception as e:
            print('Error occurred, refer `CPC.log` file for details')
            log.error('Landmark Detection inference error: ', e)
        return landmark_result

    def get_model_perf(self, perf_count, count):
        with open('./model_perf/landmark.txt', 'a') as fh:
            fh.write('Frame: '+ str(count) + '\n\n')
        for layer in perf_count:
            if perf_count[layer]['status'] == 'EXECUTED':
                # print('layer_name: ',layer)
                # perf_dict = {'layer_name': None, 'index': None, 'exec_time': None }
                perf_dict = {'index': perf_count[layer]['execution_index'], 'layer_name': layer,  'exec_time': perf_count[layer]['cpu_time'] }
                with open('./model_perf/landmark.txt', 'a') as fh:
                # fh.write('Frame: '+ str(count) + '\n')
                    fh.write(str(perf_dict))
                    fh.write('\n')

    def check_model(self):
        log.info('landmark Model Input shape: {0}'.format( str(self.input_shape) ))
        log.info('landmark Model Output shape: {0}'.format( str(self.output_shape) ))

        supported_layers = self.core.query_network(network= self.network, device_name="CPU")

        ### TODO: Check for any unsupported layers, and let the user
        ###       know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.warning("Unsupported layers found: {}".format(unsupported_layers))
            sys.exit("Add necessary extension for given hardware")

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
        # ('post process shape: ',temp.shape)
        return temp

    def preprocess_output(self, frame, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        temp_op = frame.copy()
        width = int(frame.shape[1]) #1920
        height = int(frame.shape[0]) #1080
        r_radius = 18
        c_radius = 5

        # print('landmark frame size: ', width, height)

        # print('landmark op normalized: ', outputs[0], outputs[1], outputs[2], outputs[3])
        #### HOLD
        # x_scale = 1920/367
        # y_scale = 1080/246
        x_lEye= int(outputs[0] * width)
        y_lEye = int(outputs[1] * height)
        x_rEye = int(outputs[2] * width)
        y_rEye = int(outputs[3] * height)
        x_nose = int(outputs[4] * width)
        y_nose = int(outputs[5] * height)
        x_lLip = int(outputs[6] * width)
        y_lLip = int(outputs[7] * height)
        x_rLip = int(outputs[8] * width)
        y_rLip = int(outputs[9] * height)

        # print('landmark op denormalized: ', x_lEye, y_lEye, x_rEye, y_rEye)

        # cv2.circle(temp_op, (x_lEye, y_lEye), c_radius, (0, 0, 255), 2)
        # cv2.circle(temp_op, (x_rEye, y_rEye), c_radius, (0, 0, 255), 2)
        # cv2.circle(temp_op, (x_nose, y_nose), c_radius, (0, 255, 0), 2)
        # cv2.circle(temp_op, (x_lLip, y_lLip), c_radius, (255, 0, 0), 2)
        # cv2.circle(temp_op, (x_rLip, y_rLip), c_radius, (255, 0, 0), 2)
        # cv2.imshow('class_frame', frame)

        # print(x_lEye, y_lEye)
        min_x_lEye= x_lEye-r_radius
        min_y_lEye = y_lEye-r_radius
        max_x_lEye= x_lEye+r_radius
        max_y_lEye = y_lEye+r_radius

        min_x_rEye = x_rEye-r_radius
        min_y_rEye = y_rEye-r_radius
        max_x_rEye = x_rEye+r_radius
        max_y_rEye = y_rEye+r_radius

        lEye = temp_op[y_lEye-r_radius : y_lEye+r_radius , x_lEye-r_radius : x_lEye+r_radius]
        cv2.rectangle(temp_op, (min_x_lEye, min_y_lEye), (max_x_lEye, max_y_lEye), (255, 0, 0), 2)

        cv2.imshow('lEye', lEye)
        rEye = temp_op[y_rEye-r_radius : y_rEye+r_radius , x_rEye-r_radius : x_rEye+r_radius]
        cv2.rectangle(temp_op, (min_x_rEye, min_y_rEye), (max_x_rEye, max_y_rEye), (0, 255, 0), 2)
        cv2.imshow('rEye', rEye)


        return temp_op, lEye, rEye
