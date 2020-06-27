'''
This class deals with the FaceDetection model and the various operations associated with it.
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
        # face_xml = os.path.abspath('../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml')
        face_xml = os.path.abspath(path)
        face_bin = os.path.splitext(face_xml)[0]+'.bin'

        # TODO: Initialize IENetwork object
        try:
            self.network = IENetwork(face_xml, face_bin)
        except Exception as e:
            print('Error occurred, refer `CPC.log` file for details')
            log.error('Face Detection IENetwork object could not be initialized/loaded. Check if model files are stored in correct path.', e)

        self.input = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input].shape
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
            log.error('Face Detection IECore object could not be initialized/loaded.', e)
        return

    def predict(self, image, pflag):

        # TODO: Predict and dump perf_count if pflag
        try:
            face_input = {self.input: image}
            face_result = self.exec_net.infer(face_input)

            # TODO: Post stats for every 5th frame
            if pflag == 1 and (self.count == 1 or self.count%5 == 0):
                perf_count = self.exec_net.requests[0].get_perf_counts()
                self.get_model_perf(perf_count, self.count)

            self.count += 1

            face_result = face_result['detection_out']

        except Exception as e:
            print('Error occurred, refer `CPC.log` file for details')
            log.error('Face Detection inference error: ', e)
        return face_result

    # TODO: Method to dump perf_count stats
    def get_model_perf(self, perf_count, count):
        with open('./model_perf/face.txt', 'a') as fh:
            fh.write('Frame: '+ str(count) + '\n\n')
        for layer in perf_count:
            # TODO: Check for layers with `EXECUTED` status only
            if perf_count[layer]['status'] == 'EXECUTED':
                # TODO: Extract major parameters
                perf_dict = {'index': perf_count[layer]['execution_index'], 'layer_name': layer,  'exec_time': perf_count[layer]['cpu_time'] }
                with open('./model_perf/face.txt', 'a') as fh:
                    fh.write(str(perf_dict))
                    fh.write('\n')

    def check_model(self):
        log.info('Face Model Input shape: {0} '.format( str(self.input_shape) ))
        log.info('Face Model Output shape: {0}'.format( str(self.output_shape) ))

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
        temp = image.copy()
        temp = cv2.resize(temp, (self.input_shape[3], self.input_shape[2] ) ) # n,c,h,w
        temp = temp.transpose((2, 0, 1))
        temp = temp.reshape(1, *temp.shape)
        return temp

    def preprocess_output(self, frame, outputs, vf):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function does that.
        '''
        vf = bool(vf)
        width =  int(frame.shape[1]) #1920
        height = int(frame.shape[0]) #1080
        output = np.squeeze(outputs)[0]
        x_min = int(output[3] * width)
        y_min = int(output[4] * height)
        x_max = int(output[5] * width)
        y_max = int(output[6] * height)

        f_center = (x_min + width / 2, y_min + height / 2, 0)

        # TODO: Visualize output if vflag
        if vf is True:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        face = frame[y_min:y_max, x_min:x_max]
        return frame, face, f_center
