'''
This class deals with the HeadPoseEstimation model and the various operations associated with it.
'''
import os
import sys
import time
import logging as log
import math
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore


class HeadPoseEstimation:
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
        # head_pose_xml = os.path.abspath('../intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml')
        head_pose_xml = os.path.abspath(path)
        head_pose_bin = os.path.splitext(head_pose_xml)[0]+'.bin'

        # TODO: Initialize IENetwork object
        try:
            self.network = IENetwork(head_pose_xml, head_pose_bin)
        except Exception as e:
            print('Error occurred, refer `CPC.log` file for details')
            log.error('Head Pose Estimation IENetwork object could not be initialized/loaded. Check if model files are stored in correct path.', e)

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
            log.error('Head Pose Estimation IECore object could not be initialized/loaded.', e)
        return


    def predict(self, image, pflag):

        # TODO: Predict and dump perf_count if pflag

        try:
            head_pose_input = {self.input: image}
            head_pose_result = self.exec_net.infer(head_pose_input)

            # TODO: Post stats for every 5th frame

            if pflag == 1 and (self.count == 1 or self.count%5) == 0:
                perf_count = self.exec_net.requests[0].get_perf_counts()
                self.get_model_perf(perf_count, self.count)

            self.count += 1

        except Exception as e:
            print('Error occurred, refer `CPC.log` file for details')
            log.error('Head Pose Estimation inference error: ', e)
        return head_pose_result

    # TODO: Method to dump perf_count stats

    def get_model_perf(self, perf_count, count):
        with open('./model_perf/head_pose.txt', 'a') as fh:
            fh.write('Frame: '+ str(count) + '\n\n')
        for layer in perf_count:
            # TODO: Check for layers with `EXECUTED` status only

            if perf_count[layer]['status'] == 'EXECUTED':
                # TODO: Extract major parameters

                perf_dict = {'index': perf_count[layer]['execution_index'], 'layer_name': layer,  'exec_time': perf_count[layer]['cpu_time'] }
                with open('./model_perf/head_pose.txt', 'a') as fh:
                    fh.write(str(perf_dict))
                    fh.write('\n')

    def check_model(self):
        log.info('head_pose Model Input shape: {0}'.format( str(self.input_shape) ))
        log.info('head_pose Model Output shape: {0}'.format(str(self.output_shape)))

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
        you might have to preprocess it. This function does that.
        '''
        temp = image.copy()
        temp = cv2.resize(temp, (self.input_shape[3], self.input_shape[2] ) ) # n,c,h,w
        temp = temp.transpose((2, 0, 1))
        temp = temp.reshape(1, *temp.shape)
        return temp

    # TODO: Head Pose output visualization method
    def draw_axes(self, frame, center_of_face, yaw, pitch, roll, scale, focal_length):
        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(pitch), -math.sin(pitch)],
                       [0, math.sin(pitch), math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                       [0, 1, 0],
                       [math.sin(yaw), 0, math.cos(yaw)]])
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                       [math.sin(roll), math.cos(roll), 0],
                       [0, 0, 1]])

        R = Rz @ Ry @ Rx
        camera_matrix = self.build_camera_matrix(center_of_face, focal_length)
        xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
        o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        o[2] = camera_matrix[0][0]
        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o
        xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
        xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
        xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
        yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
        p1 = (int(xp1), int(yp1))
        xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        cv2.circle(frame, p2, 3, (255, 0, 0), 2)
        return frame

    def build_camera_matrix(self, center_of_face, focal_length):
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        camera_matrix = np.zeros((3, 3), dtype='float32')
        camera_matrix[0][0] = focal_length
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = focal_length
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1
        return camera_matrix


    def preprocess_output(self, frame, outputs, f_center):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function does that.
        '''
        ## TODO: Pitch, Roll, Yaw angles
        pitch = np.squeeze(outputs['angle_p_fc'])
        roll = np.squeeze(outputs['angle_r_fc'])
        yaw = np.squeeze(outputs['angle_y_fc'])
        axes_op = np.array([pitch, roll, yaw])

        focal_length = 950.0
        scale = 50

        width =  int(frame.shape[1]) #1920
        height = int(frame.shape[0]) #1080

        frame = self.draw_axes(frame, f_center, yaw, pitch, roll, scale, focal_length)

        return frame, axes_op
