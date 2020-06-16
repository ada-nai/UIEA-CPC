'''
This script is the driver for the project, which takes input from one script and provides output to another.
'''
import os
import time
import sys
import cv2
import numpy as np
from argparse import ArgumentParser
import logging as log

from input_feeder import InputFeeder
from face import FaceDetection
from landmark import LandmarkDetection
from head_pose import HeadPoseEstimation
from gaze import GazeEstimation
from mouse_controller import MouseController





def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser('CPC App')

    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    # parser.add_argument("-l", "--cpu_extension", required=False, type=str,
    #                     default='/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so',
    #                     help="MKLDNN (CPU)-targeted custom layers."
    #                          "Absolute path to a shared library with the"
    #                          "kernels impl.")
    # parser.add_argument("-d", "--device", type=str, default="CPU",
    #                     help="Specify the target device to infer on: "
    #                          "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
    #                          "will look for a suitable plugin for device "
    #                          "specified (CPU by default)")
    # parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
    #                     help="Probability threshold for detections filtering"
    #                     "(0.5 by default)")
    args = parser.parse_args()

    return args


def flow(args):
    # TODO: Load all the models

    face = FaceDetection()
    face.load_model()
    face.check_model()

    landmark = LandmarkDetection()
    landmark.load_model()
    landmark.check_model()

    head_pose = HeadPoseEstimation()
    head_pose.load_model()
    head_pose.check_model()

    gaze_estimation = GazeEstimation()
    gaze_estimation.load_model()
    gaze_estimation.check_model()

    log.info('Models loaded')

    # TODO: Initialize InputFeeder object
    feed = InputFeeder(input_type= 'video', input_file= args.input)
    log.info('InputFeeder object initialized.')

    # TODO: load the data from source
    log.info('Loading data...')
    feed.load_data()
    log.info('Data Loaded.')
    for batch in feed.next_batch():
        # if batch is not None:
        # cv2.imshow('CPC', batch)
        # print('batch', batch.shape)
        key = cv2.waitKey(1000)
        if key == ord('x'):
            log.warning('KeyboardInterrupt \n Feed closed')
            raise KeyboardInterrupt('X was pressed')

        # else:
        #     print('Stream ended or error')
        #     sys.exit()
        # print('batch type:' ,type(batch))


        # TODO: send frame to face detection model
        face_ip = face.preprocess_input(batch)
        # print('face ip dims', face_ip.shape)
        face_op = face.predict(face_ip)
        batch, face_box = face.preprocess_output(batch, face_op)
        # cv2.imshow('CPC', face_box)


        # TODO: send face results to landmark
        landmark_ip = landmark.preprocess_input(face_box)
        # print('face ip dims', landmark_ip.shape)
        landmark_op = landmark.predict(landmark_ip)
        # print(landmark_op) #, landmark_op.shape)
        batch, lEye, rEye = landmark.preprocess_output(batch, landmark_op)
        cv2.imshow('CPC', batch)
        # log.info('eye shape - rEye: ', rEye.shape, 'lEye: ', lEye.shape)


        # TODO: send face results to head_pose
        head_pose_ip = head_pose.preprocess_input(face_box)
        head_pose_op = head_pose.predict(head_pose_ip)
        axes_op = head_pose.preprocess_output(head_pose_op)
        # print(axes_op, axes_op.shape)

        # TODO: send landmark and head_pose results to gaze
        gaze_lEye = gaze_estimation.preprocess_input(lEye)
        gaze_rEye = gaze_estimation.preprocess_input(rEye)
        # print(gaze_lEye.shape, gaze_rEye.shape)
        gaze_op = gaze_estimation.predict(axes_op, gaze_lEye, gaze_rEye)
        x, y, z = gaze_estimation.preprocess_output(gaze_op)
        # TODO: send gaze results to mouse_controller
        mc = MouseController('high', 'fast')
        mc.move(x, y)

    feed.close()
    log.info('Feed closed')




def main():
    """
    Parse the arguments
    """
    # Grab command line args
    args = build_argparser()
    log.basicConfig(filename='CPC.log',level=log.INFO)
    flow(args)



if __name__ == '__main__':
    main()
