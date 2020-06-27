'''
This script is the driver for the project, which takes input from one model and provides output to another.
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
    parser = ArgumentParser('Computer Pointer Controller App')

    parser.add_argument("-i", "--input", required=False, type=str, default = None,
                        help="Path of video file, if applicable")
    parser.add_argument("-l", "--extension", required=False, type=str,
                        default=None,
                        help="MKLDNN targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-if", "--input_type", required=True, type=str, default="video",
                        help=" Input media format to the models. 'cam' or 'video'. Default set to 'video' ")

    parser.add_argument("-fd", "--face", required=False, type=str, default = '../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml',
                        help="Path of Face Detection model xml file")
    parser.add_argument("-ld", "--landmark", required=False, type=str, default = '../intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml',
                        help="Path of Landmark Detection model xml file")
    parser.add_argument("-hpe", "--head_pose", required=False, type=str, default = '../intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml',
                        help="Path of Head Pose Estimation model xml file")
    parser.add_argument("-ge", "--gaze", required=False, type=str, default = '../intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml',
                        help="Path of Gaze Estimation model xml file")

    parser.add_argument("-vf", "--visual_flag", required= False, type=str, default= 0,
                        help="Flag for visualization of model outputs. Can be 0 or 1. Default set to 0.")

    parser.add_argument("-pf", "--perf_flag", required= False, type=str, default= 0,
                        help="Flag for analyzing layer-wise performance of models. Can be 0 or 1. Default set to 0.")
    args = parser.parse_args()

    return args


def flow(args):
    ## TODO: Convert argument flags to bool values
    if args.visual_flag is None:
        args.visual_flag == 0
        log.info('Visual Flag not input, so set to `False`')
    vflag = bool(int(args.visual_flag))

    if args.perf_flag is None:
        args.perf_flag == 0
        log.info('Performance Flag not input, so set to `False`')
    pflag = bool(int(args.perf_flag))

    ## TODO: Variables for benchmarking
    tfproc = 0 # Total face process
    tfpred = 0 # Total face prediction
    tlproc = 0 # landmark
    tlpred = 0
    thproc = 0 # head_pose
    thpred = 0
    tgproc = 0 # gaze
    tgpred = 0
    count = 0 # frame count


    log.info('----------START OF PROGRAM----------')

    # TODO: Load all the models
    face_load_start = time.time()
    face = FaceDetection(args.face, args.device, args.extension)
    face.load_model()
    face_load_time = time.time() - face_load_start
    face.check_model()
    log.info('Face Detection object initialized')

    landmark_load_start = time.time()
    landmark = LandmarkDetection(args.landmark, args.device, args.extension)
    landmark.load_model()
    landmark_load_time = time.time() - landmark_load_start
    landmark.check_model()
    log.info('Landmark Detection object initialized ')

    head_pose_load_start = time.time()
    head_pose = HeadPoseEstimation(args.head_pose, args.device, args.extension)
    head_pose.load_model()
    head_pose_load_time = time.time() - head_pose_load_start
    head_pose.check_model()
    log.info('Head Pose Estimation object initialized')

    gaze_load_start = time.time()
    gaze_estimation = GazeEstimation(args.gaze, args.device, args.extension)
    gaze_estimation.load_model()
    gaze_load_time = time.time() - gaze_load_start
    gaze_estimation.check_model()
    log.info('Gaze Estimation object initialized ')

    log.info('All models loaded')

    # TODO: Create/Overwrite perf_count files
    if pflag is True:
        with open('./model_perf/face.txt', 'w') as fh:
            fh.write('Layer wise benchmarking for `EXECUTED` layers of Face Detection Model: \n\n')
        with open('./model_perf/landmark.txt', 'w') as fh:
            fh.write('Layer wise benchmarking for `EXECUTED` layers of Landmark Detection Model:\n\n')
        with open('./model_perf/head_pose.txt', 'w') as fh:
            fh.write('Layer wise benchmarking for `EXECUTED` layers of Head Pose Estimation Model:\n\n')
        with open('./model_perf/gaze.txt', 'w') as fh:
            fh.write('Layer wise benchmarking for `EXECUTED` layers of Gaze Estimation Model:\n\n')

    # TODO: Initialize InputFeeder object
    feed = InputFeeder(input_type= args.input_type, input_file= args.input)
    log.info('InputFeeder object initialized.')

    # TODO: load the data from source
    log.info('Loading data...')
    feed.load_data()
    log.info('Data Loaded. Beginning inference...')
    for batch, flag in feed.next_batch():
        load_times = (face_load_time, landmark_load_time, head_pose_load_time, gaze_load_time)
        process_times = np.array([tfproc, tlproc, thproc, tgproc])/count
        infer_times = np.array([tfpred, tlpred, thpred, tgpred])/count

        if batch is None:
            break

        # TODO: Exit app if 'X' is pressed
        key = cv2.waitKey(1000)
        if key == ord('x'):
            log.warning('KeyboardInterrupt: `X` was pressed')
            log_stats(load_times, process_times, infer_times)
            sys.exit()

        # TODO: send frame to face detection model
        fprocs = time.time()
        face_ip = face.preprocess_input(batch) # 1920x1080x3 -> 1x3x384x672
        fproc = time.time() - fprocs
        tfproc += fproc

        fpreds = time.time()
        face_op = face.predict(face_ip, pflag)
        fpred = time.time() - fpreds
        tfpred += fpred
        batch, face_box, face_center = face.preprocess_output(batch, face_op, vflag)
        cv2.imshow('Computer Pointer Window', batch)


        # TODO: send face results to landmark
        lprocs = time.time()
        landmark_ip = landmark.preprocess_input(face_box) # 367x246x3 -> 1x3x48x48, but I need to broadcast eye coords to 1920x1080
        lproc = time.time() - lprocs
        tlproc += lproc

        lpreds = time.time()
        landmark_op = landmark.predict(landmark_ip, pflag)
        lpred = time.time() - lpreds
        tlpred += lpred

        face_box_window, lEye, rEye = landmark.preprocess_output(face_box, landmark_op)
        if vflag is True:
            cv2.imshow('Visualized Output', face_box_window)


        # TODO: send face results to head_pose
        hprocs = time.time()
        head_pose_ip = head_pose.preprocess_input(face_box) # 367x246x3 -> 1x3x60x60
        hproc = time.time() - hprocs
        thproc += hproc

        hpreds = time.time()
        head_pose_op = head_pose.predict(head_pose_ip, pflag)
        hpred = time.time() - hpreds
        thpred += hpred

        batch, axes_op = head_pose.preprocess_output(batch, head_pose_op, face_center)
        # TODO: Visualize output if vflag
        if vflag is True:
            cv2.imshow('Computer Pointer Window', batch) # Visualized Output

        # TODO: send landmark and head_pose results to gaze
        gprocs = time.time()
        gaze_lEye = gaze_estimation.preprocess_input(lEye)
        gaze_rEye = gaze_estimation.preprocess_input(rEye)
        gproc = time.time() - gprocs
        tgproc += gproc

        gpreds = time.time()
        gaze_op = gaze_estimation.predict(axes_op, gaze_lEye, gaze_rEye, pflag) # !!!
        gpred = time.time() - gpreds
        tgpred += gpred

        x, y, z = gaze_estimation.preprocess_output(gaze_op)

        # TODO: send gaze results to mouse_controller
        mc = MouseController('high', 'fast')
        mc.move(x, y)

        count += 1

    # TODO: Close feed
    feed.close()
    log.info('Feed closed')
    log_stats(load_times, process_times, infer_times)

# TODO: Logging APP timing STATS
def log_stats(load_times, process_times, infer_times):
    log.info('----------END OF PROGRAM----------')
    log.info('\n \n')
    log.info('----------APP STATS----------')
    log.info('Face Detection Model Loading Time: {0}'.format(load_times[0]))
    log.info('Landmark Detection Model Loading Time: {0}'.format(load_times[1]))
    log.info('Head Pose Estimation Model Loading Time: {0}'.format(load_times[2]))
    log.info('Gaze Estimation Model Loading Time: {0}'.format(load_times[3]))
    log.info('\n')

    log.info('Face Detection Model Processing Time: {0}'.format(process_times[0]))
    log.info('Landmark Detection Model Processing Time: {0}'.format(process_times[1]))
    log.info('Head Pose Estimation Model Processing Time: {0}'.format(process_times[2]))
    log.info('Gaze Estimation Model Processing Time: {0}'.format(process_times[3]))
    log.info('\n')

    log.info('Face Detection Model Inference Time: {0}'.format(infer_times[0]))
    log.info('Landmark Detection Model Inference Time: {0}'.format(infer_times[1]))
    log.info('Head Pose Estimation Model Inference Time: {0}'.format(infer_times[2]))
    log.info('Gaze Estimation Model Inference Time: {0}'.format(infer_times[3]))





def main():
    """
    Parse the arguments
    """
    # TODO: Grab command line args
    args = build_argparser()

    # TODO: Prepare log dump
    log.basicConfig(filename='CPC.log',level=log.INFO)

    # TODO: Call driver
    flow(args)



# TODO: MAIN
if __name__ == '__main__':
    main()
