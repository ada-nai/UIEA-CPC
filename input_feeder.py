'''
This class is used to feed input from a webcam or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2
import sys
import time
import logging as log
from numpy import ndarray

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input video file. Leave empty for cam input_type.
        '''
        self.input_type=input_type
        self.input_file = input_file
        try:
            if self.input_type=='video':
                assert self.input_file is not None
                self.input_file=input_file
        except:
            print('Error occurred, refer `CPC.log` file for details')
            log.warning('Provide input file using `-i` argument. Use --help for more info')

    def load_data(self):
        try:
            if self.input_type=='video':
                self.cap=cv2.VideoCapture(self.input_file)
            elif self.input_type=='cam':
                self.cap=cv2.VideoCapture(0)
            else:
                self.cap=cv2.imread(self.input_file)
        except Exception as e:
            print('Error occurred, refer `CPC.log` file for details')
            log.error('Error loading input media')

        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        log.info('Original Frame dimensions (width, height):{0}x{1} '.format( str(self.frame_width),  str(self.frame_height) ) )

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        while True:
            for flag in range(10):
                flag, frame=self.cap.read()
                if not flag:
                    log.info('Stream ended.')
            yield frame, flag


    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()
