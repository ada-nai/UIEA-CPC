'''
This class can be used to feed input from an image, webcam, or video to your model.
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
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file

    def load_data(self):
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
            # self.cap.open(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
            # self.cap.open(0)
        else:
            self.cap=cv2.imread(self.input_file)

        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        # log.info('Original Frame dimeensions (width, height): ',self.frame_width, self.frame_height)

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        while True:
            for flag in range(10):
                flag, frame=self.cap.read()
                if not flag:
                    log.info('Stream ended or error')
                    sys.exit()
                # print('next batch')
                # print(frame)
            yield frame


    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()
    
