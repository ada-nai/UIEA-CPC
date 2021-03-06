'''
This class controls the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
'''
import logging as log
import pyautogui

class MouseController:
    def __init__(self, precision, speed):
        precision_dict={'high':1000, 'low':100, 'medium':500}
        speed_dict={'fast':1, 'slow':10, 'medium':5}

        self.precision=precision_dict[precision]
        self.speed=speed_dict[speed]

    def move(self, x, y):
        try:
            pyautogui.move(1*x*self.precision, -1*y*self.precision, duration=self.speed)
        except Exception as e:
            print('Error occurred, refer `CPC.log` file for details')
            log.error('pyautogui error: ', e)
            pass
