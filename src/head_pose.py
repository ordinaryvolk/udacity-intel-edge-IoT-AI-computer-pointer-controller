import cv2
import os
import numpy as np

from model import Models


class HeadPoseModel(Models):
    '''
    Head pose class
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        
        Models.__init__(self, model_name, device, extensions)
        self.model_type = "HEAD"

    def preprocess_output(self, frame, output, threshold, inference_time):
        '''
        Create output for face detection

        input: frame - original frame
               output - inference output
               threshold - confidence threshold

        output: head pose estimate results 
        '''

        yaw   = output["angle_y_fc"][0][0]
        pitch = output["angle_p_fc"][0][0]
        roll  = output["angle_r_fc"][0][0]

        return [yaw, pitch, roll], inference_time
   
