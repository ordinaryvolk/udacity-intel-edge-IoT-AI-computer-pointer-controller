import cv2
import os
import time
from model import Models


class GazeEstimationModel(Models):
    '''
    Head pose class
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        
        Models.__init__(self, model_name, device, extensions)
        self.model_type = "GAZE"

    def preprocess_input(self, frame):
        '''
        Preprocess the raw input frame to prepare for inferencing
        '''

        fixed_width  = 60
        fixed_height = 60
        preprossed_frame = cv2.resize(frame, (fixed_width, fixed_height))
        preprossed_frame = preprossed_frame.transpose((2,0,1))
        b  = 1
        c  = preprossed_frame.shape[0]
        h  = fixed_height 
        w  = fixed_width 
        preprossed_frame   = preprossed_frame.reshape(b, c, h, w)

        return preprossed_frame

    def preprocess_output(self, output, inference_time):
        '''
        Create output for face detection

        input: 
               output - gaze inference output
               inference_time - time spent on inferencing 

        output: gaze estimate results
        '''

        x = output[self.output_blob][0][0]
        y = output[self.output_blob][0][1]
        z = output[self.output_blob][0][2]

        return [x, y, z], inference_time


    def predict(self, left_eye, right_eye, head_pose_angles):
       '''
       Gaze estimation function
       '''
       
       left_eye_preprocessed = self.preprocess_input(left_eye)
       right_eye_preprocessed = self.preprocess_input(right_eye)

       input_network = {"left_eye_image":left_eye_preprocessed,
                        "right_eye_image": right_eye_preprocessed,
                        "head_pose_angles": head_pose_angles
                       }

       start_time = time.time()

       outputs = self.net_plugin.infer(input_network) 

       inference_time = (time.time() - start_time) * 1000

       return self.preprocess_output(outputs, inference_time) 

 
