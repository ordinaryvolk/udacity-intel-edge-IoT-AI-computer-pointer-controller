import cv2

from model import Models

EYEOPENING = 20

class LandmarksModel(Models):
    '''
    Head pose class
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        
        Models.__init__(self, model_name, device, extensions)
        self.model_type = "LANDMARKS"

    def preprocess_output(self, frame, output, threshold, inference_time):
        '''
        Create output for face detection

        input: frame - original frame
               output - inference output
               threshold - confidence threshold

        output: face landmarks detection results
        '''

        # Extract eye locations from inference results 
        left_eye_coords_x  = int(output[self.output_blob][0][0][0][0] * self.image_width)
        left_eye_coords_y  = int(output[self.output_blob][0][1][0][0] * self.image_height)
        right_eye_coords_x = int(output[self.output_blob][0][2][0][0] * self.image_width)
        right_eye_coords_y = int(output[self.output_blob][0][3][0][0] * self.image_height)
        nose_coords_x      = int(output[self.output_blob][0][4][0][0] * self.image_width)
        nose_coords_y      = int(output[self.output_blob][0][5][0][0] * self.image_height) 
        mouth_left_x       = int(output[self.output_blob][0][6][0][0] * self.image_width) 
        mouth_left_y       = int(output[self.output_blob][0][7][0][0] * self.image_height)
        mouth_right_x      = int(output[self.output_blob][0][8][0][0] * self.image_width)
        mouth_right_y      = int(output[self.output_blob][0][9][0][0] * self.image_height) 

        converted_output = [left_eye_coords_x, left_eye_coords_y, right_eye_coords_x, right_eye_coords_y, nose_coords_x, nose_coords_y, mouth_left_x, mouth_left_y, mouth_right_x, mouth_right_y]        

        left_eye_coord_x_bounded = max(0, (left_eye_coords_x - EYEOPENING))
        left_eye_coord_y_bounded = max(0, (left_eye_coords_y - EYEOPENING))

        # Crop left and righr eye images
        cropped_left_eye  = frame[(left_eye_coord_y_bounded):(left_eye_coords_y + EYEOPENING), (left_eye_coord_x_bounded):(left_eye_coords_x + EYEOPENING)]
        cropped_right_eye = frame[(right_eye_coords_y - EYEOPENING):(right_eye_coords_y + EYEOPENING), (right_eye_coords_x - EYEOPENING):(right_eye_coords_x + EYEOPENING)] 

        eyes_coords = [
                       [(left_eye_coord_x_bounded, left_eye_coord_y_bounded), (left_eye_coords_x + EYEOPENING, left_eye_coords_y + EYEOPENING)],  
                       [(right_eye_coords_x - EYEOPENING, right_eye_coords_y - EYEOPENING), (right_eye_coords_x + EYEOPENING, right_eye_coords_y + EYEOPENING)]
                      ] 


        return cropped_left_eye, cropped_right_eye, eyes_coords, converted_output, inference_time  
   
