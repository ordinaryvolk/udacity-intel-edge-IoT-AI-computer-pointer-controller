import cv2
from model import Models


class FaceDetectionModel(Models):
    '''
    Face detsction class
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        
        Models.__init__(self, model_name, device, extensions)
        self.model_type = "FACE"

    def preprocess_output(self, frame, output, threshold, inference_time):
        '''
        Create output for face detection

        input: frame - original frame 
               output - inference output
               threshold - confidence threshold

        output: cropped image, detected face coordinates
        '''

        # Acquire face coordinates
        face_coords = []
        output_boxes = output[self.output_blob][0][0]
        for box in output_boxes:
            if box[2] > threshold:
                xmin = int(box[3] * self.image_width)
                ymin = int(box[4] * self.image_height)
                xmax = int(box[5] * self.image_width)
                ymax = int(box[6] * self.image_height)
                face_coords.append([xmin, ymin, xmax, ymax])

        # Crop the image based on the detected coordinates
        if len(face_coords) == 0: 
            print("{ERROR]: No face detected in frame!")
            exit()

        cropped_frame = frame[ymin:ymax, xmin:xmax] 

        return cropped_frame, face_coords, inference_time 

 
   
