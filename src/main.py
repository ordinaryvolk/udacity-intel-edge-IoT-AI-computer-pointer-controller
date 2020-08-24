"""Computer Pointer Controller"""


import os
import sys
import time
import socket
import json
import cv2
import traceback
import logging as log
from argparse import ArgumentParser

# Local class imports
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import FaceDetectionModel
from head_pose import HeadPoseModel
from landmarks import LandmarksModel
from gaze_estimation import GazeEstimationModel


LOGLEVEL = log.INFO 

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-fdm", "--facedetectionmodel", required=True, type=str,
                        help="Face detection model.")
    parser.add_argument("-hpm", "--headposemodel", required=True, type=str,
                        help="Head pose model.")
    parser.add_argument("-flm", "--facelandmarksnmodel", required=True, type=str,
                        help="Face landmarks model.")
    parser.add_argument("-gem", "--gazeestimationmodel", required=True, type=str,
                        help="Gaze estimation model.")
    parser.add_argument("-v", "--visualize", required=False, type=str,
                        help="Use FHLG to select which inference results to visualize")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file. Use CAM for camera input")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def main():
    """
    """

    # Grab command line args
    args = build_argparser().parse_args()
    
    input_src             = args.input
    device                = args.device
    extension             = args.cpu_extension
    prob_threshold        = args.prob_threshold

    face_detection_model  = args.facedetectionmodel
    head_pose_model       = args.headposemodel
    landmarks_model       = args.facelandmarksnmodel
    gaze_estimation_model = args.gazeestimationmodel    

    # Create log object set for console output and set log level
    log_obj = log.getLogger()
    log_obj.setLevel(LOGLEVEL)

    console_handler = log.StreamHandler()
    console_handler.setLevel(LOGLEVEL)
    log_obj.addHandler(console_handler)

    # Create detection objects
    face_detection_obj    = FaceDetectionModel(face_detection_model, device, extension)
    head_pose_obj         = HeadPoseModel(head_pose_model, device, extension)
    landmarks_obj         = LandmarksModel(landmarks_model, device, extension)
    gaze_estimation_obj   = GazeEstimationModel(gaze_estimation_model, device, extension)

    # Create mouse controller object
    mouse_controller = MouseController('medium', 'fast')
    # Place mouse at the center of the screen
    mouse_controller.init_position()
    log_obj.info("[Info]: Place mouse at the center of the screen")

    # Place holder for total inferencing time
    total_inference_time = 0

    # Load models and get the model loading times
    start_time = time.time()
    face_detection_obj.load_model()
    end_time = time.time()
    face_detection_loading_time = end_time - start_time

    start_time = time.time()
    head_pose_obj.load_model()
    end_time = time.time()
    head_pose_loading_time = end_time - start_time

    start_time = time.time()
    landmarks_obj.load_model()
    end_time = time.time()
    landmarks_detection_loading_time = end_time - start_time

    start_time = time.time()
    gaze_estimation_obj.load_model()
    end_time = time.time()
    gaze_estimation_loading_time = end_time - start_time

    # Configure input video source
    if input_src.lower() == 'cam':
        input_channel = InputFeeder(input_type='cam')
    elif not os.path.exists(input_src):
        log.error("Video file not found! Exiting....")
        exit(1)
    else:
        input_channel = InputFeeder(input_type='video', input_file=input_src)
        log_obj.info("[Info]: Opening video file ...")


    input_channel.load_data()
    video_width = int(input_channel.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(input_channel.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_channel.cap.get(cv2.CAP_PROP_FPS))
    
    frame_counter = 0
    total_face_inf_time     = 0
    total_head_inf_time     = 0
    total_lanmarks_inf_time = 0
    total_gaze_inf_time     = 0
    frame_processing_time   = 0

    # Process each frame
    try:
        for frame in input_channel.next_batch():
            frame_processing_start_time = time.time()

            frame_counter = frame_counter + 1
            key = cv2.waitKey(60)

            # Use face detection to find cropped face and provide face coordinates
            cropped_face, face_coords, face_inference_time = face_detection_obj.predict(frame, prob_threshold )
            total_face_inf_time = total_face_inf_time + face_inference_time

            #  Now use cropped face for head pose detection
            head_pose_estimate, head_inference_time = head_pose_obj.predict(cropped_face, prob_threshold ) 
            total_head_inf_time = total_head_inf_time + head_inference_time
 
            #  Now use cropped face for landmarks detection
            cropped_left_eye, cropped_right_eye, eyes_coords, converted_landmarks, landmarks_inference_time = landmarks_obj.predict(cropped_face, prob_threshold )
            total_lanmarks_inf_time = total_lanmarks_inf_time + landmarks_inference_time
 
            #  Finally gaze estimation
            gaze_vector, gaze_estimate_time = gaze_estimation_obj.predict(cropped_left_eye, cropped_right_eye, head_pose_estimate) 
            total_gaze_inf_time = total_gaze_inf_time + gaze_estimate_time

            # Move the mouse
            #mouse_controller.move(gaze_vector[0], gaze_vector[1])
 
            # Show size-reduced frame for visual comparison

            # Check potential visualize flags: 'F', 'H', 'L', 'G' 
            # If flag exist, process image to show inference results
            if args.visualize is not None:

                visualize_flag = str(args.visualize)

                # Draw bounding box around detected face
                if 'F' in visualize_flag:
                    cv2.rectangle(frame, (face_coords[0][0], face_coords[0][1]), (face_coords[0][2], face_coords[0][3]), (0,255,0), 2)
                
                # Show head pose parameters
                if 'H' in visualize_flag:
                    cv2.putText(frame, "Head pose: yaw: {:.3f}, pitch: {:.3f}, roll: {:.3f}".format(head_pose_estimate[0], head_pose_estimate[1], head_pose_estimate[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 5) 

                # Draw dots on detected facial landmarks
                if 'L' in visualize_flag:
                    cv2.circle(frame, (converted_landmarks[0] + face_coords[0][0], converted_landmarks[1] + face_coords[0][1]), 10, (0,255,0), 5) 
                    cv2.circle(frame, (converted_landmarks[2] + face_coords[0][0], converted_landmarks[3] + face_coords[0][1]), 10, (0,255,0), 5)
                    cv2.circle(frame, (converted_landmarks[4] + face_coords[0][0], converted_landmarks[5] + face_coords[0][1]), 10, (0,255,0), 5)
                    cv2.circle(frame, (converted_landmarks[6] + face_coords[0][0], converted_landmarks[7] + face_coords[0][1]), 10, (0,255,0), 5)
                    cv2.circle(frame, (converted_landmarks[8] + face_coords[0][0], converted_landmarks[9] + face_coords[0][1]), 10, (0,255,0), 5)

                # Display gaze parameters
                if 'G' in visualize_flag:
                    cv2.putText(frame, "Gaze estimate: x: {:.3f}, y: {:.3f}, z: {:.3f}".format(gaze_vector[0], gaze_vector[1], gaze_vector[2]), (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 5) 

            resized_frame = cv2.resize(frame, (640, 360))
            cv2.imshow('frame', resized_frame)

            if frame_counter % 4 == 0:
                mouse_controller.move(gaze_vector[0], gaze_vector[1])

            frame_processing_time = frame_processing_time + (time.time() - frame_processing_start_time) * 1000

            if key == 27:
                break

    except Exception as e:
        #traceback.print_exc()
        if 'shape' in str(e):
            log_obj.info("Video feed finished")
        else:
            log_obj.error("[ERROR]: " + str(e)) 
        pass

    # All done, cleaning up
    cv2.destroyAllWindows()
    input_channel.close()

    # Print out statistics
    log_obj.info("[Info]: Video source FPS: " + str(fps))
    log_obj.info("[Info]: Total frame count: " + str(frame_counter))
    log_obj.info("")
    log_obj.info("[Info]: Face detection model loading time: {:.3f} ms".format(face_detection_loading_time*1000))
    log_obj.info("[Info]: Head pose model loading time: {:.3f} ms".format(head_pose_loading_time*1000))
    log_obj.info("[Info]: Facial landmarks detection model loading time: {:.3f} ms".format(landmarks_detection_loading_time*1000))
    log_obj.info("[Info]: Gaze estimation model loading time: {:.3f} ms".format(gaze_estimation_loading_time*1000))
    log_obj.info("")
    log_obj.info("[Info]: Average  per frame total processing time : {:.3f} ms".format(frame_processing_time/frame_counter))
    log_obj.info("[Info]: Average face inferencing  time: {:.3f} ms".format(total_face_inf_time/frame_counter)) 
    log_obj.info("[Info]: Average head pose  inferencing  time: {:.3f} ms".format(total_head_inf_time/frame_counter))
    log_obj.info("[Info]: Average facial landmarks inferencing  time: {:.3f} ms".format(total_lanmarks_inf_time/frame_counter))
    log_obj.info("[Info]: Average gaze estimate  time: {:.3f} ms".format(total_gaze_inf_time/frame_counter))


if __name__ == '__main__':
    main()
