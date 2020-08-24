# Computer Pointer Controller

This is the final project of Udacity Intel Edge AI nanodegree program. In this project we use multiple Intel OpenVino pre-trained models to constuct a pipeline. Video will then be passed through this pipeline to infer the gaze stance and use it to control mouse movement on screen. 

## Project Set Up and Installation
System setup:
  - Hardware: Intel(R) Core(TM) i7-8809G CPU @ 3.10GHz with 16GB RAM
  - OS: Ubuntu 18.04.5 LTS
  - Python 3.6.9
  - OpenVino R2020.4.287

Directory structure:
.
├── bin
├── models
│   └── intel
│       ├── face-detection-adas-binary-0001
│       │   └── FP32-INT1
│       ├── gaze-estimation-adas-0002
│       │   ├── FP16
│       │   ├── FP16-INT8
│       │   └── FP32
│       ├── head-pose-estimation-adas-0001
│       │   ├── FP16
│       │   ├── FP16-INT8
│       │   └── FP32
│       └── landmarks-regression-retail-0009
│           ├── FP16
│           ├── FP16-INT8
│           └── FP32
├── src
└── venv

Directory explanation:
  - bin: stores the video file used for this project
  - models: stores the downloaded models used for inferencing
  - src: contains the source code files
  - venv: virtual environment files

Models used. To buil dpipeline for this project we use the following models:
  - face-detection-adas-binary-000
  - gaze-estimation-adas-0002
  - head-pose-estimation-adas-0001
  - landmarks-regression-retail-0009

Below are setup steps:
0.1. Create virtual environment
> pip install virtualenv
> virtualenv venv
> source ./venv/bin/activate


0.2. Install the rest of the depedencies: go back to starter coder directory and run following:
> pip3 install -r requirements.txt
> pip3 install pyautogui
> sudo apt-get install python3-tk python3-dev

1. Download models. Go to starter code root directory, then create the "models" directory and run the following commands:
> cd models
> python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
> python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
> python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
> python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"

## Demo
To run the demo, follow the sequence below
1. Go to root directory of the folder
2. Run the following commands: 
> cd src
> python3 main.py -i ../bin/demo.mp4  -fdm ../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -hpm ../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -flm ../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -gem ../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -l NONE -d CPU -pt 0.6

To use camera:
> python3 main.py -i CAM  -fdm ../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -hpm ../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -flm ../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -gem ../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -l NONE -d CPU -pt 0.6

To visualize the inference results: use the following command:
> >python3 main.py -i CAM  -fdm ../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -hpm ../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -flm ../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -gem ../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -l NONE -d CPU -pt 0.6 -v FHLG 

## Documentation
usage: main.py [-h] -fdm FACEDETECTIONMODEL -hpm HEADPOSEMODEL -flm
               FACELANDMARKSNMODEL -gem GAZEESTIMATIONMODEL [-v VISUALIZE] -i
               INPUT [-l CPU_EXTENSION] [-d DEVICE] [-pt PROB_THRESHOLD]
Detailed explanations for the parameters:
  -fdm: path to face detection model
  -hpm: path to head pose estimation model
  -flm: path to face landmarks model
  -gem: path to gaze estimate model
  -v:   (optional) Flag to show inferencing results:
        'F': show bounding box from face detection
        'H': show results  from head pose estimation
        'L': show facial landmarks
        'G': show results of gaze estimate
  -i:   path to video input source. 
        If use a video file, put path to the file here
        If use a usb camera, use 'CAM' as parameteri
  -l:   CPU extension
  -d:   Device used for inferemcing. For example, CPU or MYRIAD
  -pt:  Probability threshold for face detection

## Benchmarks
Bench marking results for different models are show below:
							INT8		FP16		FP32
Face detection model loading time: 			-		-		143.827 ms	
Head pose model loading time: 				219.214 ms	64.742 ms	160.146 ms	
Facial landmarks detection model loading time: 		71.423 ms	53.378 ms	76.742 ms
Gaze estimation model loading time: 			144.235 ms	75.864 ms	69.355 ms

Average  per frame total processing time : 		79.824 ms	83.386 ms	82.116 ms
Average face inferencing  time: 			12.196 ms	13.749 ms	13.427 ms
Average head pose  inferencing  time: 			1.266 ms	1.591 ms	1.577 ms
Average facial landmarks inferencing  time: 		0.603 ms	0.635 ms	0.663 ms
Average gaze estimate  time: 				1.322 ms	1.767 ms	1.707 ms

## Results
As can be seen from the inferencing results, the model inferencing time generally decreases with lower precisions, but the improvement doesn't seem to have major differences. INT8 models appear to have the highest loading time. It is perhaps tolerable when the models is loaded once and continuously used. If the somehow the model needs be loaded constantly, then this disadvantage will out-weight the gains in inferencing time, as the loading time is dispropotionally larger than inferencing time.   

## Standout

This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.
Didn't really use the async call so cannot comment on this

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

One issue I ran into was that at the beginning the location of the mouse is not determined. If the mouse happens to be close to the edge of the screen then it can either got stuck without showing the intended movement, or even worse, get to the screen corner and cause pyautogui to generate an exception. To fix this, I enhanced to mouse controller class to add an initial function to place the mouse at (1/4* width, 1/4 * height). That way the mouse would have enough room to manuver.

Also at the beginning the program updates the mouse movement for every frame. And I found out the mouse moves too far to quickly even when the gaze is not changed across couple of frames. I then update the program to only update mouse movement every 4 frames. With that solution the mouse moves much reasonably. 


