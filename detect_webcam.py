######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import tflite_runtime.interpreter as tflite
import importlib.util
import platform
import detect
from time import sleep
from TrackableTarget import *
import datetime
import json
import requests
import base64

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def point_in_area(point, area):
    area_xmin = area[0]
    area_ymin = area[1]
    area_xmax = area[0] + area[2]
    area_ymax = area[1] + area[3]
    xpoint = point[0]
    ypoint = point[1]
    if(xpoint > area_xmin and xpoint < area_xmax):
        if(ypoint > area_ymin and ypoint < area_ymax):
            return True
    return False

def saveImage(frame):
    filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.jpg")
    file_path = '/home/pi/workspace/svd/tmp/'+filename
    cv2.imwrite(file_path, frame)
    return file_path

def isMatchedTram(tram):
    file_config = None
    with open('/home/pi/workspace/mtr_demo_api/data.json') as json_file:
        file_config = json.load(json_file)
    print(file_config)
    cam_threshold = file_config['cam_threshold']
    for cam in cam_threshold:
        if (cam['id'] == 'cam1'):
            types = cam['type']
            colors = cam['color']
            tram_type = tram.getType()
            tram_color = tram.getColor()
            for type in types:
                if(type == tram_type):
                    return True
            for color in colors:
                if(color == tram_color):
                    return True
    return False
    
def sendData(tram):
    print('sendData')
    image_path = tram.getImagePath()
    image = base64.encodestring(open(image_path,"rb").read())
    type = tram.getType()
    color = tram.getColor()
    headers = {'Content-Type': 'application/json'}
    id = "cam1"
    payload = {
                "id": id,
                "type": type,
                "color": color,
                "image": str(image, encoding='utf-8')
    }
    print("payload:")
    print(payload)
    response = requests.post('http://192.168.1.1/api/setCamStatus.php', data = json.dumps(payload))
    print(response)
    
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.9)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
PATH_TO_CKPT, *device = PATH_TO_CKPT.split('@')
interpreter = tflite.Interpreter(
      model_path=PATH_TO_CKPT,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)
out = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc('M','J','P','G'),10,(480,640))

imW = 800
imH = 600

##focus area
focus_area = (449, 95, 204, 307)
mid_line = (551, 0, 551, 600)
standby_area_left = (347, 95, 102, 307)
standby_area_right = (653, 95, 102, 307)
needCapture = False
captureCount = 0

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    try:
        size = sum(d.stat().st_size for d in os.scandir('/home/pi/workspace/svd/raspberry_svd/tmp') if d.is_file())
        # Acquire frame and resize to expected shape [1xHxWx3]
        if( size > 8589934592):
            break
        
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()
        frame1 = cv2.flip(frame1, 0)
        frame1_resized = cv2.resize(frame1, (imW, imH))
        
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
            
        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        
        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        
        for i in range(len(scores)):
            if((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                target = TrackableTarget(boxes[i], scores, labels[int(classes[i])], (imW, imH))
                if(point_in_area(target.getCenterPoint(), focus_area)):
                    needCapture = True
        
        if(needCapture == True):
            captureCount = captureCount + 1
            if((captureCount % 3) == 0):
                saveImage(frame)
            if(captureCount == 100):
                needCapture = False
            
#        cv2.line(frame1_resized, (mid_line[0], mid_line[1]), (mid_line[2], mid_line[3]), (0, 0, 255), 4)
#        cv2.rectangle(frame1_resized, (standby_area_left[0], standby_area_left[1]), (standby_area_left[0]+standby_area_left[2],standby_area_left[1]+standby_area_left[3]), (255, 0, 0), 4)
#        cv2.rectangle(frame1_resized, (standby_area_right[0], standby_area_right[1]), (standby_area_right[0]+standby_area_right[2],standby_area_right[1]+standby_area_right[3]), (255, 0, 0), 4)
#        cv2.rectangle(frame1_resized, (focus_area[0], focus_area[1]), (focus_area[0]+focus_area[2],focus_area[1]+focus_area[3]), (0, 0, 255), 4)
#        #Draw framerate in corner of frame
#        cv2.putText(frame1_resized,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        
#        resize_frame = cv2.resize(frame1_resized, (500, 500))
#        # All the results have been drawn on the frame, so it's time to display it.
#        cv2.imshow('Object detector', resize_frame)
        
        # Calculate framerate
        t2 = cv2.getTickCount()
        time = (t2-t1)/freq
        frame_rate_calc= 1/time
        
        #cv2.waitKey(0)
        
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
    except:
        print('except')
# Clean up
out.release()
cv2.destroyAllWindows()
videostream.stop()
