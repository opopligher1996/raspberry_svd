######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/2/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video.
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
import importlib.util
import tflite_runtime.interpreter as tflite
import platform
import requests
import json
import base64
import datetime
from threading import Thread
from TrackableTarget import *
from RequestWorker import *

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
    file_path = '/home/pi/workspace/svd/raspberry_svd/tmp/'+filename
    cv2.imwrite(file_path, frame)
    return file_path

def getMse(imageA, imageB):
    print(type(imageA))
    print(type(imageB))
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.7)
parser.add_argument('--video', help='Name of the video file',
                    default='test.mp4')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
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

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

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

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = 800
imH = 600

##focus area
focus_area = (449, 95, 204, 307)
mid_line = (551, 0, 551, 600)
standby_area_left = (347, 95, 102, 307)
standby_area_right = (653, 95, 102, 307)
needCapture = False
captureCount = 0
targets = []

while(video.isOpened()):
#    try:
        size = sum(d.stat().st_size for d in os.scandir('/home/pi/workspace/svd/raspberry_svd/tmp') if d.is_file())
        # Acquire frame and resize to expected shape [1xHxWx3]
        if( size > 8589934592):
            break
        
        ret, frame = video.read()
#        r = cv2.selectROI(frame)
#        print(r)
        if ret:
           #frame = cv2.flip(frame, 0)
           fps = video.get(cv2.CAP_PROP_FPS)
           frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           display_frame = frame.copy()
           display_frame = cv2.resize(display_frame, (imW, imH))
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
        num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
       
        print('init tmp')
        tmp = []
        for i in range(len(scores)):
            if((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                target = TrackableTarget(boxes[i], scores, labels[int(classes[i])], (imW, imH), display_frame)
                tmp.append(target)
                cv2.imshow('Target Image', target.getImage())
                if(point_in_area(target.getCenterPoint(), focus_area)):
                    needCapture = True

        if(tmp != []): 
            targets = tmp
            for target in targets:
                (xmin,ymin,xmax,ymax) = target.getBBox()
                cv2.rectangle(display_frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
                center_point = (int((xmin+xmax)/2), int((ymin+ymax)/2))
                cv2.circle(display_frame, center_point, 1, (10,255,0), 5)
                object_name = target.getLabel()
                score = target.getScore()
                label = '%s' % (object_name)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)

#        if(needCapture == True):
#            captureCount = captureCount + 1
#            if((captureCount % 20) == 0):
#                saveImage(frame)
#            if(captureCount == 100):
#                needCapture = False
       
        cv2.line(display_frame, (mid_line[0], mid_line[1]), (mid_line[2], mid_line[3]), (0, 0, 255), 4)
        cv2.rectangle(display_frame, (standby_area_left[0], standby_area_left[1]), (standby_area_left[0]+standby_area_left[2],standby_area_left[1]+standby_area_left[3]), (255, 0, 0), 4)
        cv2.rectangle(display_frame, (standby_area_right[0], standby_area_right[1]), (standby_area_right[0]+standby_area_right[2],standby_area_right[1]+standby_area_right[3]), (255, 0, 0), 4)
        cv2.rectangle(display_frame, (focus_area[0], focus_area[1]), (focus_area[0]+focus_area[2],focus_area[1]+focus_area[3]), (0, 0, 255), 4)
        cv2.imshow('Object detector', display_frame)
        # Press 'q' to quit
        if cv2.waitKey(0) == ord('q'):
            break
#    except:
#        print('except')
#        break

# Clean up
video.release()
cv2.destroyAllWindows()

