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
import math
import dlib
import request_utils
from threading import Thread
from TrackableTarget import *

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

def sendData(enter_count, exit_count):
    print('send')
    
    
def saveImage(frame):
    filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.jpg")
    file_path = '/home/pi/workspace/svd/raspberry_svd/tmp/'+filename
    cv2.imwrite(file_path, frame)
    return file_path

def distance_between_points(pointA, pointB):
    d = math.sqrt((pointA[0]-pointB[0])*(pointA[0]-pointB[0]) + (pointA[1] - pointB[1])*(pointA[1] - pointB[1]))
    return d

def cal_iou(bbox1, bbox2):
#    (t_xmin,t_ymin,t_xmax,t_ymax)
    (xmin1, ymin1, xmax1, ymax1) = bbox1
    (xmin2, ymin2, xmax2, ymax2) = bbox2
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    xA = max(xmin1, xmin2)
    yA = max(ymin1, ymin2)
    xB = min(xmax1, xmax2)
    yB = min(ymax1, ymax2)
    interArea = (xB - xA) * (yB - yA)
    iou = interArea / (area1 + area2 - interArea)
    return iou
    
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
#focus_area = (449, 95, 204, 307)
mid_line = (550, 0, 550, 600)
standby_area_left = (500, 45, 50, 357)
standby_area_right = (550, 45, 50, 357)
needCapture = False
captureCount = 0
inCount = 0
exitCount = 0
targets = []
count = 0

lastDetectTime = None

request_utils.uploadHeartBeat()
while(video.isOpened()):
#    try:
        time = datetime.datetime.now().strftime("%H:%M:%S")
        print(time + ' handle count = '+ str(count))
        count = count + 1
    
        size = sum(d.stat().st_size for d in os.scandir('/home/pi/workspace/svd/raspberry_svd/tmp') if d.is_file())
        # Acquire frame and resize to expected shape [1xHxWx3]
        if( size > 8589934592):
            break
        
        ret, frame = video.read()
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
        
        tmp = []
        for i in range(len(scores)):
            if((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                t = TrackableTarget(boxes[i], scores, labels[int(classes[i])], (imW, imH), display_frame)
                tmp.append(t)
#                if(point_in_area(target.getCenterPoint(), focus_area)):
#                    needCapture = True
        updatedTargets = []
        
        for target in targets:
            min_iou = 0.4
            selected_tmp_index = None
            for i, t in enumerate(tmp):
                iou = cal_iou(target.getBBox(), t.getBBox())
                if(iou > min_iou):
                    selected_tmp_index = i
                    min_iou = iou
                    (xmin,ymin,xmax,ymax) = t.getBBox()
                    cv2.rectangle(display_frame, (xmin,ymin), (xmax,ymax), (255, 255, 255), 4)
            
            if(selected_tmp_index != None):
                tmp[selected_tmp_index].setIsSelected(True)
                target.update(tmp[selected_tmp_index], display_frame )
                inOrOut = target.checkStatus()
                if(inOrOut == "running"):
                    updatedTargets.append(target)
                elif(inOrOut == "in"):
                    inCount = inCount + 1
                else:
                    exitCount = exitCount + 1
            else:
                countDown = target.countDown()
                if(countDown > 0):
                    updatedTargets.append(target)
        
        for t in tmp:
            if(t.getIsSelected() == False):
                updatedTargets.append(t)

        
        targets = updatedTargets.copy()
        trackers = []
        for target in targets:
            target_init_status = str(target.getInitStatus())
            target_status = str(target.getStatus())
            target_count = str(target.getCount())
            target_id = str(target.getId()) + ' count: ' + target_count
            target_status = 'init_status: ' + target_init_status + ' status: '+ target_status
            (xmin,ymin,xmax,ymax) = target.getBBox()
            cv2.rectangle(display_frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
            center_point = (int((xmin+xmax)/2), int((ymin+ymax)/2))
            cv2.circle(display_frame, center_point, 1, (10,255,0), 5)
            cv2.putText(display_frame, target_id, (xmin+10, ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, target_status, (xmin+10, ymin+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(display_frame, 'In = '+str(inCount), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, 'Out = '+str(exitCount), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.line(display_frame, (mid_line[0], mid_line[1]), (mid_line[2], mid_line[3]), (0, 0, 255), 4)
        cv2.rectangle(display_frame, (standby_area_left[0], standby_area_left[1]), (standby_area_left[0]+standby_area_left[2],standby_area_left[1]+standby_area_left[3]), (255, 0, 0), 4)
        cv2.rectangle(display_frame, (standby_area_right[0], standby_area_right[1]), (standby_area_right[0]+standby_area_right[2],standby_area_right[1]+standby_area_right[3]), (255, 0, 0), 4)
#        cv2.rectangle(display_frame, (focus_area[0], focus_area[1]), (focus_area[0]+focus_area[2],focus_area[1]+focus_area[3]), (0, 0, 255), 4)
        cv2.imshow('Object detector', display_frame)
        # Press 'q' to quit
        if count > 9999:
            if cv2.waitKey(0) == ord('q'):
                break
        else:
            if cv2.waitKey(1) == ord('q'):
                break
#    except:
#        print('except')
#        break

# Clean up
video.release()
cv2.destroyAllWindows()

