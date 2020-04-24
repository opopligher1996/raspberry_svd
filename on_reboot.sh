#!/bin/bash

cd /home/pi/workspace/svd/raspberry_svd
/usr/bin/python3.7 /home/pi/workspace/svd/raspberry_svd/detect_video.py --modeldir=Sample_TFLite_model --video=videos/project.mp4 --edgetpu