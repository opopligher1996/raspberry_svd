import random
import cv2

mid_line = 550
standby_area_left = 500
standby_area_right = 600
     
class TrackableTarget:
 def __init__(self, bbox, score, label, imSize, frame):
     # store the object ID, then initialize a list of centroids
     self.id = random.randint(1, 1001)
     imW = imSize[0]
     imH = imSize[1]
     ymin = int(max(1,(bbox[0] * imH)))
     xmin = int(max(1,(bbox[1] * imW)))
     ymax = int(min(imH,(bbox[2] * imH)))
     xmax = int(min(imW,(bbox[3] * imW)))
     self.bbox = (xmin,ymin,xmax,ymax)
     self.center_point = (int((xmin+xmax)/2) , int((ymin+ymax)/2))
     self.score = score
     self.count = 3
     self.label = label
     self.path = None
     self.image = frame[ymin:ymax, xmin:xmax]
     self.isSelected = False
     self.isMissing = False
     self.setInitStatus()
     self.tracker = None

 def setInitStatus(self):
     center_point = self.center_point
     x = center_point[0]
     
     if(x < mid_line):
         if( x > standby_area_left):
             self.initStatus = 'standByLeft'
             self.status = 'standByLeft'
         else:
             self.initStatus = 'left'
             self.status = 'left'
     elif(x >= mid_line):
         if( x < standby_area_right):
             self.initStatus = 'standByRight'
             self.status = 'standByRight'
         else:
             self.initStatus = 'right'
             self.status = 'right'

 def setIsSelected(self, isSelected):
     self.isSelected = isSelected
     
 def update(self, trackableTarget, frame):
     self.count = 3
     self.bbox = trackableTarget.getBBox()
     self.center_point = trackableTarget.getCenterPoint()
     (xmin,ymin,xmax,ymax) = self.bbox
     center_point = self.center_point
     x = center_point[0]
     if(x < mid_line):
         if( x > standby_area_left):
             self.status = 'standByLeft'
         else:
             self.status = 'left'
     elif(x >= mid_line):
         if( x < standby_area_right):
             self.status = 'standByRight'
         else:
             self.status = 'right'
 
 def updateFrame(self, frame):
     self.frame = frame
     
 def isClosed():
     print('isClosed')

 def getIsSelected(self):
     return self.isSelected
     
 def getId(self):
     return self.id
    
 def getCenterPoint(self):
     return self.center_point

 def getBBox(self):
     return self.bbox

 def getCount(self):
     return self.count
    
 def getLabel(self):
     return self.label
    
 def getScore(self):
     return self.score
 
 def getFrame(self):
     return self.frame
    
 def getImage(self):
     return self.image
    
 def getInitStatus(self):
     return self.initStatus
 
 def getStatus(self):
     return self.status
 
 def getCount(self):
     return self.count
 
 def getTracker(self):
     return self.tracker
     
 def setTracker(self, tracker):
     self.tracker = tracker
     
 def countDown(self):
     self.count = self.count - 1
     return self.count

 def checkStatus(self):
     if(self.initStatus == 'standByLeft' or self.initStatus == 'left'):
         if(self.status == 'right'):
             return 'out'
     else:
         if(self.status == 'left'):
             return 'in'
     return 'running'