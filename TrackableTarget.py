import random
import cv2

class TrackableTarget:
 def __init__(self, bbox, score, label, imSize, frame):
     # store the object ID, then initialize a list of centroids
     self.id = random.randint(1, 1001)
     self.status = "init"
     imW = imSize[0]
     imH = imSize[1]
     ymin = int(max(1,(bbox[0] * imH)))
     xmin = int(max(1,(bbox[1] * imW)))
     ymax = int(min(imH,(bbox[2] * imH)))
     xmax = int(min(imW,(bbox[3] * imW)))
     self.bbox = (xmin,ymin,xmax,ymax)
     self.center_point = (int((xmin+xmax)/2) , int((ymin+ymax)/2))
     self.score = score
     self.count = 5
     self.label = label
     self.path = None
     self.image = frame[ymin:ymax, xmin:xmax]
     self.tracker = cv2.TrackerKCF_create()
     self.tracker.init(frame, (xmin,ymin,xmax-xmin,ymax-ymin))
     self.isSelected = False
     self.isMissing = False
#     self.centroid = centroid
#     self.counted = False

 def setSelect(self):
     self.isSelected = True
     
 def update(self, trackableTarget, frame):
     self.bbox = trackableTarget.getBBox()
     self.center_point = trackableTarget.getCenterPoint()
     self.tracker = cv2.TrackerKCF_create()
     (xmin,ymin,xmax,ymax) = self.bbox
     self.tracker.init(frame, (xmin,ymin,xmax-xmin,ymax-ymin))
 
 def updateFrame(self, frame):
     self.frame = frame
     
 def isClosed():
     print('isClosed')

 def getIsSelected(self):
     return self.isSelected
    
 def getTracker(self, frame):
     ok, bbox = self.tracker.update(frame)
     return bbox
     
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
    
 def countDown(self):
     self.count = self.count - 1
     if(self.count == 0):
         self.isMissing = True
     return self.count
