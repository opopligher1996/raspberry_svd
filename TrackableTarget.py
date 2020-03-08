import random

class TrackableTarget:
 def __init__(self, bbox, score, label, imSize):
   # store the object ID, then initialize a list of centroids
   self.id = random.randint(1, 1001)
   self.status = "init"
   imW = imSize[0]
   imH = imSize[1]
   ymin = int(max(1,(bbox[0] * imH)))
   xmin = int(max(1,(bbox[1] * imW)))
   ymax = int(min(imH,(bbox[2] * imH)))
   xmax = int(min(imW,(bbox[3] * imW)))
   self.bbox = ((xmin,ymin), (xmax,ymax))
   self.center_point = (int((xmin+xmax)/2) , int((ymin+ymax)/2))
   self.score = score
   self.count = 10
   self.label = label
   self.path = None
   # self.centroid = centroid
   # self.counted = False

 def update(self, trackableTarget):
     self.bbox = trackableTarget.bbox
     self.center_point = trackableTarget.center_point
     self.label = trackableTarget.label
 
 def updateFrame(self, frame):
     self.frame = frame
     
 def isClosed():
     print('isClosed')

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
     
 def countDown(self):
     self.count = self.count - 1
     return self.count
