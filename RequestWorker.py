import requests
import calendar
import time
import json
import base64

class RequestWorker():
    def __init__(self):
        print('start thread')
    
    def sendRequest(self, trackableTarget):
        
        headers = {'Content-Type': 'application/json'}
        
        ts = calendar.timegm(time.gmtime())
        bbox = trackableTarget.getBBox()
        score = trackableTarget.getScore()
        label = trackableTarget.getLabel()
        frame = str(trackableTarget.getFrame(), encoding='utf-8')
        payload = {
            "trainingResult":{
                "time": ts,
                "img": frame,
                "result": label,
                "bbox": bbox,
                "score": score.tolist()
            }
        }
        response = requests.post('https://3d76a619.ngrok.io/api/uploadTrainingResult', headers=headers, data=json.dumps(payload))