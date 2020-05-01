import requests
import calendar
import time
import json
import base64

class RequestWorker():
    def __init__(self):
        print('start thread')
    
    def uploadResult(self, license, enter_count, exit_count):
        headers = {'Content-Type': 'application/json'}
        payload = {
            "license": license,
            "enter_count": enter_count,
            "exit_count": exit_count
        }
        response = requests.post('https://3d76a619.ngrok.io/api/uploadTrainingResult', headers=headers, data=json.dumps(payload))