import requests
import calendar
import time
import json
import base64

def uploadResult(license, enter_count, exit_count):
    headers = {'Content-Type': 'application/json'}
    payload = {
        "license": license,
        "enter_count": enter_count,
        "exit_count": exit_count
    }
    response = requests.post('https://75e16d9c.ngrok.io/api/uploadResult', headers=headers, data=json.dumps(payload))

def getResults():
    response = requests.get('https://75e16d9c.ngrok.io/api/getResults')
    print(response)
    
#uploadResult('AB777', 10, 10)
#getResults()