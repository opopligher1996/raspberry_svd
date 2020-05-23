import requests
import calendar
import time
import json
import base64

def uploadHeartBeat():
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post('http://staging.socif.co:8011/api/uploadHeartBeat')
    except:
        print('upload Heart Beat Error')
    
def uplaodFullAlert():
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post('http://staging.socif.co:8011/api/uploadFullAlert')
    except:
        print('upload Full Alert Error')
    
def uploadResult(license, enter_count, exit_count):
    try:
        headers = {'Content-Type': 'application/json'}
        payload = {
            "license": license,
            "enter_count": enter_count,
            "exit_count": exit_count
        }
        response = requests.post('http://staging.socif.co:8011/api/uploadResult', headers=headers, data=json.dumps(payload))
    except:
        print('upload Result Error')
        
def getResults():
    response = requests.get('https://staging.socif.co:8011/api/getResults')
    print(response)
    
#uploadResult('AB777', 10, 10)
#getResults()