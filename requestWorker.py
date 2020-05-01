import os
import requests

def getResult():
    r = requests.get('http://e8633280.ngrok.io/api/getResults')
    print(r.json())
    
getResult()