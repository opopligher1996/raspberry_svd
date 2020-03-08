class RequestWorker():
    def __init__(self):
        print('start thread')
    
    def sendRequest(self):
        import requests
        payload = {
            "time": "1234567890",
            "img": "kkkkk",
            "result": "person"
        }
        response = requests.post('https://3d76a619.ngrok.io/api/uploadTrainingResult', data = payload)
        print(response)
        
    def run(self):
        import threading
        import time
        while True:
            self.sendRequest()
            time.sleep(2)