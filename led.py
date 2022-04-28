from gpiozero import LED
from time import sleep
import time
import pyrebase
#from firebasedata import LiveData
class LED_code:
    config = {
    "apiKey": "AIzaSyBqH-1SFbLkCjaW_Qilj7TQq-2E3cy43FQ",
    "authDomain": "iot-project-e0803.firebaseapp.com",
    "databaseURL": "https://iot-project-e0803-default-rtdb.firebaseio.com/",
    "projectId": "iot-project-e0803",
    "storageBucket": "iot-project-e0803.appspot.com",
    "messagingSenderId": "292898203599",
    "appId": "1:292898203599:web:382bc1a3eb5c9e28c4d4ba",
    "measurementId": "G-FY4R61HR0P"
  	};
    app = pyrebase.initialize_app(config)
    db=app.database()
    flag = 1
    Green=LED(18)
    Red=LED(8)
    while(1):
        i=db.child('/Led/flag').get()
        print(i.val())
        if(i.val()==1):
            Red.off()
            Green.on()
            time.sleep(5)
            db.child('/Led').set({
            'flag': 0})
            Green.off()
            flag=1
        if(i.val()==0 and flag ==1):
            flag = 0
            Red.on()