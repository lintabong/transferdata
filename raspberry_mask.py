from tensorflow.keras.models    import load_model
from serial                     import Serial
from serial.tools               import list_ports
from time                       import sleep
import pyrebase
import datetime
import cv2
import mediapipe
import numpy

# setting serial
listport = list_ports.comports()
for i in listport:
    print(i)

ser = Serial('COM9', 9600, timeout=0.05)

# setting firebase
firebaseConfig = {
    'apiKey': 'AIzaSyAa3YScorFNdoAKeGklFvwSH-wn40OfoQY',
    'authDomain': 'absensiraspberry.firebaseapp.com',
    'databaseURL': 'https://absensiraspberry-default-rtdb.firebaseio.com',
    'projectId': 'absensiraspberry',
    'storageBucket': 'absensiraspberry.appspot.com',
    'messagingSenderId': '561469182049',
    'appId': '1:561469182049:web:1f55ca257ce4f1a20d9a20',
    'measurementId': 'G-ECL04FT05M',
}

firebase    = pyrebase.initialize_app(firebaseConfig)
storage     = firebase.storage()
db          = firebase.database()
user        = db.child("user").get().val()
id_list     = []
idu         = ''
for i in user:
    id_list.append(i)

print(id_list)

# setting mask detector (Machine Learning)
model           = load_model('model0.h5')
face_detection  = mediapipe.solutions.face_detection.FaceDetection()
x, y, w, h      = 0, 0, 0, 0

CATEGORIES  = ['mask', 'no mask']
index       = 0
prediction  = 0
cap         = cv2.VideoCapture(0)

# function timde
def date():
    month = ['januari', 'pebruari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus', 'september', 'oktober',
             'nopember', 'desember']
    timenow = datetime.datetime.now()
    nYear   = str(timenow.year)
    nMonth  = int(timenow.month)
    nDay    = str(timenow.day)

    if int(timenow.hour) < 9: nHour = '0' + str(timenow.hour)
    else: nHour = str(timenow.hour)

    if int(timenow.minute) < 9: nMinute = '0' + str(timenow.minute)
    else: nMinute = str(timenow.minute)

    nHour = nHour + ':' + nMinute

    return str(nYear + '/' + month[nMonth - 1] + '/' + nDay + '/' + nHour)


while True:
    _, frame    = cap.read()
    img         = frame.copy()
    frame       = cv2.flip(frame, 1)
    try:
        height, width, channel = frame.shape
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detection.process(imgRGB)

        for count, detection in enumerate(result.detections):
            box = detection.location_data.relative_bounding_box
            x, y, w, h = int(box.xmin * width), int(box.ymin * height), int(box.width * width), int(box.height * height)

        imgcrop = img[y:y + h, x:x + w]
        imgcrop = cv2.resize(imgcrop, (100, 100))
        imgcrop = numpy.expand_dims(imgcrop, axis=0)

        prediction = model.predict(imgcrop)
        index = numpy.argmax(prediction)
        res = CATEGORIES[index]

        if index == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, res, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    except:
        pass

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

    if index == 0:
        print(CATEGORIES[index])
        ser.write('B'.encode())
        while True:
            sleep(0.014)
            rs  = str(ser.readline()).replace('b', '').replace('\\r\\n', '').replace('\'', '')
            rsa = rs.split('&')
            if len(rs) != 0:
                if len(rsa) > 1:
                    print('====================')
                    print('id finger: ', rsa[1])
                    print('id user  : ', id_list[int(rsa[1])])
                    print('id nama  : ', user[id_list[int(rsa[1])]]['username'])
                    print('suhu     : ', rsa[2])
                    data = {'suhu/' + id_list[int(rsa[1])] + '/' + date(): rsa[2]}
                    db.update(data)
                    break
    else:
        ser.write('A'.encode())
    sleep(0.01)

cap.release()
cv2.destroyAllWindows()
