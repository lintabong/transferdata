from keras.models import load_model
import cv2
import mediapipe
import numpy


# setting mask detector (Machine Learning)
model = load_model('model0.h5')
face_detection = mediapipe.solutions.face_detection.FaceDetection()
x, y, w, h = 0, 0, 0, 0

CATEGORIES = ['mask', 'no mask']
index = 0
prediction = 0
cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()
    img = frame.copy()

    frame = cv2.flip(frame, 1)

    img = cv2.flip(img, 1)
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

cap.release()
cv2.destroyAllWindows()
