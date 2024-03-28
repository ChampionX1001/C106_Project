import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

vid = cv2.VideoCapture('walking.avi')

while True:
    ret, frame = vid.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, 1.2, 5)
    faces = faceCascade.detectMultiScale(grey)
    print(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('boyImg', frame)
        if cv2.waitKey(25) == 25:
            break
    

vid.release()
cv2.destroyAllWindows()