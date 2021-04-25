import cv2
face_id = input('Enter your face id: ')
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
count = 1
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.05, minNeighbors = 10)
    for x, y, w, h in faces:
        print(count)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite('face_dataset/' + str(face_id) + '.'  +  str(count) + ".jpg", gray[y:y+h,x:x+w])
        count+=1
        
    frame = cv2.flip(frame, 1)
    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == 27:   #Esc to quit
        break
    elif count>30:
        break
video.release()
cv2.destroyAllWindows()
