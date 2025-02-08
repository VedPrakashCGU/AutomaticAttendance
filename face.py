import os
import cv2
import face_recognition
import csv
import numpy as np
from datetime import datetime
known_encoding=[]
names=[]

path="./knownImages"
for file in os.listdir(path):
    imgFile=os.path.join(path,file)
    if imgFile:
        img=cv2.imread(imgFile)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        imgLoc=face_recognition.face_locations(img)
        img_encoding=face_recognition.face_encodings(img,imgLoc)
        if img_encoding:
            known_encoding.append(img_encoding[0])
            names.append(os.path.splitext(file)[0])
marked_name=set()
attend="Attendance.csv"
def attendance(name):
    if name not in marked_name:
        now=datetime.now()
        date=now.strftime("%d/%m/%Y")
        time=now.strftime("%H-%M-%S")
        exist=os.path.isfile(attend)
        with open(attend,"a") as f:
            writer=csv.writer(f)
            if not exist:
                writer.writerow(["Name","Time","Date"])
            writer.writerow([name,time,date])
        marked_name.add(name)

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        print("Frame is not readable")
        break
    rgbFrame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frameLoc=face_recognition.face_locations(rgbFrame)
    face_encod=face_recognition.face_encodings(rgbFrame,frameLoc)
    for (top,right,botom,left),encode in zip(frameLoc,face_encod):
        matches=face_recognition.compare_faces(known_encoding,encode)
        face_dist=face_recognition.face_distance(known_encoding,encode)
        if True in matches:
            idx=np.argmin(face_dist)
            namess=names[idx]
            attendance(namess)
            cv2.rectangle(frame,(left,top),(right,botom),(0,255,0),2)
            cv2.putText(frame,namess,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    cv2.imshow("attendance",frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
            
        
    
    
    
    
    