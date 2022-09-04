import cv2
import numpy as np
import face_recognition as face_rec
import os
import pyttsx3 as textSpeach
from datetime import  datetime

engine = textSpeach.init()

def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

path = 'student_images'
studentImg = []
studentName = []
myList = os.listdir(path)
for cl in myList :
    curimg = cv2.imread(f'{path}/{cl}')
    studentImg.append(curimg)
    studentName.append(os.path.splitext(cl)[0])

def findEncoding(images) :
    imgEncodings = []
    for img in images :
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings
def MarkAttendence(name):
    with open('attendence.csv', 'r+') as f:
        myDatalist =  f.readlines()
        nameList = []
        for line in myDatalist :
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            print("True")
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            datestr = now.strftime("%B %d, %Y")
            f.writelines(f'\n{name}, {timestr}, {datestr}')
            statement = str('Welcome ' + name)
            print(statement)
            engine.say(statement)
            engine.runAndWait()




EncodeList = findEncoding(studentImg)

vid = cv2.VideoCapture(0)
while True :
    success, frame = vid.read()
    frames = cv2.resize(frame, (0,0), fx =0.25, fy= 0.25)

    facesInFrame = face_rec.face_locations(frames)
    encodeFacesInFrame = face_rec.face_encodings(frames, facesInFrame)
    count=0
    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame) :
        matches = face_rec.compare_faces(EncodeList, encodeFace)
        facedis = face_rec.face_distance(EncodeList, encodeFace)
        #print(facedis)
        matchIndex = np.argmin(facedis)
        #print(matches,matchIndex)
            #print(check)
        if False in matches:
            count=count+1
        print(count)
        if count==0:    
            cv2.putText(frame,"Unknown", (50,150), cv2.FONT_ITALIC,1,(255,0,255),2)
        #print(count)
        #if count==0:
        #    cv2.putText(frame,"Unknown", (0,0), cv2.FONT_ITALIC,2,(255,0,255),2)
        
        if matches[matchIndex] :
            name = studentName[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            #y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
            #print(name)
            MarkAttendence(name)

    cv2.imshow('video',frame)
    cv2.waitKey(1)
