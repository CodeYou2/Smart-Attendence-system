from tkinter import *
from tkinter import messagebox 
from PIL import Image,ImageTk
import cv2
import numpy as np
import face_recognition as face_rec
import os
import pyttsx3 as textSpeach
from datetime import  datetime

import numpy as np
import os
import pandas as pd
# from ctypes import cast, POINTER
# ##################
# Cam, hCam = 640, 480w
##################

root=Tk()
root.title("Smart Attendance")
root.geometry("600x600")
Label(root,text="Take attendance with you camera",font=("Calibri 13 bold")).grid(row=0,column=1)
desc_user=Label(root,text="""Attendance made easy""", font=5)
desc_user.grid(row=1,column=1)
# pic_label=Label(image=PhotoImage(file="ngo.png")).grid(row=2,column=0)



def register():
    #write the code
    stuName=input()
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        #stuName=input()
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            pathway = 'student_images'
            img_name = "{}.jpeg".format(stuName)
            cv2.imwrite(os.path.join(pathway , img_name), frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

def attendance():
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
                now = datetime.now()
                timestr = now.strftime('%H:%M')
                f.writelines(f'\n{name}, {timestr}')
                statment = str('welcome to class' + name)
                engine.say(statment)
                engine.runAndWait()




    EncodeList = findEncoding(studentImg)

    vid = cv2.VideoCapture(0)
    while True :
        success, frame = vid.read()
        Smaller_frames = cv2.resize(frame, (0,0), None, 0.25, 0.25)

        facesInFrame = face_rec.face_locations(Smaller_frames)
        encodeFacesInFrame = face_rec.face_encodings(Smaller_frames, facesInFrame)

        for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame) :
            matches = face_rec.compare_faces(EncodeList, encodeFace)
            facedis = face_rec.face_distance(EncodeList, encodeFace)
            print(facedis)
            matchIndex = np.argmin(facedis)

            if matches[matchIndex] :
                name = studentName[matchIndex].upper()
                y1, x2, y2, x1 = faceloc
                #y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                MarkAttendence(name)

        cv2.imshow('video',frame)
        cv2.waitKey(1)

def button():
    #write the code
    df=pd.read_csv('D:\Works\Opencv project\Attendance count using face recognition\student ML\stuData.csv').dropna(axis=1)
    def  name():
        name=input('Enter the name of the employee: ')
        mask=df['Name']==name
        df1=df[mask]
        df1.sort_values('Date',ascending=True)
        print(df1)
        return df1
    name()


Button(root,text="Register",command=register).grid(row=2,column=2,padx=20,pady=150)
Button(root,text="Attendance",command=attendance).grid(row=2,column=1,padx=20,pady=150)
Button(root,text="Button",command=button).grid(row=2,column=0,padx=20,pady=150)
Button(root,text="Exit",command=root.destroy).grid(row=3,column=0,padx=20,pady=150)






root.mainloop()