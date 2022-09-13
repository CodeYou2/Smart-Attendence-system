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

root=Tk()
root.title("Smart Attendance")
root.configure(bg='#E3FDFD')
root.geometry("500x500")
top=Label(root,text="SMART ATTENDENCE",font=("Calibri 13 bold"))
top.configure(bg='#CBF1F5')
top.place(relx=0.31,rely=0.025)
desc_user=Label(root,text="""Attendance made easy :)""", font=5)
desc_user.configure(bg='#CBF1F5')
desc_user.place(relx=0.22,rely=0.1)
# pic_label=Label(image=PhotoImage(file="ngo.png")).grid(row=2,column=0)

def register():
    stuName=input("Enter your name")
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
            pathway = 'C:\Attendence\student_images'
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
                print("true")
                now = datetime.now()
                timestr = now.strftime('%H:%M')
                datestr = now.strftime("%B %d, %Y")
                f.writelines(f'\n{name}, {timestr}, {datestr}')
                statement = str('Welcome' + name)
                print(statement)
                engine.say(statement)
                engine.runAndWait()




    EncodeList = findEncoding(studentImg)

    vid = cv2.VideoCapture(0)
    while True :
        success, frame = vid.read()
        Smaller_frames = cv2.resize(frame, (0,0), None, 0.25, 0.25)

        facesInFrame = face_rec.face_locations(Smaller_frames)
        encodeFacesInFrame = face_rec.face_encodings(Smaller_frames, facesInFrame)
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
            #print(count)
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
                cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                MarkAttendence(name)

        cv2.imshow('video',frame)
        cv2.waitKey(1)


def stats():
    df=pd.read_csv('D:\Works\Opencv project\Attendance count using face recognition\student ML\stuData.csv').dropna(axis=1)
    def  name():
        name=input('Enter the name of the employee: ')
        mask=df['Name']==name
        df1=df[mask]
        df1.sort_values('Date',ascending=True)
        print(df1)
        return df1
    name()


b1=Button(root,text="Register 👍",command=register,bg='#A6E3E9')
b1.place(relx=0.13,rely=0.65,width=80,height=30)

b2=Button(root,text="Attendance 😎",command=attendance,bg='#A6E3E9')
b2.place(relx=0.425,rely=0.65,width=80,height=30)

b3=Button(root,text="Statistics 📈",command=stats,bg='#A6E3E9')
b3.place(relx=0.725,rely=0.65,width=80,height=30)


b4=Button(root,text="Exit 👋",command=root.destroy,bg='#71C9CE')
b4.place(relx=0.425,rely=0.85,width=75,height=30)







root.mainloop()
