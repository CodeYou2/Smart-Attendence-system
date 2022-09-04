import cv2
import os
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
        pathway = 'D:\Works\Opencv project\Attendance count using face recognition\student_images'
        img_name = "{}.jpeg".format(stuName)
        cv2.imwrite(os.path.join(pathway , img_name), frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()