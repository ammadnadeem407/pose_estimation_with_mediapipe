import cv2 as cv
import time
import pose_module as pm
import math


cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'H264')
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
pTime = 0
detector = pm.PoseDetection()
while True:
    success, img = cap.read()
    img = detector.find_pose(img)
    lm_list = detector.find_position(img, draw=False)
    # if len(lm_list) != 0:
    #     print(lm_list[12])
    #     cv.circle(img, (lm_list[12][1], lm_list[12][2]),
    #               15, (0, 255, 0), cv.FILLED)
    angle = detector.find_angle(img, 16, 14, 12)
    # print(angle)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70, 50),
               cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv.imshow('CAM', img)
    out.write(img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
