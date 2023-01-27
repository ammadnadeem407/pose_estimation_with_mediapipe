import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import math


class PoseDetection():
    def __init__(self, image_mode=False,
                 model_compl=1,
                 smooth_lm=True,
                 enable_seg=False,
                 smooth_seg=True,
                 min_detection_conf=0.5,
                 min_tracking_conf=0.5):

        self.image_mode = image_mode
        self.model_compl = model_compl
        self.smooth_lm = smooth_lm
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.image_mode, self.model_compl, self.smooth_lm,
                                     self.enable_seg, self.smooth_seg, self.min_detection_conf, self.min_tracking_conf)

    def find_pose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        self.lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (0, 0, 255), cv.FILLED)
        return self.lm_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        if len(self.lm_list) != 0:
            x1, y1 = self.lm_list[p1][1:]
            x2, y2 = self.lm_list[p2][1:]
            x3, y3 = self.lm_list[p3][1:]

            angle = math.degrees(math.atan2(
                y3-y3, x3-x2)-math.atan2(y1-y2, x1-x2))
            if angle < 0:
                angle += 360

            if draw:
                cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv.line(img, (x3, y3), (x2, y2), (255, 0, 0), 3)
                cv.circle(img, (x1, y1), 10, (0, 255, 0), cv.FILLED)
                cv.circle(img, (x1, y1), 15, (0, 255, 0), 2)
                cv.circle(img, (x2, y2), 10, (0, 255, 0), cv.FILLED)
                cv.circle(img, (x2, y2), 15, (0, 255, 0), 2)
                cv.circle(img, (x3, y3), 10, (0, 255, 0), cv.FILLED)
                cv.circle(img, (x3, y3), 15, (0, 255, 0), 2)
                cv.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                           cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            return angle


def main():
    cap = cv.VideoCapture(0)

    # fourcc = cv.VideoWriter_fourcc(*'mp4v')
    # out = cv.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
    pTime = 0
    detector = PoseDetection()
    while True:
        success, img = cap.read()
        img = detector.find_pose(img)
        lm_list = detector.find_position(img, draw=False)
        # if len(lm_list) != 0:
        #     print(lm_list)
        #     # cv.circle(img, (lm_list[12][1], lm_list[12][2]),
        #     #           15, (0, 255, 0), cv.FILLED)
        angle = detector.find_angle(img, 16, 14, 12)
        # print(angle)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (70, 50),
                   cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv.imshow('CAM', img)
        # out.write(img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


    # out.release()
if __name__ == "__main__":
    main()
