import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 640)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

segmentor = SelfiSegmentation(1)
BGimages = []
for img in os.listdir("/Users/shashwateshtripathi/Downloads/images2"):
    BGimages.append(cv2.imread(f"/Users/shashwateshtripathi/Downloads/images2/{img}"))

imgIndex = 0
right = False
left = False
i=0
while True:
    success, img = cap.read()
    if i>3:
        bgimage = cv2.resize(BGimages[imgIndex], (640,480))
        img2 = segmentor.removeBG(img, bgimage, threshold=0.6)
        imgStack = cvzone.stackImages([img, img2],2,1)

        cv2.imshow("Video", imgStack)
        cv2.waitKey(1)


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        multiLandMarks = results.multi_hand_landmarks

        if multiLandMarks:
            handPoints = []
            for handLms in multiLandMarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                for idx, lm in enumerate(handLms.landmark):
                    # print(idx,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    handPoints.append((cx, cy))
            print(handPoints[8])
            if not left and handPoints[8][0] < 220:
                left = True
                if imgIndex > 0:
                    imgIndex -= 1
                    print("Left")
            if not right and handPoints[8][0] > 420:
                right = True
                if imgIndex < 3:
                    imgIndex += 1
                    print("Right")


            if handPoints[8][0] > 220 and handPoints[8][0] < 420:
                left = False
                right = False
    i+=1



