import cv2
import handtrackingmodule as htm
import numpy as np
import os

brushThickness = 15
eraserThickness = 100
drawColor = (255, 0, 255)

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Load Header Images
folderPath = "Header"
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in os.listdir(folderPath)]
header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.75, maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]  # Index Finger Tip
        x2, y2 = lmList[12][1], lmList[12][2]  # Middle Finger Tip

        fingers = detector.fingersUp()

        # Selection Mode - Two fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0  # Reset drawing position
            if y1 < 125:
                if 250 < x1 < 450:
                    header, drawColor = overlayList[0], (255, 0, 255)
                elif 550 < x1 < 750:
                    header, drawColor = overlayList[1], (255, 0, 0)
                elif 800 < x1 < 950:
                    header, drawColor = overlayList[2], (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header, drawColor = overlayList[3], (0, 0, 0)

                xp, yp = 0, 0  # Ensure no line is drawn from the previous position

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # **Drawing Mode** - Index finger up
        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
            cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)

            xp, yp = x1, y1

    # **Smooth Overlay**
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # **Set Header**
    img[0:125, 0:1280] = header

    cv2.imshow("Image", img)
    cv2.waitKey(1)
