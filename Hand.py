import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands  #Importing the features
hands = mpHands.Hands() #Setting up the Parameters in this case ,Setting it as default
mp_track = mp.solutions.drawing_utils # To draw the points on the palm

cTime = 0
pTime = 0

while True:
    success,img = cap.read()
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Converting the color as the mediapipe only takes rgb image
    results = hands.process(img_rgb) #Processing the image using built-in

    if results.multi_hand_landmarks:
        for hand_dot in results.multi_hand_landmarks:
            for id,lm in enumerate(hand_dot.landmark): #This will show us the landmark of the dots
                w,h,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                print(cx,cy)

            mp_track.draw_landmarks(img, hand_dot,mpHands.HAND_CONNECTIONS)#Shows the dots and connections

    cTime = time.time()         #The current timestamp
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 0), 3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)