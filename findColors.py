import cv2
import numpy as np


def nothing(*arg):
    pass


icol = (0, 0, 0, 255, 255, 255)  # White

cv2.namedWindow('colorTest')
# Lower range colour sliders.
cv2.createTrackbar('lowHue', 'colorTest', icol[0], 255, nothing)
cv2.createTrackbar('lowSat', 'colorTest', icol[1], 255, nothing)
cv2.createTrackbar('lowVal', 'colorTest', icol[2], 255, nothing)
# Higher range colour sliders.
cv2.createTrackbar('highHue', 'colorTest', icol[3], 255, nothing)
cv2.createTrackbar('highSat', 'colorTest', icol[4], 255, nothing)
cv2.createTrackbar('highVal', 'colorTest', icol[5], 255, nothing)

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()

    blur = cv2.GaussianBlur(frame, (15, 15), 0)

    # Show the original image.
    cv2.imshow('Frame', frame)
    cv2.imshow('Blurred Img', blur)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Get HSV values from the GUI sliders.
    lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
    lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
    lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')

    highHue = cv2.getTrackbarPos('highHue', 'colorTest')
    highSat = cv2.getTrackbarPos('highSat', 'colorTest')
    highVal = cv2.getTrackbarPos('highVal', 'colorTest')

    # HSV values to define a colour range.
    colorLow = np.array([lowHue, lowSat, lowVal])
    colorHigh = np.array([highHue, highSat, highVal])
    mask = cv2.inRange(hsv, colorLow, colorHigh)

    # Show the first mask
    cv2.imshow('mask-plain', mask)

    # Put mask over top of the original image.
    result = cv2.bitwise_and(blur, blur, mask=mask)

    # Show final output image
    cv2.imshow('colorTest', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
