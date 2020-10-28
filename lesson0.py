import numpy as np
import cv2

# Specifies which camera to use
cap = cv2.VideoCapture(0)

# Calculating size of image:
# each color is 8 bits ==> byte
# each pixel is 8 * 3 bits = 24 bits, 3 bytes
# 640x480 = 307,200
# 307200 * 3 bytes = 921,600 bytes
# 1 kilo bytes = 1024 bytes
# 900 kilo bytes
# 1 MB = 1024 kB
# image is 0.88 Mega Bytes
# 30 FPS
# 26.4 MB/s
# 210.9 Mbps (mega bits per second)

# insane compression!!!!!

print(((640*480*3)/(1024**2))*30*8)

while True:
    # Capture frame-by-frame
    # has 3 color channels (BGR) 0-255
    ret, frame = cap.read()
    cv2.imshow('Frame', frame)

    # print(ret)
    # print(frame)
    # frame, kernel size, standard deviation (just keep at 0 for default)
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    cv2.imshow('Gaussian Blur Image 5', blur)

    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    cv2.imshow('Gaussian Blur Image 15', blur)

    # Our operations on the frame come here
    # stores one color channel 0-255
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # print(gray)

    # Display the resulting frame
    # Params: frame name, image object
    cv2.imshow('Gray Image', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
