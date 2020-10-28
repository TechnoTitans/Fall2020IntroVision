import numpy as np
import cv2

# Specifies which camera to use
cap = cv2.VideoCapture(2)


def print_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(frame[y][x])


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow('Frame', frame)

    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    cv2.imshow('Gaussian Blur Image', blur)

    hsv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    cv2.setMouseCallback('Gaussian Blur Image', print_color)

    # Feature Extraction: find desired object in frame

    lower_color_bound = np.array([20, 20, 100])
    upper_color_bound = np.array([90, 80, 255])

    # mask is image of only 0 or 1 (colors are defined by one bit)
    color_mask = cv2.inRange(blur, lower_color_bound, upper_color_bound)
    cv2.imshow('Mask Image', color_mask)

    contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = cv2.drawContours(frame.copy(), contours, -1, (255, 0, 0))
    cv2.imshow('Contour Image', contour_image)

    # bitwise-and is just applying mask to original frame
    result = cv2.bitwise_and(frame, frame, mask=color_mask)
    cv2.imshow('Resulting Image', result)

    # Task is to come up with certain metrics and bounds on those metrics
    # to isolate the desired contour and draw a box around it

    aspect_ratios = np.zeros_like(contours)

    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])

        aspect_ratios[i] = float(w) / h

    max_aspect_ratio_contour_index = np.argmax(aspect_ratios)

    x, y, w, h = cv2.boundingRect(contours[max_aspect_ratio_contour_index])

    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Image with bounding box", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
