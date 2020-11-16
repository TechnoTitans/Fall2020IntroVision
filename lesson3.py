import numpy as np
import cv2

# Specifies which camera to use: Start at 0, and go up until you find the cam u wanna use
cap = cv2.VideoCapture(2)


# Params: event (int), x location of mouse, y location of mouse
def print_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(frame[y][x])


while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Params: str: name of window, source image
    cv2.imshow('Frame', frame)

    # Params: source image, kernel size (as a tuple), sigma for gaussian distribution (default 0)
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    cv2.imshow('Gaussian Blur Image', blur)

    # Params: source image, color conversion: from BGR (Blue Green Red) to HSV (Hue, Saturation, Value)
    # All caps variables are generally constants
    hsv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Params: Source Window, callback function
    # when a mouse event occurs on the source window window, the callback function is called
    cv2.setMouseCallback('Gaussian Blur Image', print_color)

    # === Feature Extraction: find desired "feature" or in our case object in frame ===

    lower_color_bound = np.array([5, 64, 150])
    upper_color_bound = np.array([39, 224, 255])

    # Params: source image, lower bound, upper bound
    # if the hsv values of a pixel on the frame are within the lower and upper bounds, then that pixel
    # is marked as a 1, and other pixels are marked as 0s
    # This type of image is called a mask and only has 0s or 1s (colors are defined by one bit)
    color_mask = cv2.inRange(hsv_frame, lower_color_bound, upper_color_bound)

    cv2.imshow('Mask Image', color_mask)

    # Params: Source image, use default, use default (google if you really want to know)
    # a contour is just a group of touching pixels
    # findContours returns an array of contours and a hierarchy array
    contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Params: input image, contour array, contour indexes to draw (-1 draws all), color to draw contours in as a tuple
    contour_image = cv2.drawContours(frame.copy(), contours, -1, (255, 0, 0))
    cv2.imshow('Contour Image', contour_image)

    # bitwise-and is just applying mask to original frame
    # Params: second input image, second input image, mask of pixels to keep
    result = cv2.bitwise_and(frame, frame, mask=color_mask)
    cv2.imshow('Resulting Image', result)

    # Task is to come up with certain metrics and bounds on those metrics
    # to isolate the desired contour and draw a box around it
    # Use this: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html

    # Area
    # Orientation
    # Aspect Ratio

    # For every feature candidate we want to generate a score
    # candidate with max score, is our object

    if len(contours) == 0:
        continue

    scores = np.zeros(len(contours))

    # delete below TRY NOT TO COPY HEREEEEEEE YOU CAN COPY ABOVE IF YOU REALLY HAVE TO BUT DONT COPY BELOW

    area_weight = 0.4
    ideal_area = 30000

    orientation_weight = 0

    aspect_ratio_weight = 0.6
    ideal_aspect_ratio = 0.636363636

    # creates an 0 np array of same outer dimensions as input param
    for i in range(len(contours)):
        # Area Metric
        contour_area = cv2.contourArea(contours[i])
        # https://numpy.org/doc/stable/reference/generated/numpy.clip.html
        area_score = np.clip(contour_area / ideal_area, 0, 1)

        # Orientation Metric
        # Try to execute this code, if an issue pops up, do the except
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(contours[i])

            orientation_score = np.abs((1 / 90) * (angle - 90))
        except:
            orientation_score = 0

        # Aspect Ratio Metric
        try:
            rect = cv2.minAreaRect(contours[i])

            shorter_side = np.min(rect[1][0], rect[1][1])
            longer_side = np.max(rect[1][0], rect[1][1])

            aspect_ratio = shorter_side / longer_side

            aspect_ratio_score = -1 * np.abs((1 / ideal_aspect_ratio) * (aspect_ratio - ideal_aspect_ratio)) + 1
        except:
            aspect_ratio_score = 0

        scores[i] = area_weight * area_score + orientation_weight * orientation_score + aspect_ratio_weight * aspect_ratio_score

    # delete above

    best_contour = contours[np.argmax(scores)]

    x, y, w, h = cv2.boundingRect(best_contour)

    # params: source image, top left rectangle coordinate tuple, bottom right rectangle coordinate tuple,
    # tuple of rectangle color, rectangle width (in pixels)
    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Image with bounding box", frame)

cap.release()
cv2.destroyAllWindows()
