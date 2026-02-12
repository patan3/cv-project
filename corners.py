#length is 1.7cm
#9x6 is internal corners
import numpy as np
import cv2 as cv
import glob

def get_four_corners_manually(img, window="img"):
    points = []

    def click_event(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
            print(x, y)
            points.append((x, y))
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(img, f"{x},{y}", (x, y), font, 1, (255, 0, 0), 2)
            cv.imshow('img', img)

    cv.namedWindow(window, cv.WINDOW_NORMAL)  
    cv.setMouseCallback(window, click_event)

    while True:
        cv.imshow('img', img)
        cv.waitKey(100)
        if len(points) == 4:
            cv.destroyWindow(window)
            return np.array(points, dtype=np.float32)
        
def interpolate_chessboard_corners(points, pattern_size):
    cols, rows = pattern_size
    tl, tr, br, bl = points

    corners = []
    for j in range(rows):
        v = j / (rows - 1) if rows > 1 else 0.0

        # left and right endpoints of this row
        left = (1 - v) * tl + v * bl
        right = (1 - v) * tr + v * br

        for i in range(cols):
            u = i / (cols - 1) if cols > 1 else 0.0
            p = (1 - u) * left + u * right
            corners.append(p)

    corners = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
    return corners

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

square_size = 0.017
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp *= square_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('images/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        points4 = get_four_corners_manually(img)
        corners = interpolate_chessboard_corners(points4, pattern_size=(6,9))
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)
        
cv.destroyAllWindows()

#bad ones:
"""
images\WIN_20260211_15_53_19_Pro.jpg
images\WIN_20260211_15_53_23_Pro.jpg
images\WIN_20260211_15_53_25_Pro.jpg
images\WIN_20260211_15_53_35_Pro.jpg
images\WIN_20260211_15_54_49_Pro.jpg
images\WIN_20260211_15_54_53_Pro.jpg
images\WIN_20260211_15_54_57_Pro.jpg
"""
