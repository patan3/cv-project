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

def run_calibration(_objpoints, _imgpoints):
    """
    Solves for K, dist, R, t using the provided images.
    """
    
    # Flags:
    # CALIB_FIX_ASPECT_RATIO for fx != fy and CALIB_FIX_PRINCIPAL_POINT to enforce center
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        _objpoints, 
        _imgpoints, 
        gray.shape[::-1], # Image Size (width, height)
        None, 
        None,
        flags=0 # Ensure fx != fy AND camera center to be estimated
    )

    return ret, mtx, dist, rvecs, tvecs

# 3D World Points (X_world) #
# define the board coordinates in the 3D world with z=0
# i.e., the points are (0,0,0), (1.7,0,0), (3.4,0,0), etc.
square_size = 0.017 # 1.7 cm in meters
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp *= square_size

# Arrays to store object points and image points from all the images for runs 1-3.
auto_objpoints = [] # 3D points for auto-detected images
auto_imgpoints = [] # 2D points for auto-detected images
manual_objpoints = [] # 3D points for manually clicked images
manual_imgpoints = [] # 2D points for manually clicked images

images = glob.glob('images/*.jpg')
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        auto_objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        auto_imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.circle(img, tuple(corners2[0][0].astype(int)), 10, (0, 255, 0), -1) # Green = Start
        cv.circle(img, tuple(corners2[-1][0].astype(int)), 10, (0, 0, 255), -1) # Red = End
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        points4 = get_four_corners_manually(img)
        corners = interpolate_chessboard_corners(points4, pattern_size=(6,9))
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        manual_objpoints.append(objp)
        manual_imgpoints.append(corners2)
        
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.circle(img, tuple(corners2[0][0].astype(int)), 10, (0, 255, 0), -1) # Green = Start
        cv.circle(img, tuple(corners2[-1][0].astype(int)), 10, (0, 0, 255), -1) # Red = End
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



# - RUN 1: Use ALL 25 images (Auto + Manual) -
run1_obj = auto_objpoints + manual_objpoints
run1_img = auto_imgpoints + manual_imgpoints

rms1, K1, dist1, rvecs1, tvecs1 = run_calibration(run1_obj, run1_img)
print(f"Run 1 Reprojection Error: {rms1}")
print("K Matrix Run 1:\n", K1)

# - RUN 2: Use 5 Auto + 5 Manual -
# (we take the first 5 from auto, in total we have 7)
run2_obj = auto_objpoints[:5] + manual_objpoints
run2_img = auto_imgpoints[:5] + manual_imgpoints

rms2, K2, dist2, rvecs2, tvecs2 = run_calibration(run2_obj, run2_img)
print(f"Run 2 Reprojection Error: {rms2}")
print("K Matrix Run 2:\n", K2)

# - RUN 3: Use 5 Auto -
run3_obj = auto_objpoints[:5]
run3_img = auto_imgpoints[:5]

rms3, K3, dist3, rvecs3, tvecs3 = run_calibration(run3_obj, run3_img)
print(f"Run 3 Reprojection Error: {rms3}")
print("K Matrix Run 3:\n", K3)