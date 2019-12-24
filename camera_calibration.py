import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
%matplotlib inline
%matplotlib qt

#找棋盘角，相机标定得到相机矩阵和失真系数

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
Chessboard_images = glob.glob('camera_cal/cal*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(Chessboard_images):
    chessboard_img = cv2.imread(fname)
    chessboard_gray = cv2.cvtColor(chessboard_img, cv2.COLOR_BGR2GRAY)
    img_size = (chessboard_img.shape[1], chessboard_img.shape[0])
    
    # Find the chessboard corners
    chessboard_ret, chessboard_corners = cv2.findChessboardCorners(chessboard_gray, (9,6), None)

    # If found, add object points, image points
    if chessboard_ret == True:
        objpoints.append(objp)
        imgpoints.append(chessboard_corners)
        print(fname + " test ok.")
        # Draw and display the corners
        cv2.drawChessboardCorners(chessboard_img, (9,6), chessboard_corners, chessboard_ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        cv2.imshow('img', chessboard_img)
        cv2.waitKey(500)
    else:
        print(fname + " 棋盘图像没有正确检测")
cv2.destroyAllWindows()

calibrate_ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)    
# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )