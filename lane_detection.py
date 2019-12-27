'''
输入：车辆前置摄像头原始视频文件
输出：车辆当前位置与车道线中心位置的偏差实时值的视频文件
注：
Lines_num = 1   表示道路只有一条中心车道线，通过寻中间车道线计算车辆偏离路中心偏差
Lines_num = 2   表示道路只有左右两侧边缘线，通过两侧两条线计算车辆偏离路中心偏差
'''
import numpy as np
from cv2 import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle


dist_pickle = pickle.load( open( "camera_cal/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

video_input = 'challenge_video.mp4'
video_output = 'result_challenge_video1.mp4'

cap = cv2.VideoCapture(video_input)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_output, fourcc, 20.0, (1280, 720))

detected = False
left_fit = []
right_fit = []
ploty = []
i=0
while(True):
    ret, image = cap.read()
    #print(ret)
    if ret:
        i=i+1
        print(i)
        undistort_image = img_undistort(image, mtx, dist)
        #color_result_img,combined_binary_img = pipeline(undistort_image)
        #top_down_warped_img,Minv = warper(combined_binary_img)
        top_down_warped,Minv = warper(undistort_image)
        color_result_img,combined_binary_img = pipeline(top_down_warped)
        '''
        warp_image, M, Minv = warpImage(undistort_image, src, dst)
        hlsL_binary = hlsLSelect(warp_image)
        labB_binary = labBSelect(warp_image, (205, 255))
        combined_binary = np.zeros_like(sx_binary)
        combined_binary[(hlsL_binary == 1) | (labB_binary == 0)] = 1
        '''
        #print('detected = ',detected)
        #left_fit_warped,right_fit_warped,warped_leftx, warped_lefty, warped_rightx, warped_righty,left_fitx, right_fitx,ploty,result = fit_polynomial(combined_binary_img)
        
        if detected == False:
            left_fit_warped,right_fit_warped,warped_leftx, warped_lefty, warped_rightx, warped_righty,left_fitx, right_fitx,ploty,result = fit_polynomial(combined_binary_img)
            #out_img, left_fit, right_fit, ploty = fit_polynomial(combined_binary, nwindows=9, margin=80, minpix=40)
            #print(len(right_fit))
            #print(len(right_fit)>0 and len(left_fit)>0)
            print('k')
            if (len(left_fit_warped) > 0 and len(right_fit_warped) > 0) :
                detected = True
            else :
                #print('1')
                detected = False
        else:            
            left_fit_warped,right_fit_warped,warped_leftx, warped_lefty, warped_rightx, warped_righty,left_fitx, right_fitx,ploty,result = search_around_poly(left_fit_warped,right_fit_warped,combined_binary_img)
            #track_result, left_fit, right_fit, ploty,  = search_around_poly(combined_binary, left_fit, right_fit)
            print('l')
            #print(left_fit_warped)
            #print(right_fit_warped)
            if (len(left_fit_warped) > 0 and len(right_fit_warped) > 0) :
                detected = True
            else :
                detected = False
        
        #result = drawing(undistort_image, combined_binary, warp_image, left_fitx, right_fitx)
        draw_result = drawing(undistort_image , combined_binary_img,Minv, left_fitx, right_fitx, ploty)
        out.write(draw_result)
    else:
        break
        
cap.release()
out.release()