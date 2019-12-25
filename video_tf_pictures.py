'''
输入：原始视频文件
输出：从原始视频中提取出图像并进行失真矫正后保存
'''
import numpy as np
from cv2 import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle

video_input = 'GOPR1772.MP4'
cap = cv2.VideoCapture(video_input)
fps=cap.get(cv2.CAP_PROP_FPS) # 获取帧率
totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT) #获取总帧数
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 获取宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 获取高度
print('\n帧率: ' + str(fps) , '\n总帧数: ' + str(totalFrameNumber), '\n像素宽度: ' + str(width), '\n像素高度: ' + str(height))


dist_pickle = pickle.load( open( "./Demo_camera_cal/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

#图像失真矫正
def img_undistort(img,mtx,dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

count = 0
#提取视频的频率，每多少帧提取一个
frameFrequency=1
while count < totalFrameNumber:
    ret, image = cap.read()
    count += 1
    if (count % frameFrequency) == 0:
        if ret:
            undistort_image = img_undistort(image, mtx, dist)
            cv2.imwrite('./GOPR1772_image/' + str(count) + '.jpg', undistort_image)
            print(str(count))
        else:
            print(str(count)+' not ret, not image')
print(".........end.........")
cap.release()
