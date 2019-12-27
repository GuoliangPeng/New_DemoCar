'''
输入：前置摄像头原始图像
测试：透视变换参数，图片二值化方式及参数
'''
import numpy as np
from cv2 import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle


dist_pickle = pickle.load( open( "Demo_camera_cal/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

#原始图像失真矫正
def img_undistort(img,mtx,dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

'''
#测试-失真矫正（对原始图像-->失真矫正）
img = cv2.imread('camera_cal/calibration2.jpg')
dst_img = img_undistort(img,mtx,dist)
cv2.imwrite('output_images/test_undist.jpg',dst_img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst_img)
ax2.set_title('Undistorted Image', fontsize=30)
plt.savefig('./writeup_images/image1.jpg')
'''
#透视变换
def warper(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[580, 460], [700, 460], [1096, 720], [200, 720]])
    dst = np.float32([[300, 0], [950, 0], [950, 720], [300, 720]])
    # Compute and apply perpective transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    #print(M)
    #print(Minv)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped,Minv
'''
#测试-透视变换参数（对原始直道图像-->失真矫正-->透视变换）
test_img1 = mpimg.imread('test_images/straight_lines1.jpg')
test_distort_image1 = img_undistort(test_img1,mtx,dist)
test_top_down_warped1,test_Minv1 = warper(test_distort_image1)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_distort_image1)
ax1.set_title('Undistorted Image', fontsize=50)
ax2.imshow(test_top_down_warped1)
ax2.set_title('Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('./writeup_images/image3.jpg')
'''

#渐变和色彩空间,输出二值化图像

def abs_sobel_thresh(img,orient,sxy_thresh):
     
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
    if orient == 'y':
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3) # Take the derivative in y
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sxybinary = np.zeros_like(scaled_sobel)
    sxybinary[(scaled_sobel >= sxy_thresh[0]) & (scaled_sobel <= sxy_thresh[1])] = 1
    binary_output = sxybinary
    return binary_output

def mag_thresh(img, sobel_kernel, mag_thresh):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    binary_output = sxbinary
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely),np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir>= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

# Edit this function to create your own pipeline.
def pipeline(img, r_thresh=(230, 255), l_thresh=(210,255),s_thresh=(120,255),b_lab_thresh=(220,255),sxy_thresh=(45, 100)):#170, 255

    #RGB
    r_rgb = img[:,:,0]
    r_binary = np.zeros_like(r_rgb)
    r_binary[(r_rgb >= r_thresh[0]) & (r_rgb <= r_thresh[1])] = 1
    
    # HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    #Lab
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_lab = lab[:,:,2]
    if np.max(b_lab) > 100:
        b_lab = b_lab*(255/np.max(b_lab))
    b_binary = np.zeros_like(b_lab)
    b_binary[((b_lab > b_lab_thresh[0]) & (b_lab <= b_lab_thresh[1]))] = 1
    
    '''
    # Sobel x
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))   
    #Sobel
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    '''
    
    sxbinary=abs_sobel_thresh(img,'x',sxy_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    
    # Stack each channel
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    color_binary = np.dstack((r_binary, sxbinary, l_binary)) * 255
   # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary==1)&(dir_binary==1)|(r_binary == 1)|(l_binary == 1)|(sxbinary == 1)] = 1#(b_binary == 1)
    return color_binary,combined_binary

'''
#测试-渐变和色彩空间,输出二值化图像(输入图像为已经过失真矫正，透视变换的图像)
color_result_test,combined_binary_test = pipeline(top_down_warped_test)
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(top_down_warped_test)
ax1.set_title('Original Image', fontsize=40)
ax2.imshow(combined_binary_test,cmap='gray')
ax2.set_title('Pipeline Result', fontsize=40)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('./writeup_images/image5.jpg')
'''