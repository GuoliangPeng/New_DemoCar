'''
输入：车辆前置摄像头原始视频文件
输出：车辆当前位置与车道线中心位置的偏差实时值的视频文件
注：
Lines_num = 1   表示道路只有一条中心车道线，通过寻中间车道线计算车辆偏离路中心偏差
Lines_num = 2   表示道路只有左右两侧边缘线，通过两侧两条线计算车辆偏离路中心偏差
'''
import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import image_process

'''
###测试一：渐变和色彩空间,输出二值化图像(输入图像为原始图像)###
#test_img = mpimg.imread('./20191231test/2.chewei7ma.jpg')
test_img = mpimg.imread('./20191231test/1.chenei15ma_imgs/1.jpg')
dist_pickle = pickle.load( open( "./Demo_camera_cal/wide_dist_pickle.p", "rb" ) )
warper_pickle = pickle.load( open( "./Demo_camera_cal/img_warper_pickle.p", "rb" ) )


a = image_process.Image_process_binary(test_img,dist_pickle,warper_pickle)
color_result_test,combined_binary_test = a.pipeline()
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_img)
ax1.set_title('Original Image', fontsize=40)
ax2.imshow(combined_binary_test,cmap='gray')
ax2.set_title('Pipeline Result', fontsize=40)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('./writeup_images/1.chenei15ma_imgs_1..png')
plt.show() 
'''
'''
###测试二：单帧图像中车道线检测###
# Load our image
#test_img = mpimg.imread('./20191231test/2.chewei7ma.jpg')
test_img = mpimg.imread('./20191231test/1.chenei15ma_imgs/1.jpg')
dist_pickle = pickle.load( open( "./Demo_camera_cal/wide_dist_pickle.p", "rb" ) )
warper_pickle = pickle.load( open( "./Demo_camera_cal/img_warper_pickle.p", "rb" ) )
a = image_process.Image_process_binary(test_img,dist_pickle,warper_pickle)
color_result_test,combined_binary_test = a.pipeline()
Lines_num = 1
binary_warped_test = combined_binary_test
b = image_process.Image_process_line(binary_warped_test,Lines_num,warper_pickle)
if Lines_num==1:
    center_fit,warped_centerx, warped_centery,center_fitx,ploty, result = b.fit_polynomial()
    print(center_fit)
elif Lines_num==2:
    left_fit,right_fit,warped_leftx, warped_lefty, warped_rightx, warped_righty,left_fitx, right_fitx, ploty, result = fit_polynomial(binary_warped_test)
    print(left_fit)
    print(right_fit)    
plt.imshow(result)
plt.savefig('./writeup_images/image6.png')
plt.show()
'''

'''
###测试三：连续帧图像中车道线检测###
# Load our image
test_img = mpimg.imread('./20191231test/1.chenei15ma_imgs/1.jpg')
dist_pickle = pickle.load( open( "./Demo_camera_cal/wide_dist_pickle.p", "rb" ) )
warper_pickle = pickle.load( open( "./Demo_camera_cal/img_warper_pickle.p", "rb" ) )
a = image_process.Image_process_binary(test_img,dist_pickle,warper_pickle)
color_result_test,combined_binary_test = a.pipeline()
Lines_num = 1
binary_warped_test = combined_binary_test
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_img)
ax1.set_title('Original Image', fontsize=40)
ax2.imshow(combined_binary_test ,cmap='gray')
ax2.set_title('Pipeline Result', fontsize=40)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('./writeup_images/image9.png')
plt.show()
b = image_process.Image_process_line(binary_warped_test,Lines_num,warper_pickle)
if Lines_num==1:
    center_fit,warped_centerx, warped_centery,center_fitx,ploty, result = b.fit_polynomial()
    print(center_fit)
elif Lines_num==2:
    left_fit,right_fit,warped_leftx, warped_lefty, warped_rightx, warped_righty,left_fitx, right_fitx, ploty, result = b.fit_polynomial()
    print(left_fit)
    print(right_fit)
plt.imshow(result)
plt.savefig('./writeup_images/image10.png')
plt.show() 
# Load our image
test_img = mpimg.imread('./20191231test/1.chenei15ma_imgs/2.jpg')
#dist_pickle = pickle.load( open( "./Demo_camera_cal/wide_dist_pickle.p", "rb" ) )
a.img= test_img
color_result_test,combined_binary_test = a.pipeline()
Lines_num = 1
binary_warped_test = combined_binary_test
b.binary_warped = binary_warped_test
if Lines_num==1:
    centerx,centery,center_fitx,ploty, result = b.search_around_poly()
    #print(center_fit)
#elif Lines_num==2:
    #left_fit,right_fit,warped_leftx, warped_lefty, warped_rightx, warped_righty,left_fitx, right_fitx, ploty, result = fit_polynomial(binary_warped_test)
    #print(left_fit)
    #print(right_fit)    
plt.imshow(result)
plt.savefig('./writeup_images/image11.png')
plt.show() 


center_curverad, distance_from_center= image_process.measure_curvature_real_center(binary_warped_test,ploty,center_fitx)
print("曲率为：", center_curverad,"m")
print("中心偏差为：", distance_from_center,"m")

Minv = warper_pickle["Minv"]
result2 = image_process.drawing_center(test_img, binary_warped_test,Minv, center_fitx, ploty)
plt.imshow(result2)
plt.savefig('./writeup_images/image12.png')
plt.show() 

result3 = image_process.putimg(result2, center_curverad, distance_from_center)
plt.imshow(result3)
plt.savefig('./writeup_images/image13.png')
plt.show() 
'''

###原始视频输入检测
video_input = './20191231test/1.chenei15ma.mp4'
video_output = './20191231test/result_1.chenei15ma.mp4'

cap = cv2.VideoCapture(video_input)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_output, fourcc, 20.0, (1928, 1208))

detected = False
i=0
Lines_num=1

dist_pickle = pickle.load( open( "./Demo_camera_cal/wide_dist_pickle.p", "rb" ) )
warper_pickle = pickle.load( open( "./Demo_camera_cal/img_warper_pickle.p", "rb" ) )


ret, image = cap.read()
if ret:
    image_binary = image_process.Image_process_binary(image,dist_pickle,warper_pickle)
    image_line = image_process.Image_process_line(image_binary,Lines_num)
    print("初始化成功")
else:
    print("初始化错误")

while(True):
    ret, image = cap.read()
    if ret:
        i=i+1
        print(i)
        image_binary.img = image
        dst_img,color_result_img,combined_binary_img = image_binary.pipeline()

        if detected == False:
            image_line.binary_warped = combined_binary_img
            center_fit,centerx,centery,center_fitx,ploty,out_img = image_line.fit_polynomial()
            print('k')
        else:
            image_line.binary_warped = combined_binary_img
            center_fit,centerx,centery,center_fitx,ploty,out_img = image_line.search_around_poly()
            print('l')
        
        if (len(center_fit) > 0):
            detected = True
        else :
            detected = False
        center_curverad, distance_from_center = image_process.measure_curvature_real_center(out_img,ploty,center_fitx)
        line_img = image_process.drawing_center(dst_img, combined_binary_img, warper_pickle, center_fitx, ploty)
        output_result = image_process.putimg(line_img, center_curverad, distance_from_center)
        out.write(output_result)
    else:
        break
        
cap.release()
out.release()
cv2.destroyAllWindows()