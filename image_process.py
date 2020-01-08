import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle

class Image_process_binary:
    def __init__(self,img,dist_pickle):
        """初始化数据"""
        self.img = img
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]

        self.orient = 'x'
        self.sobel_kernel = 3
        self.sxy_thresh = (40, 80)
        self.mag_thresh = (0, 255)#未使用
        self.dir_thresh=(0.1, 1.3) # dir_thresh -> (0, np.pi/2)
        self.r_thresh = (150, 255)
        self.g_thresh=(125, 255)
        self.h_thresh=(15, 120)
        self.l_thresh=(210, 255)
        self.s_thresh=(40, 255)
        self.l_lab_thresh=(140, 200)
        self.b_lab_thresh=(225, 255)

    def img_undistort(self):
        """图像失真矫正"""
        dst = cv2.undistort(self.img, self.mtx, self.dist, None, self.mtx)
        return dst
    
    def abs_sobel_thresh(self):
        """x或y方向的梯度大小二值化"""
        dst_img = self.img_undistort()
        gray_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2GRAY)
        if self.orient == 'x':
            sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel) # Take the derivative in x
        if self.orient == 'y':
            sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel) # Take the derivative in y
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        sxybinary = np.zeros_like(scaled_sobel)
        sxybinary[(scaled_sobel >= self.sxy_thresh[0]) & (scaled_sobel <= self.sxy_thresh[1])] = 1
        binary_output = sxybinary
        return binary_output

    def mag_thresh(self):
        """x，y综合梯度大小二值化"""
        dst_img = self.img_undistort()
        gray_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.mag_thresh[0]) & (scaled_sobel <= self.mag_thresh[1])] = 1
        binary_output = sxbinary
        return binary_output

    def dir_threshold(self):
        """梯度方向二值化"""
        dst_img = self.img_undistort()
        gray_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        absgraddir = np.arctan2(np.absolute(sobely),np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir>= self.dir_thresh[0]) & (absgraddir <= self.dir_thresh[1])] = 1
        return binary_output
    
    def rgb_threshold(self):
        """RGB颜色空间二值化"""
        dst_img = self.img_undistort()
        r_rgb = dst_img[:,:,0]
        r_binary = np.zeros_like(r_rgb)
        r_binary[(r_rgb >= self.r_thresh[0]) & (r_rgb <= self.r_thresh[1])] = 1

        g_rgb = dst_img[:,:,1]
        g_binary = np.zeros_like(g_rgb)
        g_binary[(g_rgb >= self.g_thresh[0]) & (g_rgb <= self.g_thresh[1])] = 1
        return r_binary,g_binary

    def hls_threshold(self):
        """HLS颜色空间二值化"""
        dst_img = self.img_undistort()
        hls = cv2.cvtColor(dst_img, cv2.COLOR_RGB2HLS)

        h_channel = hls[:,:,0]
        h_binary = np.zeros_like(h_channel)
        h_binary[(h_channel >= self.h_thresh[0]) & (h_channel <= self.h_thresh[1])] = 1

        l_channel = hls[:,:,1]
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= self.l_thresh[0]) & (l_channel <= self.l_thresh[1])] = 1
        
        s_channel = hls[:,:,2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 1
        return h_binary,l_binary,s_binary

    def lab_threshold(self):
        """Lab颜色空间二值化"""
        dst_img = self.img_undistort()
        lab = cv2.cvtColor(dst_img, cv2.COLOR_RGB2Lab)

        b_lab = lab[:,:,2]
        if np.max(b_lab) > 100:
            b_lab = b_lab*(255/np.max(b_lab))
        b_lab_binary = np.zeros_like(b_lab)
        b_lab_binary[((b_lab > self.b_lab_thresh[0]) & (b_lab <= self.b_lab_thresh[1]))] = 1
        
        l_lab = lab[:,:,0]
        l_lab_binary = np.zeros_like(l_lab)
        l_lab_binary[((l_lab > self.l_lab_thresh[0]) & (l_lab <= self.l_lab_thresh[1]))] = 1
        return b_lab_binary,l_lab_binary

    def pipeline(self):

        sxbinary = self.abs_sobel_thresh()
        dir_binary = self.dir_threshold()
        r_binary, g_binary = self.rgb_threshold()
        h_binary, l_binary, s_binary = self.hls_threshold()
        b_lab_binary, l_lab_binary = self.lab_threshold()
        
        # Stack each channel
        #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        color_binary = np.dstack((r_binary, sxbinary, l_binary)) * 255
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(l_lab_binary==1)&(b_lab_binary==1)] = 1#(l_lab_binary==1)&(b_lab_binary==1)  (g_binary==1)&(r_binary==1)  (s_binary==1)&(dir_binary==1)  |(sxbinary == 1)&(dir_binary==1)#(s_binary==1)&(dir_binary==1)|(r_binary == 1)|(l_binary == 1)|(sxbinary == 1)#(b_binary == 1)
        return color_binary,combined_binary
'''
class Image_process_line(Image_process_binary):
    def __init__(self):
        
        super().__init__(img,dist_pickle)
    
    #单帧图像中车道线检测

    def find_lane_pixels(binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        #plt.imshow(out_img)
        #plt.imshow(binary_warped,cmap='gray')
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 60
        # Set minimum number of pixels found to recenter window
        minpix = 40

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        #print(nonzerox.shape)
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin  # Update this
            win_xleft_high = leftx_current + margin # Update this
            win_xright_low = rightx_current - margin  # Update this
            win_xright_high = rightx_current + margin  # Update this
            #print('(',win_xleft_low,',',win_y_low,')',' (',win_xleft_high,',',win_y_high,')')
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            
            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) &(nonzeroy < win_y_high)
            & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            ### TO-DO: If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        if len(leftx) == 0:
            print('leftx:0')
        if len(lefty) == 0:
            print('lefty:0')
        return leftx, lefty, rightx, righty, out_img

    def fit_polynomial(binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

        ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
        
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        print(left_fit)
        print(right_fit)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty
        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

        return left_fit,right_fit,leftx,lefty,rightx,righty,left_fitx, right_fitx, ploty,out_img
'''