import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle

class Image_process_binary:
    def __init__(self,img,dist_pickle,warper_pickle):
        """初始化数据"""
        self.img = img #原始图像
        self.dst_img = np.zeros((self.img.shape[0],self.img.shape[1],self.img.shape[2])) #私有化属性
        self.warper_img = np.zeros((self.img.shape[0],self.img.shape[1],self.img.shape[2])) #私有化属性
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]
        self.M = warper_pickle["M"]

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
        self.l_lab_thresh=(150, 200)
        self.b_lab_thresh=(225, 255)

    def img_undistort(self):
        """图像失真矫正"""
        dst = cv2.undistort(self.img, self.mtx, self.dist, None, self.mtx)
        return dst
    
    def img_warper(self):
        """图像透视变换"""
        img_size = (self.dst_img.shape[1], self.dst_img.shape[0])
        warped = cv2.warpPerspective(self.dst_img, self.M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
        return warped

    def abs_sobel_thresh(self):
        """x或y方向的梯度大小二值化"""
        gray_img = cv2.cvtColor(self.warper_img, cv2.COLOR_RGB2GRAY)
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
        gray_img = cv2.cvtColor(self.warper_img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.mag_thresh[0]) & (scaled_sobel <= self.mag_thresh[1])] = 1
        binary_output = sxbinary
        return binary_output,warper_pickle

    def dir_threshold(self):
        """梯度方向二值化"""
        gray_img = cv2.cvtColor(self.warper_img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        absgraddir = np.arctan2(np.absolute(sobely),np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir>= self.dir_thresh[0]) & (absgraddir <= self.dir_thresh[1])] = 1
        return binary_output
    
    def rgb_threshold(self):
        """RGB颜色空间二值化"""
        r_rgb = self.warper_img[:,:,0]
        r_binary = np.zeros_like(r_rgb)
        r_binary[(r_rgb >= self.r_thresh[0]) & (r_rgb <= self.r_thresh[1])] = 1

        g_rgb = self.warper_img[:,:,1]
        g_binary = np.zeros_like(g_rgb)
        g_binary[(g_rgb >= self.g_thresh[0]) & (g_rgb <= self.g_thresh[1])] = 1
        return r_binary,g_binary

    def hls_threshold(self):
        """HLS颜色空间二值化"""
        hls = cv2.cvtColor(self.warper_img, cv2.COLOR_RGB2HLS)

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
        lab = cv2.cvtColor(self.warper_img, cv2.COLOR_RGB2Lab)

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
        self.dst_img = self.img_undistort() # self.dst_img更新
        #self.dst_img = self.img
        self.warper_img = self.img_warper() #self.warper_img更新
        
        sxbinary = self.abs_sobel_thresh()
        dir_binary = self.dir_threshold()
        r_binary, g_binary = self.rgb_threshold()
        h_binary, l_binary, s_binary = self.hls_threshold()
        b_lab_binary, l_lab_binary = self.lab_threshold()
        
        # Stack each channelbin_warped
        #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        color_binary = np.dstack((r_binary, sxbinary, l_binary)) * 255
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(l_lab_binary==1)&(b_lab_binary==1)] = 1#(l_lab_binary==1)&(b_lab_binary==1)  (g_binary==1)&(r_binary==1)  (s_binary==1)&(dir_binary==1)  |(sxbinary == 1)&(dir_binary==1)#(s_binary==1)&(dir_binary==1)|(r_binary == 1)|(l_binary == 1)|(sxbinary == 1)#(b_binary == 1)
        return self.dst_img,color_binary,combined_binary

class Image_process_line:
    def __init__(self,binary_warped_img,Lines_num):
        self.binary_warped = binary_warped_img
        #self.Minv = warper_pickle["Minv"]
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        self.nwindows = 9
        # Set the width of the windows +/- self.margin
        self.margin = 80
        # Set minimum number of pixels found to recenter window
        self.minpix = 40
        #self.Lines_num = 1   表示道路只有一条中心车道线，通过寻中间车道线计算车辆偏离路中心偏差
        #self.Lines_num = 2   表示道路只有左右两侧边缘线，通过两侧两条线计算车辆偏离路中心偏差
        self.Lines_num = Lines_num
        if self.Lines_num == 1:
            self.center_fit = [] #私有属性，拟合的二次曲线中心线方程三个参数
            self.last_centerx = [] #上次拟合二次曲线中心线所需的x坐标
            self.last_centery = [] #上次拟合二次曲线中心线所需的y坐标
        elif self.Lines_num == 2:
            self.left_fit = [] #私有属性，拟合的二次曲线左侧线方程三个参数
            self.right_fit = [] #私有属性，拟合的二次曲线右侧线方程三个参数
            #self.leftx = [] #拟合的二次曲线左侧线的x坐标
            #self.lefty = [] #拟合的二次曲线左侧线的y坐标
            #self.rightx = [] #拟合的二次曲线右侧线的x坐标
            #self.righty = [] #拟合的二次曲线右侧线的y坐标
    
    def find_lane_pixels(self):
        """'''单帧图像中车道线像素点检测"""
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.binary_warped[self.binary_warped.shape[0]//3:,:], axis=0)#self.binary_warped.shape[0]//2
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))*255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        if self.Lines_num == 1:
            centerx_base = np.argmax(histogram[:])
            # Current positions to be updated later for each window in self.nwindows
            centerx_current = centerx_base
            # Create empty lists to receive left and right lane pixel indices
            center_lane_inds = []
        elif self.Lines_num == 2:
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            # Current positions to be updated later for each window in self.nwindows
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []
        
        #plt.imshow(out_img)
        #plt.imshow(binary_warped,cmap='gray')

        # Set height of windows - based on self.nwindows above and image shape
        window_height = np.int(self.binary_warped.shape[0]//self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        #print(nonzerox.shape)

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.binary_warped.shape[0] - (window+1)*window_height
            win_y_high = self.binary_warped.shape[0] - window*window_height
            ### Find the four below boundaries of the window ###
            if self.Lines_num == 1:
                win_xcenter_low = centerx_current - self.margin  # Update this
                win_xcenter_high = centerx_current + self.margin  # Update this
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xcenter_low,win_y_low),(win_xcenter_high,win_y_high),(0,255,0), 2)
                ### Identify the nonzero pixels in x and y within the window ###
                good_center_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xcenter_low) &  (nonzerox < win_xcenter_high)).nonzero()[0]
                # Append these indices to the lists
                center_lane_inds.append(good_center_inds)
                if len(good_center_inds) > self.minpix:
                    centerx_current = np.int(np.mean(nonzerox[good_center_inds]))
            elif self.Lines_num == 2:
                win_xleft_low = leftx_current - self.margin  # Update this
                win_xleft_high = leftx_current + self.margin # Update this
                win_xright_low = rightx_current - self.margin  # Update this
                win_xright_high = rightx_current + self.margin  # Update this
                #print('(',win_xleft_low,',',win_y_low,')',' (',win_xleft_high,',',win_y_high,')')
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
                ### Identify the nonzero pixels in x and y within the window ###
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) &(nonzeroy < win_y_high) & 
                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]              
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)                
                ### If you found > self.minpix pixels, recenter next window ###
                ### (`right` or `leftx_current`) on their mean position ###
                if len(good_left_inds) > self.minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > self.minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        if self.Lines_num == 1:
            # Concatenate the arrays of indices (previously was a list of lists of pixels)
            try:
                center_lane_inds = np.concatenate(center_lane_inds)
            except ValueError:
                # Avoids an error if the above is not implemented fully
                pass
            # Extract left and right line pixel positions
            centerx = nonzerox[center_lane_inds]
            centery = nonzeroy[center_lane_inds]
            if len(centerx) == 0:
                print('centerx:0')
            if len(centery) == 0:
                print('centery:0')
            return centerx, centery, out_img        
        elif self.Lines_num == 2:
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

    def fit_polynomial(self):
        """单帧图像上车道线拟合"""
        if self.Lines_num == 1:
            # Find our lane pixels first
            centerx, centery, out_img = self.find_lane_pixels()
            ### Fit a second order polynomial to each using `np.polyfit` ###      
            self.center_fit = np.polyfit(centery, centerx, 2)
            print(self.center_fit)
            # Generate x and y values for plotting
            ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0])
            try:
                center_fitx = self.center_fit[0]*ploty**2 + self.center_fit[1]*ploty + self.center_fit[2]
            except TypeError:
                # Avoids an error if `left` and `right_fit` are still none or incorrect
                print('The function failed to fit a line!')
                center_fitx = 1*ploty**2 + 1*ploty
            ## Visualization ##
            # Colors in the left and right lane regions
            out_img[centery, centerx] = [255, 0, 0]
            # Plots the left and right polynomials on the lane lines
            plt.plot(center_fitx, ploty, color='yellow')
            return self.center_fit,centerx,centery,center_fitx,ploty,out_img
        elif self.Lines_num == 2:
            # Find our lane pixels first
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels()
            ### Fit a second order polynomial to each using `np.polyfit` ###      
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
            print(self.left_fit)
            print(self.right_fit)
            # Generate x and y values for plotting
            ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0])
            try:
                left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
                right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
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
            return self.left_fit,self.right_fit,leftx,lefty,rightx,righty,left_fitx, right_fitx, ploty,out_img

    def search_around_poly(self):
        """连续帧图像上车道线检测拟合"""
        # HYPERPARAMETER
        # Choose the width of the self.margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!

        # Grab activated pixels
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Generate x and y values for plotting
        ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0])
        
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- self.margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        if self.Lines_num == 1:
            center_lane_inds = ((nonzerox > (self.center_fit[0]*(nonzeroy**2) + self.center_fit[1]*nonzeroy + self.center_fit[2] - self.margin))
                & (nonzerox < (self.center_fit[0]*(nonzeroy**2) + self.center_fit[1]*nonzeroy + self.center_fit[2] + self.margin)))
            # Again, extract left and right line pixel positions
            # 新二值化图像帧上符合条件的点的x，y坐标的列表
            centerx = nonzerox[center_lane_inds]
            centery = nonzeroy[center_lane_inds]
            if len(centerx)>30:
                self.last_centerx = centerx
                self.last_centery = centery
            else:
                centerx = self.last_centerx
                centery = self.last_centery
            print(len(centerx))
            ### Fit a second order polynomial to each with np.polyfit() ###
            ### 曲线拟合，参数更新
            self.center_fit = np.polyfit(centery,centerx,2)
            ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
            ### 拟合后的曲线的x坐标列表
            center_fitx = self.center_fit[0]*(ploty**2) + self.center_fit[1]*ploty + self.center_fit[2]
            ## Visualization ##
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))*255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[center_lane_inds], nonzerox[center_lane_inds]] = [255, 0, 0]
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            center_line_window1 = np.array([np.transpose(np.vstack([center_fitx-self.margin, ploty]))])
            center_line_window2 = np.array([np.flipud(np.transpose(np.vstack([center_fitx+self.margin, ploty])))])
            center_line_pts = np.hstack((center_line_window1, center_line_window2))
            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([center_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            # Plot the polynomial lines onto the image
            plt.plot(center_fitx, ploty, color='yellow')
            ## End visualization steps ## 
            return self.center_fit,centerx,centery,center_fitx,ploty, result
        elif self.Lines_num == 2:            
            left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - self.margin))
                & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + self.margin)))
            right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - self.margin))
                & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + self.margin))) 
            # Again, extract left and right line pixel positions
            # 新二值化图像帧上符合条件的点的x，y坐标的列表
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]                      
            ### Fit a second order polynomial to each with np.polyfit() ###
            ### 曲线拟合，参数更新
            self.left_fit = np.polyfit(lefty,leftx,2)
            self.right_fit = np.polyfit(righty,rightx,2)
            ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
            ### 拟合后的曲线的x坐标列表
            left_fitx = self.left_fit[0]*(ploty**2) + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*(ploty**2) + self.right_fit[1]*ploty + self.right_fit[2]     
            ## Visualization ##
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))*255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))
            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            # Plot the polynomial lines onto the image
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            ## End visualization steps ## 
            return self.left_fit,self.right_fit,leftx,lefty,rightx,righty,left_fitx, right_fitx, ploty, result
    
def measure_curvature_real_center(img,ploty,center_fitx):
    """计算偏差和曲率（中心车道线）"""
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 6.5/1450 # meters per pixel in x dimension
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    center_fit_cr = np.polyfit(ploty*ym_per_pix, center_fitx*xm_per_pix, 2)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    #ploty = np.linspace(0, binary_warped_img.shape[0]-1, binary_warped_img.shape[0])
    y_eval = np.max(ploty)   
    #####Implement the calculation of R_curve (radius of curvature) #####
    center_curverad = (1+(2*center_fit_cr[0]*y_eval*ym_per_pix+center_fit_cr[1])**2)**1.5/np.absolute(2*center_fit_cr[0])  
    ## Implement the calculation of the left line here
    #left_down_realx = left_fit_cr[0]*((y_eval*ym_per_pix)**2)+left_fit_cr[1]*(y_eval*ym_per_pix)+left_fit_cr[2]
    #right_down_realx =right_fit_cr[0]*((y_eval*ym_per_pix)**2)+right_fit_cr[1]*(y_eval*ym_per_pix)+right_fit_cr[2]
    #distance_from_center = (right_down_realx - left_down_realx)/2 - (binary_warped_img.shape[1]/2)*xm_per_pix
    lane_xm_per_pix = xm_per_pix
    line_pos = center_fitx[img.shape[0]-1] * lane_xm_per_pix
    veh_pos = ((img.shape[1] * lane_xm_per_pix) / 2.)
    distance_from_center = veh_pos - line_pos
    return center_curverad, distance_from_center

def measure_curvature_real_double(img,ploty,left_fitx,right_fitx):
    """计算偏差和曲率(左右车道线)"""
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    #left_fit_cr = np.polyfit(warped_lefty*ym_per_pix, warped_leftx*xm_per_pix, 2)
    #right_fit_cr = np.polyfit(warped_righty*ym_per_pix, warped_rightx*xm_per_pix, 2)
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    #ploty = np.linspace(0, binary_warped_img.shape[0]-1, binary_warped_img.shape[0])
    y_eval = np.max(ploty)   
    #####Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = (1+(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**1.5/np.absolute(2*left_fit_cr[0])  ## Implement the calculation of the left line here
    right_curverad = (1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**1.5/np.absolute(2*right_fit_cr[0])  ## Implement the calculation of the right line here
    curve_direction = (left_curverad+right_curverad)/2.0
    #left_down_realx = left_fit_cr[0]*((y_eval*ym_per_pix)**2)+left_fit_cr[1]*(y_eval*ym_per_pix)+left_fit_cr[2]
    #right_down_realx =right_fit_cr[0]*((y_eval*ym_per_pix)**2)+right_fit_cr[1]*(y_eval*ym_per_pix)+right_fit_cr[2]
    #distance_from_center = (right_down_realx - left_down_realx)/2 - (binary_warped_img.shape[1]/2)*xm_per_pix
    lane_width = np.absolute(left_fitx[-1] - right_fitx[-1])
    lane_xm_per_pix=3.7/lane_width
    veh_pos = (((left_fitx[719] + right_fitx[719]) * lane_xm_per_pix) / 2.)
    cen_pos = ((img.shape[1] * lane_xm_per_pix) / 2.)
    distance_from_center = veh_pos - cen_pos
    return left_curverad, right_curverad, curve_direction, distance_from_center

def drawing_center(undist, bin_warped,warper_pickle,center_fitx, ploty):
    """将检测到的车道边界扭曲回失真矫正图像（中心车道线）"""
    Minv = warper_pickle["Minv"]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(bin_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_center1 = np.array([np.transpose(np.vstack([center_fitx-20, ploty]))])
    pts_center2 = np.array([np.flipud(np.transpose(np.vstack([center_fitx+20, ploty])))])
    pts = np.hstack((pts_center1, pts_center2))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    line_img = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return line_img

def drawing_double(undist, bin_warped,Minv, left_fitx, right_fitx, ploty):
    """将检测到的车道边界扭曲回失真矫正图像（左右车道线）"""
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(bin_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    line_img = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return line_img

def putimg(line_img, curvature, offset):
    """视频上显示实时文字(车道线曲率和当前测偏距离)"""
    info_cur="Curvature: {:6.3f} m".format(curvature)
    #info_offset = "Off center: {0} {1:3.3f}m".format(direction, offset)
    info_offset = "Off center: {:3.3f}m".format(offset)
    cv2.putText(line_img,info_cur,(100,50),cv2.FONT_HERSHEY_PLAIN,3.0,(0,0,255),3)
    cv2.putText(line_img,info_offset,(100,150),cv2.FONT_HERSHEY_PLAIN,3.0,(0,0,255),3)
    result = line_img
    return result