# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:08:38 2020

@author: zongnan
"""
  
import os           
import numpy as np
import cv2 

def seperate(imagetest,image1,size_w=0,size_h=0,resize_mode=0,show_mode=1,dilation_ratio=0,area_threshold=3000): #input image, mask, resize mode, show mode, dilatio_ratio
    
    sh = imagetest.shape
    
    ww = sh[1]
    hh = sh[0]
    
    a = np.array([ww,hh])
    
    if size_h > 0 and size_w > 0:
        a[1] = size_h
        a[0] = size_w
    
    print ('Image size should be: width: ',a[0],' height: ',a[1])
    
    if resize_mode != 1:
        p1=cv2.resize(image1,(a[0],a[1]),
                       interpolation=cv2.INTER_CUBIC)
        imagetest = cv2.resize(imagetest,(a[0],a[1]),
                       interpolation=cv2.INTER_CUBIC)
        origin = imagetest.copy()
        
    if resize_mode == 1:   
        p1=cv2.resize(image1,(a[0],a[1]),
                       interpolation=cv2.INTER_AREA)
        imagetest = cv2.resize(imagetest,(a[0],a[1]),
                       interpolation=cv2.INTER_AREA)
        origin = imagetest.copy()
        
    np1 = p1
    
    gray1=cv2.cvtColor(p1,cv2.COLOR_BGR2GRAY)
    
    for i in range(a[1]):
        for j in range(a[0]):
            if gray1[i][j] < 20:
                np1[i][j][0] = 0
                np1[i][j][1] = 0
                np1[i][j][2] = 0
            else:
                np1[i][j][0] = 255
                np1[i][j][1] = 255
                np1[i][j][2] = 255
                
    
    #OpenCV定义的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    
    #腐蚀图像
    #eroded1 = cv2.erode(np1,kernel)
    #显示腐蚀后的图像
    #cv2.imshow("Eroded Image",eroded1);
    
    #膨胀图像
    if dilation_ratio <= 0:
        dil_coeff = 10
    else:
        dil_coeff = dilation_ratio
    
    dil_num=int(origin.shape[0]/dil_coeff)
    
    for i in range(dil_num):
        dilated1 = cv2.dilate(np1,kernel)        
       

    #边缘检测
    gray = cv2.cvtColor(dilated1.copy(), cv2.COLOR_BGR2GRAY)
    thresh = gray
    print ("Calculating contours...")
    
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print ("contours 数量：",len(contours))
#    for i in range(len(contours)):
#        print ('contours['+str(i)+']点的个数：',len(contours[i]))
    
    Rect_x = [] #位置
    Rect_area = [] #面积
    
    for cnt in contours:
    	# 源轮廓
        cv2.drawContours(imagetest, [cnt], -1, (0, 255, 0), 2)
        
    	# 近似多边形, 轮廓近似,是Douglas-Peucker算法的一种实现方式
    	# epsilon 为近似度参数，该值需要轮廓的周长信息
    	# 多边形周长与源轮廓周长之比就是epsilon
        epsilon = 0.01 * cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(imagetest, [approx], -1, (255, 255, 0), 2)
     
    	# 凸包
    #	hull = cv2.convexHull(cnt)
    #	cv2.drawContours(origin, [hull], -1, (0, 0, 255), 2)
     
    	#直边外接矩形
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
    	# 这里可以限制外接矩形大小，如果太小就不画出来。
        if w*h>area_threshold:
            print('x,y,w,h:', x, ',', y,',', w, ',', h)
            Rect_x.append((x,y,w,h))
            Rect_area.append(area)
            cv2.rectangle(imagetest,(x,y),(x+w,y+h),(0,255,0),2)
     
    	# 最小外接矩形
    #	rect = cv2.minAreaRect(cnt)
    #	box = cv2.boxPoints(rect)
    #	box = np.int0(box)
    #	print(box)
    #	cv2.drawContours(origin,[box],0,(0,0,255),2)
    	# cv2.imwrite('./test_rect.png', img)
     
    	# 拟合直线,这个还没有试过
    	# rows,cols = img.shape[:2]
    	# [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    	# lefty = int((-x*vy/vx) + y)
    	# righty = int(((cols-x)*vy/vx)+y)
    	# cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
    print ("Contours done!")
    ################################后景####################################
    print('Dividing back & fore ground...')
    backimg = origin.copy()
    for i in range(len(Rect_x)):
        for xx in range(Rect_x[i][0],Rect_x[i][0]+Rect_x[i][2]):
            for yy in range(Rect_x[i][1],Rect_x[i][1]+Rect_x[i][3]):
                backimg[yy][xx][0] = 0
                backimg[yy][xx][1] = 0
                backimg[yy][xx][2] = 0
    
    ################################前景####################################
    foreimglist = []
    for i in range(len(Rect_x)):
        x = Rect_x[i][0]
        y = Rect_x[i][1]
        w = Rect_x[i][2]
        h = Rect_x[i][3]
    
    ############单独取前景############### 
    #    blank_image = np.zeros((h,w,3),np.uint8)
        
    ############带位置取前景#############
        blank_image = np.zeros((a[1],a[0],3),np.uint8)
    #    print(x,y,w,h)
        for xx in range(x,x+w):
            for yy in range(y,y+h):
                ############单独取前景###############
    #            blank_image[yy-y][xx-x][0] = imagetest[yy][xx][0]
    #            blank_image[yy-y][xx-x][1] = imagetest[yy][xx][1]
    #            blank_image[yy-y][xx-x][2] = imagetest[yy][xx][2]
                
                ############带位置取前景#############
                blank_image[yy][xx][0] = origin[yy][xx][0]
                blank_image[yy][xx][1] = origin[yy][xx][1]
                blank_image[yy][xx][2] = origin[yy][xx][2]
        foreimglist.append(blank_image)
        
    ############单独显示#################  
              
    #for i in range(len(foreimg)):
    #    cv2.imshow('fore images',foreimg[i])
    #    cv2.waitKey(0)
        
    ############叠加显示#################     
    foreimg = foreimglist[0] 
    if len(foreimglist) <= 1:
        foreimg = foreimglist[0]
    else:
        for i in range(1,len(foreimglist)):
            foreimg = foreimg + foreimglist[i]  
    
    
    #cv2.imshow('image1_3次插值法',dilated1)
    #cv2.waitKey(0)
    print('Done!')
    
    if show_mode == 1:
        cv2.imshow("contour_image",imagetest)
        cv2.waitKey(0)
        
        cv2.imshow("background_image",backimg)
        cv2.waitKey(0)
        
        cv2.imshow("foreground_image",foreimg)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        
    return foreimglist,foreimg,backimg,imagetest
 
  
def file_name(file_dir):   
    FileList=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.png':  
                FileList.append(os.path.join(file))  
    return root,FileList  


if __name__ == "__main__":
    
    size_h = 1024
    size_w = 2048
    resize_mode = 0
    show_mode = 0
    dilation_ratio = 5
    area_threshold = 1000
    save_mode = 1
    
    rootf,flist = file_name('f:/M0316/saved_0401/img/')
    rootm,mlist = file_name('f:/M0316/saved_0401/mask/')
    
    print(rootf,rootm)
    print(flist,mlist)
    print()
    
    for i in range(len(flist)):
        
        print('processing image name: '+flist[i]+ ' at number: '+ str(i+1)+'/'+str(len(flist))+' : ')
        
        input_image = cv2.imread(rootf+flist[i])
        mask = cv2.imread(rootm+mlist[i])
   
        forelist,fore,back,cntimg = seperate(input_image, mask, size_w, size_h, resize_mode, show_mode, dilation_ratio, area_threshold)
        #input_image, mask, resize_mode, show_mode, dilation_ratio, area_threshold of rects
        if save_mode == 1:
            print('Saving...')
            cv2.imwrite('./saved_0401/fore/f'+flist[i],fore,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
            cv2.imwrite('./saved_0401/back/b'+flist[i],back,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
            cv2.imwrite('./saved_0401/contour/c'+flist[i],cntimg,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
            print('Save Done!')
    