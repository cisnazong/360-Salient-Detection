# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:09:51 2020

@author: zongnan
"""

import cv2 
import os

def file_name(file_dir):   
    FileList=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.png':  
                FileList.append(os.path.join(file))  
    return root,FileList 

def img_2_xsize(inputimg,w,h):
    img = cv2.resize(inputimg,(w,h),
                interpolation=cv2.INTER_CUBIC)
    return img

if __name__ == "__main__":
    
    rootf,flist = file_name('f:/M0316/saved_0401/back/')
    rootout = ('f:/M0316/saved_0401/back_lowered/')
    
    size_w = int(2048/2)
    size_h = int(1024/2)
    
    for f in flist:
        print('processing image: '+f)
        img = cv2.imread(rootf+f)
        outimg = img_2_xsize(img,size_w,size_h)
        cv2.imwrite(rootout+f,outimg,[int(cv2.IMWRITE_PNG_COMPRESSION),9])