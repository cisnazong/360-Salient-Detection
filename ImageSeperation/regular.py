# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:40:36 2020

@author: zongnan
"""

import os           
import cv2 

def file_name(file_dir):   
    FileList=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.png' or os.path.splitext(file)[1] == '.jpg':  
                FileList.append(os.path.join(file))  
    return root,FileList  

def regular(imgname,namenum,root):
    image = cv2.imread(imgname)
    if namenum+1<10 and namenum>=0:
        cv2.imwrite(root+'00'+str(namenum+1)+'.png',image,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
    elif namenum+1<100:
        cv2.imwrite(root+'0'+str(namenum+1)+'.png',image,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
    else:
        cv2.imwrite(root+str(namenum+1)+'.png',image,[int(cv2.IMWRITE_PNG_COMPRESSION),9])

if __name__ == "__main__":
    root,flist = file_name('f:/M0316/test/')
    newroot = 'f:/M0316/test_reg/'
    for i in range(len(flist)):
        regular(root+flist[i],i,newroot)