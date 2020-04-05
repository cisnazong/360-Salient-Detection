# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:19:45 2020

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

def assemble(foreimg,backimg):
    a = foreimg.shape
    backimg = cv2.resize(backimg,(a[1],a[0]),
                       interpolation=cv2.INTER_CUBIC)
    foreimg = cv2.resize(foreimg,(a[1],a[0]),
                       interpolation=cv2.INTER_CUBIC)
    img = backimg + foreimg
#    for i in range(a[0]):
#        for j in range(a[1]):
#            if (img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0 
#                and foreimg[i][j][0] != 0 and foreimg[i][j][1] != 0 and foreimg[i][j][2] != 0):           
#                img[i][j][0] = foreimg[i][j][0] + backimg[i][j][0]
#                img[i][j][1] = foreimg[i][j][0] + backimg[i][j][0]
#                img[i][j][2] = foreimg[i][j][0] + backimg[i][j][0]
    return img

if __name__ == "__main__":
    rootf,flist = file_name('f:/M0316/saved_0401/fore/')
    rootb,blist = file_name('f:/M0316/saved_0401/back_lowered/')
    rootout = 'f:/M0316/saved_0401/assemble_lowered/'
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
#    print(f,b)
    for f in flist:
        for b in blist:
            if f[1:] == b[1:]:
                print('processing image: '+f+' and '+b)
                imgf = cv2.imread(rootf+f)
                imgb = cv2.imread(rootb+b)
                outimg = assemble(imgf,imgb)
               
                cv2.imwrite(rootout+'o'+f[1:],outimg,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
                print('Done!')
                print()