# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:50:08 2020

@author: zongnan
"""

import os
import numpy as np

def get_FileSize(filePath):
 
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024 * 1024)
 
    return round(fsize, 2)


def file_name(file_dir):   
    FileList=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.png' or os.path.splitext(file)[1] == '.jpg':  
                FileList.append(os.path.join(file))  
    return root,FileList 

 
if __name__ == '__main__':
    root1,flist1 = file_name('f:/M0316/saved_0401/origin/')
    root2,flist2 = file_name('f:/M0316/saved_0401/fore_standard/')
    root3,flist3 = file_name('f:/M0316/saved_0401/back_lowered/')
    
    sizelist1 = []
    sizelist2 = []
    sizelist3 = []
    percentage = []
    
    print
    
    for f in flist1:
        size = get_FileSize(root1+f)
        #print("origin文件 "+f+" 的大小：%.2f MB"%(size))
        sizelist1.append(size)
    for ff in flist2:
        size = get_FileSize(root2+ff)
        #print("fore_standard文件 "+ff+" 的大小：%.2f MB"%(size))
        sizelist2.append(size)
    for fff in flist3:
        size = get_FileSize(root3+fff)
        #print("back_lowered文件 "+fff+" 的大小：%.2f MB"%(size))
        sizelist3.append(size)
        
        
    for i in range(len(sizelist1)):
        perc = float(sizelist2[i]+sizelist3[i])/sizelist1[i]
        percentage.append(perc)
        #print(flist1[i]+'处理后的对比原图百分比： ',perc)
        print(perc)
    meanpercentage = np.mean(percentage)
    print('Terminated.')
    #print("平均百分比: ",meanpercentage)