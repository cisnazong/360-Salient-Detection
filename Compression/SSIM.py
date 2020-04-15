# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:55:54 2020

@author: zongnan
"""

import pytorch_ssim
import torch
from torch.autograd import Variable
import os           
import numpy as np
import cv2 

def file_name(file_dir):   
    FileList=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.png':  
                FileList.append(os.path.join(file))  
    return root,FileList  



if __name__ == "__main__":
    
    root1,flist1 = file_name('f:/M0316/saved_0401/origin/')
    root2,flist2 = file_name('f:/M0316/saved_0401/assemble_lowered/')#('f:/M0316/test_reg/')
    compare = 1
    
    res = []
    
    for i in range(len(flist2)):
        
        if compare == 1:
            print('calculating ssim of image1: '+flist1[i]+ ' & image 2 : '+ flist2[i])
                    
            imgi1 = cv2.imread(root1+flist1[i])
            imgi2 = cv2.imread(root2+flist2[i])
        else:
            print('calculating ssim of image1: '+flist1[0]+ ' & image 2 : '+ flist2[i])
                    
            imgi1 = cv2.imread(root1+flist1[0])
            imgi2 = cv2.imread(root2+flist2[i])
            
        standard_prop = 200
        
#        print("Shapes")
#        print(imgi1.shape)
#        print(imgi2.shape)
        
        standard_height = imgi1.shape[0]
        standard_width = imgi1.shape[1]
        
        ########################## RESIZE TO SAME ##############################
        if imgi1.shape != imgi2.shape:
            imgi1 = cv2.resize(imgi1,(standard_width,standard_height),interpolation=cv2.INTER_NEAREST)
            imgi2 = cv2.resize(imgi2,(standard_width,standard_height),interpolation=cv2.INTER_NEAREST)
        
        img1_cv = cv2.cvtColor(imgi1, cv2.COLOR_BGR2RGB)
        img2_cv = cv2.cvtColor(imgi2, cv2.COLOR_BGR2RGB)
        
        img1 = Variable(torch.from_numpy(np.expand_dims(np.transpose(img1_cv, (2, 0, 1)),0)))
        img2 = Variable(torch.from_numpy(np.expand_dims(np.transpose(img2_cv, (2, 0, 1)),0)))
        
        ########################## TO TENSOR FLOAT##############################
        img1=img1.float()
        img2=img2.float()
        
        #################### MS-SSIM PARAMETER: WINDOW SIZE ####################
        h = standard_height
        w = standard_width
        proportion = standard_prop * standard_prop
        default_window_size = 11
    #    img1 = Variable(torch.rand(1, 1, 256, 256))
    #    img2 = Variable(torch.rand(1, 1, 256, 256))
    #    
        
        if torch.cuda.is_available():
            img1 = img1.cuda()
            img2 = img2.cuda()
        
        #print('SSIM : ', pytorch_ssim.ssim(img1, img2))
        
        ssim_loss = pytorch_ssim.SSIM(window_size = int(h*w/proportion))
        
        #print('MS_SSIM : ', ssim_loss(img1, img2))
        
        res.append(ssim_loss(img1, img2))
    
    rl = []
    for r in res:
        rl.append(r.cpu().numpy())
        
    mean_res = np.mean(rl)
    min_res = np.min(rl)
    max_res = np.max(rl)
    max_ind = rl.index(max_res)
    min_ind = rl.index(min_res)
    
    print('Max SSIM: ',max_res)
    print('Max SSIM index: ',max_ind)
    print('Min SSIM: ',min_res)
    print('Min SSIM index: ',min_ind)
    print('Average SSIM: ',mean_res)
