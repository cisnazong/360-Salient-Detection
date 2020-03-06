import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os
from scipy import misc
from datetime import datetime

from utils.data import test_dataset
from model.ResNet_models import SCRN

model = SCRN()
model.load_state_dict(torch.load('./model/model.pth'))
model.cuda()
model.eval()

data_path = 'C:/Users/RAZER/Desktop/SCRN-master/F-360iSOD/'
# valset = ['ECSSD', 'HKUIS', 'PASCAL', 'DUT-OMRON', 'THUR15K', 'DUTS-TEST']
valset = ['SOD']
for dataset in valset:
    save_path = './saliency_maps/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = 'C:/Users/RAZER/Desktop/SCRN-master/F-360iSOD/stimulis/'
    gt_root = 'C:/Users/RAZER/Desktop/SCRN-master/F-360iSOD/objects/'
    test_loader = test_dataset(image_root, gt_root, testsize=352)

    with torch.no_grad():
        print(test_loader.size)
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.array(gt).astype('float')
            gt = gt / (gt.max() + 1e-8)
            image = Variable(image).cuda()
            
            res, edge = model(image)

            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=True)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            misc.imsave(save_path + name + '.png', res)

