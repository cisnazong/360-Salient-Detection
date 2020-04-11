from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class SalObjDataset360(Dataset):
    def __init(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = list (sorted (os.listdir (os.path.join (root, "images"))))
        self.masks = list (sorted (os.listdir (os.path.join (root, "masks"))))

    def __len__(self):
        return len (self.imgs)

    def __getitem__(self,idx):
        img_path = os.path.join (self.root, "images", self.imgs[idx])
        mask_path = os.path.join (self.root, "masks", self.masks[idx])

        img = Image.open (img_path).convert ("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)

        sample = {'image':img, 'mask':mask}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

