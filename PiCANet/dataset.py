import os
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = img.resize((self.size, self.size), resample=Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                             resample=Image.BILINEAR)
        return {'image': img, 'mask': mask}


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = img.resize((256, 256), resample=Image.BILINEAR), mask.resize((256, 256), resample=Image.BILINEAR)
        h, w = img.size
        new_h, new_w = self.size, self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img = img.crop((left, top, left + new_w, top + new_h))
        mask = mask.crop((left, top, left + new_w, top + new_h))

        return {'image': img, 'mask': mask}


class RandomFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self.flip = transforms.RandomHorizontalFlip(1.)

    def __call__(self, sample):
        if np.random.random_sample() < self.prob:
            img, mask = sample['image'], sample['mask']
            img = self.flip(img)
            mask = self.flip(mask)
            return {'image': img, 'mask': mask}
        else:
            return sample


class ToTensor(object):
    def __init__(self):
        self.tensor = transforms.ToTensor()

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = self.tensor(img), self.tensor(mask)
        return {'image': img, 'mask': mask}


class DUTSDataset(data.Dataset):
    def __init__(self, root_dir, train=True, data_augmentation=True):
        self.root_dir = root_dir
        self.train = train
        self.image_list = sorted(os.listdir('{}/DUTS-{}-Image'.format(root_dir, 'TR' if train else 'TE')))
        self.mask_list = sorted(os.listdir('{}/DUTS-{}-Mask'.format(root_dir, 'TR' if train else 'TE')))
        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(224),
             ToTensor()])
        if not (train and data_augmentation):
            self.transform = transforms.Compose([Resize(224), ToTensor()])
        self.root_dir = root_dir
        self.train = train
        self.data_augmentation = data_augmentation

    def arrange(self):
        flag = True
        if len(self.image_list) > len(self.mask_list):
            for image in self.image_list:
                for mask in self.mask_list:
                    if image.split("Image")[-1].split(".")[-2] == mask.split("Mask")[-1].split(".")[-2]:
                        print(image.split("Image")[-1].split(".")[-2])
                        flag = False
                if flag:
                    print(image + ' Deleted')
                    os.remove('{}/DUTS-{}-Image/{}'.format(self.root_dir, 'TR' if self.train else 'TE', image))
        else:
            for mask in self.mask_list:
                for image in self.image_list:
                    if image.split("Image")[-1].split(".")[-2] == mask.split("Mask")[-1].split(".")[-2]:
                        print(image.split("Image")[-1].split(".")[-2])
                        flag = False
                if flag:
                    print(mask + ' Deleted')
                    os.remove('{}/DUTS-{}-Mask/{}'.format(self.root_dir, 'TR' if self.train else 'TE', mask))
        self.image_list = sorted(os.listdir('{}/DUTS-{}-Image'.format(self.root_dir, 'TR' if self.train else 'TE')))
        self.mask_list = sorted(os.listdir('{}/DUTS-{}-Mask'.format(self.root_dir, 'TR' if self.train else 'TE')))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = '{}/DUTS-{}-Image/{}'.format(self.root_dir, 'TR' if self.train else 'TE', self.image_list[item])
        mask_name = '{}/DUTS-{}-Mask/{}'.format(self.root_dir, 'TR' if self.train else 'TE', self.mask_list[item])
        img = Image.open(img_name)
        mask = Image.open(mask_name)
        img = img.convert('RGB')
        mask = mask.convert('L')
        sample = {'image': img, 'mask': mask}

        sample = self.transform(sample)
        return sample


class PairDataset(data.Dataset):
    def __init__(self, root_dir, train=True, data_augmentation=True):
        self.root_dir = root_dir
        self.train = train
        self.image_list = sorted(os.listdir(os.path.join(root_dir, 'images')))
        self.mask_list = sorted(os.listdir(os.path.join(root_dir, 'masks')))
        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(224),
             ToTensor()])
        if not (train and data_augmentation):
            self.transform = transforms.Compose([Resize(224), ToTensor()])
        self.root_dir = root_dir
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = os.path.join(self.root_dir, 'images', self.image_list[item])
        mask_name = os.path.join(self.root_dir, 'masks', self.mask_list[item])
        img = Image.open(img_name)
        mask = Image.open(mask_name)
        img = img.convert('RGB')
        mask = mask.convert('L')
        sample = {'image': img, 'mask': mask}

        sample = self.transform(sample)
        return sample


class CustomDataset(data.Dataset):
    def __init__(self, root_dir):
        self.image_list = sorted(os.listdir(root_dir))
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = '{}/{}'.format(self.root_dir, self.image_list[item])
        img = Image.open(img_name)
        sample = img.convert('RGB')
        sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    ds = PairDataset('./dataset')
    print(ds)
    len(ds)
