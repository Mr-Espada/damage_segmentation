import torch
import numpy as np
import os
import cv2

class BuildingsDataset(torch.utils.data.Dataset):

    """Massachusetts Buildings Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            paths, 
            nclasses=2, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.image_paths = []
        for path in paths:
            self.image_paths.extend([ f'geotiffs/{path}/images512/{file}' for file in os.listdir(f'geotiffs/{path}/images512')  if file.endswith('.png') and 'post' in file])
        
        self.nclasses = nclasses
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.image_paths[i].replace('images512', 'masks512').replace('.png' ,'_mask-nodamage.png'), 0)
        mask2 = cv2.imread(self.image_paths[i].replace('images512', 'masks512').replace('.png' ,'_mask-minordamage.png'), 0)
        mask3 = cv2.imread(self.image_paths[i].replace('images512', 'masks512').replace('.png' ,'_mask-majordamage.png'), 0)
        mask4 = cv2.imread(self.image_paths[i].replace('images512', 'masks512').replace('.png' ,'_mask-destoryed.png'), 0)

        mask[mask[:,:]>0] = 1
        mask2[mask2[:,:]>0] = 1
        mask3[mask3[:,:]>0] = 1
        mask4[mask4[:,:]>0] = 1

        mask = np.where((mask == 0) & (mask2 == 1), 2, mask)
        mask = np.where((mask == 0) & (mask3 == 1), 3, mask)
        mask = np.where((mask == 0) & (mask4 == 1), 4, mask)

        # one-hot-encode the mask
        #mask = one_hot(torch.tensor(mask).to(torch.int64), num_classes=self.nclasses).numpy()
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], torch.as_tensor(sample['mask'], dtype=torch.int64).unsqueeze(0).numpy()
            
            
        return image, mask
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)