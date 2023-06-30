from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import torch
import torchvision.transforms as transforms

class MyDataset(Dataset):
    def __init__(self,df,image_dir,mask_dir,cfg,aug):
        super().__init__()
        self.df = df
        self.cfg = cfg
        self.aug = aug
        self.image_dir = image_dir
        self.blood_mask = mask_dir

    __len__ = lambda self : len(self.df)
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        t = transforms.ToTensor()
        
        img_path = os.path.join(self.image_dir, f"{sample.id}.tif")
        image = cv2.imread(img_path)

        blood_mask_path = os.path.join(self.blood_mask,f"{sample.id}.png")
        blood_mask = cv2.imread(blood_mask_path,0)

        transformed = self.aug(image = image, blood_mask = blood_mask)
        image = transformed["image"]
        blood_mask = transformed["blood_mask"]

        data = {
            "image": t(image),
            "mask": t(blood_mask)
        }
        return data
    