import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class PolypDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # 获取所有图像文件并过滤隐藏文件
        all_files = os.listdir(image_dir)
        self.images = [f for f in all_files if not f.startswith('.') and 
                      not os.path.isdir(os.path.join(image_dir, f)) and
                      '.ipynb_checkpoints' not in f]
        
        print(f"加载了 {len(self.images)} 个图像文件 (过滤掉 {len(all_files) - len(self.images)} 个隐藏文件/目录)")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            
            if self.transform is not None:
                image = self.transform(image)
                mask = self.transform(mask)
            
            # 确保掩码是二值的
            mask = (mask > 0.5).float()
            
            return image, mask
        except Exception as e:
            print(f"Error loading image {img_path} or mask {mask_path}: {e}")
            return None, None 