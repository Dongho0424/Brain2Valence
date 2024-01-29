import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class NSDDataset(Dataset):
    def __init__(self, data_path, split, subj):
        self.data_path = data_path
        self.split = split
        self.subj = subj
        # Here assume that self.data_path = "/home/USERNAME/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split"
        self.metadata = pd.read_csv(os.path.join(self.data_path, f'{self.split}_subj0{self.subj}_metadata.csv'))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        img = Image.open(os.path.join(self.data_path, self.split, sample['img']))
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
         
        voxel = torch.from_numpy(np.load(os.path.join(self.data_path, self.split, sample['voxel'])))
        coco = torch.from_numpy(np.load(os.path.join(self.data_path, self.split, sample['coco'])))
        return voxel, img, coco
    
class NSDConcatDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        
        self.metadata = pd.concat([self.dataset1.metadata, self.dataset2.metadata])
        self.metadata.reset_index(inplace=True, drop=True)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]