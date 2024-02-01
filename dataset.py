import scipy
import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset

# TODO: *** sub별로 학습하고 eval 한 거 따로 해보기. ***
# TODO: train val 합쳐서 8:2로 나누기.
# - suffle해서 나누기
# - suffle seed 고정

class BrainValenceDataset(Dataset):
    def __init__(self,  data_path, split, emotic_annotations, nsd_df, target_cocoid, subjects=[1, 2, 5, 7]):
        self.data_path = data_path
        self.split = split # train, val, test
        self.subjects = subjects # [1, 2, 5, 7]
        
        dfs = [pd.read_csv(os.path.join(self.data_path, f'{self.split}_subj0{subj}_metadata.csv')) for subj in self.subjects]
        self.metadata = pd.concat(dfs)
        self.metadata.reset_index(inplace=True, drop=True)       

        np.random.seed(0) # split seed
        np.random.shuffle(self.metadata)
        # 8:2로 나눠서 하기!

        ## get joint data between NSD and EMOTIC and COCO
        self.nsd_df = nsd_df
        self.emotic_annotations = emotic_annotations
        self.target_cocoid = target_cocoid

        # pre convert nsd data appropriately
        self.coco_id = self.nsd2coco()

        ## use only joint and target(one person in picture) image
        # target_cocoid와 joint 한 것만 남긴다.
        isin = self.coco_id.isin(self.target_cocoid)
        self.coco_id = self.coco_id[isin]
        # metadata도 남길 애들만 남긴다.
        self.metadata = self.metadata[isin]
        

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        sample = self.metadata.iloc[idx]
        id = self.coco_id.iloc[idx]
        brain_3d = torch.from_numpy(np.load(os.path.join(self.data_path, self.split, sample['mri']))) # (3, *, *, *)
        brain_3d = torch.mean(brain_3d, dim=0) # (*, *, *)
        brain_3d = self.reshape_brain3d(brain_3d) # (96, 96, 96)
        
        valence = np.mean(self.emotic_annotations[id]['valence'])

        return brain_3d, valence

    def nsd2coco(self) -> pd.Series :
        # metadata.csv의 coco column은 nsd_id이므로, 이를 `nsd_stim_info_merged.csv` 를 읽어온 후, 
        # nsdid => coco id 로 바꿔줌

        nsd_id = self.metadata['coco'].apply(lambda x: np.load(self.data_path+f"/{self.split}/"+x)[-1])
        coco_id = []
        isin = nsd_id.isin(self.nsd_df['nsdId'])
        for i, x in nsd_id.items():
            if isin[i]: coco_id.append(self.nsd_df.loc[x, 'cocoId'])

        coco_id = pd.Series(coco_id)

        return coco_id
    
    def reshape_brain3d(self, brain_3d: torch.Tensor):
        # brain_3d: (*, *, *)
        # return: (96, 96, 96)

        shape_x_diff = 96 - brain_3d.shape[0]
        shape_y_diff = 96 - brain_3d.shape[1]
        shape_z_diff = 96 - brain_3d.shape[2]

        shape_x_diff_1 = shape_x_diff // 2
        shape_x_diff_2 = shape_x_diff - shape_x_diff_1
        shape_y_diff_1 = shape_y_diff // 2
        shape_y_diff_2 = shape_y_diff - shape_y_diff_1
        shape_z_diff_1 = shape_z_diff // 2
        shape_z_diff_2 = shape_z_diff - shape_z_diff_1

        brain_3d = torch.nn.functional.pad(brain_3d, (shape_z_diff_1, shape_z_diff_2, shape_y_diff_1, shape_y_diff_2, shape_x_diff_1, shape_x_diff_2), mode='constant', value=0)

        return brain_3d