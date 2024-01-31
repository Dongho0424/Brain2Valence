import scipy
import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset

class BrainValenceDataset(Dataset):
    def __init__(self,  data_path, split, emotic_annotations, nsd_df, target_cocoid, subjects=[1, 2, 5, 7]):
        self.data_path = data_path
        self.split = split # train, val, test
        self.subjects = subjects # [1, 2, 5, 7]
        if split in ['train', 'val']:
            dfs = [pd.read_csv(os.path.join(self.data_path, f'{self.split}_subj0{subj}_metadata.csv')) for subj in self.subjects]
            self.metadata = pd.concat(dfs)
        elif split == 'test':
            # As test data is all same to every subjects,
            df = pd.read_csv(os.path.join(self.data_path, 'test_subj01_metadata.csv'))
            self.metadata = df
        else: IndexError("Wrong split type")
        
        self.metadata.reset_index(inplace=True, drop=True)       

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
        brain_3d = np.mean(brain_3d, axis=0) # (*, *, *)
        brain_3d = self.reshape_brain3d(brain_3d, target_shape=(96, 96, 96)) # (96, 96, 96)
        
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
    
    def reshape_brain3d(brain_3d, target_shape=(96, 96, 96)):
        # Initialize pad_width
        pad_width = [(0, 0)] * 3  # For 3D brain_3day

        # Calculate padding needed for each dimension
        for i in range(3):
            current_size = brain_3d.shape[i]
            if current_size < target_shape[i]:
                # Calculate padding
                total_pad = target_shape[i] - current_size
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pad_width[i] = (pad_before, pad_after)

        # Apply padding
        brain_3d = np.pad(brain_3d, pad_width=pad_width, mode='constant', constant_values=0)

        # Apply truncation if necessary
        brain_3d = brain_3d[:target_shape[0], :target_shape[1], :target_shape[2]]

        return brain_3d