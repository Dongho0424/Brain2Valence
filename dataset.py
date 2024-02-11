import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset


class BrainValenceDataset(Dataset):
    def __init__(self,
                 data_path,
                 split,
                 emotic_annotations,
                 nsd_df,
                 target_cocoid,
                 subjects=[1, 2, 5, 7],
                 model_type="reg",
                 num_classif=3,
                 ):

        self.data_path = data_path
        self.split = split  # train, val, test
        self.subjects = subjects  # [1, 2, 5, 7]
        self.model_type = model_type
        self.num_classif = num_classif

        if split in ['train', 'val']:
            # firstly, concat boath train and val csv file corresponding to each subject
            dfs = [pd.read_csv(os.path.join(
                self.data_path, f'train_subj0{subj}_metadata.csv')) for subj in self.subjects]
            dfs += [pd.read_csv(os.path.join(self.data_path,
                                f'val_subj0{subj}_metadata.csv')) for subj in self.subjects]
            self.metadata = pd.concat(dfs)
            self.metadata.reset_index(inplace=True, drop=True)

            # then split the metadata into train and test with 80/20 split
            # randomly shuffle with fixed seed in order to get same splitted index whenever call this dataset.
            fixed_suffle_seed = 0
            self.metadata = self.metadata.sample(
                frac=1, random_state=fixed_suffle_seed).reset_index(drop=True)
            self.train_metadata = self.metadata.iloc[:int(len(self.metadata)*0.8)]
            self.val_metadata = self.metadata.iloc[int(len(self.metadata)*0.8):].reset_index(drop=True)

            if split == 'train':
                self.metadata = self.train_metadata
            elif split == 'val':
                self.metadata = self.val_metadata
            else:
                ValueError("split should be one of 'train', 'val', 'test'")
        elif split == 'test':
            dfs = [pd.read_csv(os.path.join(
                self.data_path, f'{self.split}_subj0{subj}_metadata.csv')) for subj in self.subjects]
            self.metadata = pd.concat(dfs)
            self.metadata.reset_index(inplace=True, drop=True)
        else:
            ValueError("split should be one of 'train', 'val', 'test'")

        # get joint data between NSD and EMOTIC and COCO
        self.nsd_df = nsd_df # given NSD dataset metadata file
        self.emotic_annotations = emotic_annotations 
        self.target_cocoid = target_cocoid

        # pre convert nsd data
        # appropriately matching exact id of COCO image given metadata.csv
        self.coco_id = self.nsd2coco()
        # use only joint and target(one person in picture) image
        # target_cocoid와 joint 한 것만 남긴다.
        isin = self.coco_id.isin(self.target_cocoid)
        self.coco_id = self.coco_id[isin]
        # metadata도 남길 애들만 남긴다.
        self.metadata = self.metadata[isin]

        self.set_weights()

    def set_weights(self):
        """
        Get weights for each class in classification task in order to use weighted random sampler.

        Goal
        -----
        1. Get valence from emotic_annotations corresponding to coco_id.
        2. Divide valence into intervals according to num_classif 
        3. Get weight of each interval.
        """
        
        # Get valence from emotic_annotations corresponding to coco_id.
        valence = [np.mean(self.emotic_annotations[coco_id]['valence']) for coco_id in self.coco_id]
        self.metadata['valence'] = valence

        # Divide valence into intervals according to num_classif
        bins = [0, 2, 4, 6, 8, 10] if self.num_classif == 5 else [0, 4, 7, 10]
        self.metadata['valence_interval'] = pd.cut(self.metadata['valence'], bins=bins, labels=False, include_lowest=True)
        
        # Get num classes of each interval.
        class_sample_counts = self.metadata['valence_interval'].value_counts().sort_index()

        # save weight of each interval, which is the invert of count per sample.
        class_weights = 1. / class_sample_counts
        self.metadata['weight'] = self.metadata['valence_interval'].apply(lambda x: class_weights[x])

    def get_weights(self):
        return self.metadata['weight']

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        sample = self.metadata.iloc[idx]
        split = self.get_split_info(sample['coco']) # 'coco' or 'voxel' or ... don't matter
        brain_3d = torch.from_numpy(np.load(os.path.join(self.data_path, split, sample['mri']))) # (3, *, *, *)
        brain_3d = torch.mean(brain_3d, dim=0) # (*, *, *)
        brain_3d = self.reshape_brain3d(brain_3d) # (96, 96, 96)
        
        # regression task: normalized valence
        # classification task: valence_interval with respect to num_classif
        valence = (sample['valence'] / 10.0) if self.model_type == 'reg' else sample['valence_interval']
        return brain_3d, valence


    def nsd2coco(self) -> pd.Series:
        # metadata.csv의 coco column은 nsd_id이므로, 이를 `nsd_stim_info_merged.csv` 를 읽어온 후,
        # nsdid => coco id 로 바꿔줌

        nsd_id = self.metadata['coco'].apply(lambda x: np.load(
            os.path.join(self.data_path, self.get_split_info(x), x))[-1])
        coco_id = []
        isin = nsd_id.isin(self.nsd_df['nsdId'])
        for i, x in nsd_id.items():
            if isin[i]:
                coco_id.append(self.nsd_df.loc[x, 'cocoId'])

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

        brain_3d = torch.nn.functional.pad(brain_3d, (shape_z_diff_1, shape_z_diff_2, shape_y_diff_1,
                                           shape_y_diff_2, shape_x_diff_1, shape_x_diff_2), mode='constant', value=0)

        return brain_3d

    # FIXME: 나이브하게 짠 것 바꾸면 좋겠다. 하지만 이건 후순위
    def get_split_info(self, x: str) -> str:
        # unfortunately build this naive way
        # may fix later
        if "train" in x:
            return "train"
        elif "val" in x:
            return "val"
        elif 'test' in x:
            return 'test'
        else:
            ValueError("get_split_info: couldn't get split info")

