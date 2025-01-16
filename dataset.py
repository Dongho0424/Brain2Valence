import torch
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import utils
import h5py
from ast import literal_eval
import re

class BrainValenceDataset(Dataset):
    def __init__(self,
                 data_path,
                 split,
                 emotic_annotations,
                 nsd_df,
                 target_cocoid,
                 subjects=[1, 2, 5, 7],
                 task_type="reg",
                 num_classif=3,
                 data: str = 'brain3d',
                 use_sampler: bool = False,
                 use_body: bool = False,
                 transform=None,
                 ):

        self.data_path = data_path
        self.split = split  # train, val, test
        self.subjects = subjects  # [1, 2, 5, 7]
        self.task_type = task_type # ['reg', 'classif', 'img2vad']
        self.num_classif = num_classif
        self.data = data
        self.use_sampler = use_sampler
        self.use_body = use_body
        self.transform = transform

        if split in ['train', 'val']:
            # firstly, concat boath train and val csv file corresponding to each subject
            dfs = [pd.read_csv(os.path.join(
                self.data_path, f'train_subj0{subj}_metadata.csv')) for subj in self.subjects]
            dfs += [pd.read_csv(os.path.join(self.data_path,
                                f'val_subj0{subj}_metadata.csv')) for subj in self.subjects]
            self.metadata = pd.concat(dfs)
            self.metadata.reset_index(inplace=True, drop=True)

            # then split the metadata into train and test with 9:1 split
            # randomly shuffle with fixed seed in order to get same splitted index whenever call this dataset.
            fixed_suffle_seed = 0
            self.train_metadata, self.val_metadata = train_test_split(self.metadata, test_size=0.1, random_state=fixed_suffle_seed)

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
        self.nsd_df = nsd_df  # given NSD dataset metadata file
        self.emotic_annotations = emotic_annotations
        self.target_cocoid = target_cocoid

        self.get_cocoid()

        # get joint data between NSD and EMOTIC and COCO
        # by using 'coco_id' column of metadata
        isin = self.metadata['coco_id'].isin(self.target_cocoid)
        self.metadata = self.metadata[isin]

        # add bbox, VAD to metadata
        self.set_annotations()

        # Divide vad into intervals according to num_classif
        if self.task_type == 'classif':
            self.devide_vad()

        if self.use_sampler:
            self.set_weights()

    def set_weights(self):
        """
        Get weights for each class in classification task in order to use weighted random sampler.

        Goal
        -----
        - Divide valence into intervals according to num_classif 
        """

        # Get num classes of each interval.
        class_sample_counts = self.metadata['valence_interval'].value_counts(
        ).sort_index()
        # print(class_sample_counts)

        # save weight of each interval, which is the invert of count per sample.
        class_weights = 1. / class_sample_counts
        self.metadata['weight'] = self.metadata['valence_interval'].apply(lambda x: class_weights[x])

    def get_weights(self):
        assert self.use_sampler, "You should set use_sampler to True in order to use this method"

        return self.metadata['weight']

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        repeat_index = idx % 3

        sample = self.metadata.iloc[idx]
        # 'img' or 'voxel' or ... don't matter
        split = sample['img'].split('_')[0]

        data = None

        if self.data == 'brain3d':
            brain_3d = torch.from_numpy(np.load(os.path.join(
                self.data_path, split, sample['mri'])))  # (3, *, *, *)
            # brain_3d = torch.mean(brain_3d, dim=0) # (*, *, *)
            brain_3d = brain_3d[repeat_index]
            brain_3d = self.reshape_brain3d(brain_3d)  # (96, 96, 96)

            data = brain_3d
        elif self.data == 'roi':
            if len(self.subjects) > 1:
                raise ValueError("Only one subject's roi data is available")
            roi = torch.from_numpy(np.load(os.path.join(
                self.data_path, split, sample['voxel'])))  # (3, *)
            # roi = torch.mean(roi, dim=0) # (*, )
            roi = roi[repeat_index]

            data = roi
            
        elif self.data == 'emo_vis_roi' or self.data == 'emo_roi':
            if len(self.subjects) > 1:
                raise ValueError("Only one subject's roi data is available")
            
            roi_path = f"/home/data/nsd_aws/nsddata/ppdata/subj0{self.subjects[0]}/func1pt8mm/roi"
            hcp_mmp_roi = nib.load(os.path.join(roi_path, 'HCP_MMP1.nii.gz')).get_fdata()
            nsdgeneral_roi = nib.load(os.path.join(roi_path, 'nsdgeneral.nii.gz')).get_fdata()
            
            emotion_related_roi_idx = [104, 106, 109, 111, 112, 126, 127, 155, 167,168, 178]
            
            nsdgeneral_mask = (nsdgeneral_roi == 1)
            emotion_related_mask = np.isin(hcp_mmp_roi, emotion_related_roi_idx)
            
            if self.data == 'emo_vis_roi':
                roi = nsdgeneral_mask | emotion_related_mask
            elif self.data == 'emo_roi':
                roi = emotion_related_mask
            
            brain_3d = torch.from_numpy(np.load(os.path.join(
                self.data_path, split, sample['mri'])))  # (3, *, *, *)
            # brain_3d = torch.mean(brain_3d, dim=0) # (*, *, *)
            brain_3d = brain_3d[repeat_index]
            data = brain_3d[roi].flatten()
            

        # regression task: normalized valence
        # classification task: valence_interval with respect to num_classif
        valence = (sample['valence'] / 10.0) if self.task_type in ['reg', 'img2vad'] else sample['valence_interval']
        arousal = (sample['arousal'] / 10.0) if self.task_type in ['reg', 'img2vad'] else sample['arousal_interval']
        dominance = (sample['dominance'] / 10.0) if self.task_type in ['reg', 'img2vad'] else sample['dominance_interval']

        coco_path = '/home/dongho/brain2valence/data'
        orig_img = sample['orig_img']
        orig_image = Image.open(os.path.join(coco_path, orig_img.split('_')[1], orig_img))
        
        # crop body from image
        # using bbox
        if self.use_body:
            bbox = sample['bbox']
            image = orig_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        # use transform
        if self.transform is not None:
            image = self.transform(image)

        return data, image, valence, arousal, dominance

    def get_cocoid(self) -> pd.Series:
        """
        As the data of 'coco' column of `self.metadata` is nsd_id,
        1. Rename it with 'nsd_id'
        2. From `self.nsd_df`, add corresponding 'coco_id' data to `self.metadata`
        """

        # rename
        self.metadata = self.metadata.rename(columns={'coco': 'nsd_id'})

        # get nsd_id from numpy data    
        nsd_id = self.metadata['nsd_id'].apply(lambda x: np.load(os.path.join(self.data_path, x))[-1])

        # get corresponding coco_id from nsd_df
        coco_id = nsd_id.apply(lambda x: self.nsd_df.loc[x, 'cocoId'])

        # add new column to metadata, which is 'coco_id'
        self.metadata['coco_id'] = coco_id

    def set_annotations(self):
        """
        - Make (brain3d, image) be matched with bbox and VAD
            - As we use data regardless of the number of people in image,
            - particular (brain3d, image) may be repeated but corresponding (bbox, VAD) is unique.
        - Also total the number of dataset is increased.
        """
        # Add a new column, original image address as 'orig_img'
        temp_dict = dict([(e['coco_id'], e['filename']) for e in self.emotic_annotations])

        self.metadata['orig_img'] = self.metadata['coco_id'].apply(lambda x: temp_dict[x])

        # Create a new DataFrame to store the squeezed data
        new_columns = list(self.metadata.columns) + ['bbox', 'valence', 'arousal', 'dominance']
        new_metadata = pd.DataFrame(columns=new_columns)

        for idx, row in self.metadata.iterrows():

            coco_id = row['coco_id']
            annot = [e['people'] for e in self.emotic_annotations if e['coco_id'] == coco_id][0]
            # As different number of people is in the image
            # Repeat the same row as the number of people in the image
            for a in annot:
                new_row = row.copy()
                new_row['bbox'] = a['bbox']
                new_row['valence'] = a['valence']
                new_row['arousal'] = a['arousal']
                new_row['dominance'] = a['dominance']
                new_row = pd.DataFrame(new_row).T  # for concatenating
                new_metadata = pd.concat([new_metadata, new_row], ignore_index=True)

        self.metadata = new_metadata.reset_index(drop=True)

    def devide_vad(self):
        bins = []
        if self.num_classif == 3:
            bins = [0, 4, 7, 10]
        elif self.num_classif == 5:
            bins = [0, 2, 4, 6, 8, 10]
        elif self.num_classif == 10:
            bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        else:
            ValueError("num_classif should be one of 3, 5, 10")
        self.metadata['valence_interval'] = pd.cut(self.metadata['valence'], bins=bins, labels=False, include_lowest=True)
        self.metadata['arousal_interval'] = pd.cut(self.metadata['arousal'], bins=bins, labels=False, include_lowest=True)
        self.metadata['dominance_interval'] = pd.cut(self.metadata['dominance'], bins=bins, labels=False, include_lowest=True)

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

class EmoticDataset(Dataset):
    def __init__(self,
                 data_path,
                 split,
                 emotic_annotations: pd.DataFrame,
                 context_transform=None,
                 body_transform=None,
                 normalize=False,
                 dataset_ver=2, # default 2
                 exclude_least=False, # w/o 1, 17, 22
                 exclude_low=False,   # w/o 1, 4, 6, 10, 15, 17, 20, 22
                 exclude_strategy=1,  # 1: delete entire row, 2: delete corrsponding categories only
                 cluster=False, # cluster categories into 4 groups, angry, happy, neutral, sad
                 ):

        self.data_path = data_path
        self.split = split  # train, val, test
        self.cluster = cluster
        self.metadata = self.set_metadata(emotic_annotations, exclude_least, exclude_low, exclude_strategy, cluster)
        self.context_transform = context_transform
        self.body_transform = body_transform
        self.normalize = normalize
        self.dataset_ver = dataset_ver

    def set_metadata(self, metadata: pd.DataFrame, exclude_least, exclude_low, exclude_strategy, cluster):

        if not exclude_least and not exclude_low and not cluster: # just default setting
            # eusure 'category' column is a list of integer
            metadata['category'] = metadata['category'].apply(lambda x: [int(i) for i in literal_eval(x)])
            return metadata

        if exclude_least and exclude_low:
            raise ValueError("You should set either exclude_least or exclude_low to True")

        self.is_exclude = exclude_least or exclude_low

        def contains_category(x, minor_category):
            x_int = [int(i) for i in literal_eval(x)]
            return any(cat in minor_category for cat in x_int)
        def filter_and_remove_categories(x, minor_category):
            x_int = [int(i) for i in literal_eval(x)]
            filtered = [_x for _x in x_int if _x not in minor_category]
            if len(filtered) == 0:
                return np.nan
            return filtered

        if self.is_exclude:
            # - 1. Anger: 26 
            # - 17. Pain: 28 
            # - 22. Suffering: 35
            least_category = [1, 17, 22]
            # - 4. Aversion: 51
            # - 6. Disapproval: 83
            # - 10. Embarrassment: 56
            # - 15. Fear: 75
            # - 20. Sadness: 52
            low_category = least_category + [4, 6, 10, 15, 20]
            self.minor_category = least_category if exclude_least else low_category

            print("# Excluding minor categories #")
            print(f"w/o categories: {self.minor_category}")
            print(f"exclude strategy: {exclude_strategy}")

            # strategy 1: delete entire row if any of the category is in minor_category
            if exclude_strategy == 1:
                if exclude_least:
                    metadata = metadata[~metadata['category'].apply(lambda x: contains_category(x, least_category))]
                elif exclude_low:
                    metadata = metadata[~metadata['category'].apply(lambda x: contains_category(x, low_category))]
                # eusure 'category' column is a list of integer
                metadata['category'] = metadata['category'].apply(lambda x: [int(i) for i in literal_eval(x)])
            
            # strategy 2: delete only corresponding category in a 'category' column. If nothing left, delete row itself
            elif exclude_strategy == 2:
                if exclude_least:
                    metadata['category'] = metadata['category'].apply(lambda x: filter_and_remove_categories(x, least_category))
                elif exclude_low:
                    metadata['category'] = metadata['category'].apply(lambda x: filter_and_remove_categories(x, low_category))
                # If nothing left, drop the row
                metadata = metadata.dropna(subset=['category'])
            else: 
                raise ValueError("exclude_strategy should be either 1 or 2")

        elif cluster:
            print("# Clustering into groups #")
            # Angry
            self.angry_idx = [1, 2, 4, 6]
            # Happy
            self.happy_idx = [0, 5, 12, 13, 16, 18, 19, 24]
            # Neutral
            self.neutral_idx = [3, 7, 9, 11, 14, 23]
            # Sad
            self.sad_idx = [8, 10, 15, 17, 20, 21, 22, 25]

            metadata['category'] = metadata['category'].apply(lambda x: [int(i) for i in literal_eval(x)])
            metadata['category'] = metadata['category'].apply(
                lambda x: [0 if i in self.angry_idx else 1 if i in self.happy_idx else 2 if i in self.neutral_idx else 3 for i in x])
            metadata['category'] = metadata['category'].apply(lambda x: list(set(x))) # remove duplicates
        else:
            raise ValueError("You should set either exclude_least or exclude_low to True")

        return metadata

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):

        sample = self.metadata.iloc[idx]
        
        context_image = Image.open(os.path.join(self.data_path, sample['folder'], sample['filename']))
        
        # crop body from image
        # using bbox
        if self.dataset_ver == 2:
            bbox = literal_eval(sample['bbox'])
        else:
            bbox = sample['bbox']
        body_image = context_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        # use transform
        if self.context_transform is not None:
            context_image = context_image.convert("RGB")
            context_image = self.context_transform(context_image)
        if self.body_transform is not None:
            body_image = body_image.convert("RGB")
            body_image = self.body_transform(body_image)

        # get VAD
        valence = sample['valence'] / 10.0 if self.normalize else sample['valence']
        arousal = sample['arousal'] / 10.0 if self.normalize else sample['arousal']
        dominance = sample['dominance'] / 10.0 if self.normalize else sample['dominance']

        # get category label torch.tensor
        cat_label_temp = np.zeros(26)
        for cat in sample['category']:
            cat_label_temp[cat] = 1

        if self.is_exclude:
            # Remove the elements whose index is in minor_category
            cat_label = np.array([item for idx, item in enumerate(cat_label_temp) if idx not in self.minor_category])
            # Ensure this removing must not change the true category 
            # because we already filtered out the minor categories in metadata
            assert np.sum(cat_label_temp) == np.sum(cat_label), "Minor categories are not properly removed"
            assert len(cat_label) == 26 - len(self.minor_category), "Minor categories are not properly removed"
        elif self.cluster:
            cat_label = np.zeros(4) # angry, happy, neutral, sad
            for cat in sample['category']:
                if cat in self.angry_idx:     cat_label[0] = 1
                elif cat in self.happy_idx:   cat_label[1] = 1
                elif cat in self.neutral_idx: cat_label[2] = 1
                elif cat in self.sad_idx:     cat_label[3] = 1
        else:
            cat_label = cat_label_temp
        cat_label = torch.from_numpy(cat_label)

        return context_image, body_image, valence, arousal, dominance, cat_label

class BrainDataset2(Dataset):
    def __init__(self,
                 subjects,
                 split,
                 data_type='roi',
                 pool_num=2048,
                 context_transform=None,
                 body_transform=None,
                 normalize=False,
                 exclude_least=False, # w/o 1, 17, 22
                 exclude_low=False,   # w/o 1, 4, 6, 10, 15, 17, 20, 22
                 exclude_strategy=1,  # 1: delete entire row, 2: delete corrsponding categories only
                 cluster=False, # cluster categories into 4 groups, angry, happy, neutral, sad
                 ):
        self.subjects = subjects
        self.split = split
        
        print("### Initializing BrainDataset v2 ###")
        print("Emotic Split: ", split)
        print("Subjects: ", subjects)
        print("Pool Num: ", pool_num)
        self.metadata = self.set_metadata(exclude_least, exclude_low, exclude_strategy, cluster)
        self.cluster = cluster
            
        self.context_transform = context_transform
        self.body_transform = body_transform
        self.pool_num = pool_num
        self.normalize = normalize
        self.data_type = data_type
        assert data_type in ['roi', 'emo_roi', 'emo_vis_roi'], "data_type should be either 'roi', 'emo_roi', or 'emo_vis_roi'"
        self.coco_data_path = "/home/dongho/brain2valence/data/emotic"

        self.num_voxels = {}
        self.voxel_means = {}
        self.voxel_stds = {}
        self.voxel_paths = {}
        basedir = '/home/dongho/brain2valence/data'

        for s in self.subjects:
            if data_type == 'roi':
                subdir = os.path.join(basedir, f'vis')
                mean = np.load(os.path.join(subdir, f'vis_subj{s}_train_beta_mean.npy'))
                std = np.load(os.path.join(subdir, f'vis_subj{s}_train_beta_std.npy'))
                self.voxel_paths[f'subj{s}'] = os.path.join(subdir, f'vis_subj{s}_all_beta')
            elif data_type == 'emo_roi':
                subdir = os.path.join(basedir, f'emo')
                mean = np.load(os.path.join(subdir, f'emo_subj{s}_train_beta_mean.npy'))
                std = np.load(os.path.join(subdir, f'emo_subj{s}_train_beta_std.npy'))
                self.voxel_paths[f'subj{s}'] = os.path.join(subdir, f'emo_subj{s}_all_beta')
            elif data_type == 'emo_vis_roi':
                subdir = os.path.join(basedir, f'emo_vis')
                mean = np.load(os.path.join(subdir, f'emo_vis_subj{s}_train_beta_mean.npy'))
                std = np.load(os.path.join(subdir, f'emo_vis_subj{s}_train_beta_std.npy'))
                self.voxel_paths[f'subj{s}'] = os.path.join(subdir, f'emo_vis_subj{s}_all_beta')
                
            self.num_voxels[f'subj{s}'] = mean.shape[0]
            self.voxel_means[f'subj{s}'] = mean
            self.voxel_stds[f'subj{s}'] = std

    def set_metadata(self, exclude_least, exclude_low, exclude_strategy, cluster):
        metadata = pd.read_csv('/home/dongho/brain2valence/emotic_nsd_joint_metadata_split.csv', dtype={'subject': str})
        metadata = metadata[metadata['emotic_split'] == self.split]
        # if subj=1, then ['1', 'all_1']
        # if subj=1 or 2, then ['1', '2', 'all_1', 'all_2']
        metadata = metadata[metadata['subject'].isin([f"{s}" for s in self.subjects] + [f"all_{s}" for s in self.subjects])] 
        metadata.reset_index(inplace=True, drop=True)

        if not exclude_least and not exclude_low and not cluster: # just default setting
            # eusure 'category' column is a list of integer
            metadata['category'] = metadata['category'].apply(lambda x: [int(i) for i in literal_eval(x)])
            return metadata

        if exclude_least and exclude_low:
            raise ValueError("You should set either exclude_least or exclude_low to True")

        self.is_exclude = exclude_least or exclude_low

        def contains_category(x, minor_category):
            x_int = [int(i) for i in literal_eval(x)]
            return any(cat in minor_category for cat in x_int)
        def filter_and_remove_categories(x, minor_category):
            x_int = [int(i) for i in literal_eval(x)]
            filtered = [_x for _x in x_int if _x not in minor_category]
            if len(filtered) == 0:
                return np.nan
            return filtered

        if self.is_exclude:
            # - 1. Anger: 26 
            # - 17. Pain: 28 
            # - 22. Suffering: 35
            least_category = [1, 17, 22]
            # - 4. Aversion: 51
            # - 6. Disapproval: 83
            # - 10. Embarrassment: 56
            # - 15. Fear: 75
            # - 20. Sadness: 52
            low_category = least_category + [4, 6, 10, 15, 20]
            self.minor_category = least_category if exclude_least else low_category

            print("# Excluding minor categories #")
            print(f"w/o categories: {self.minor_category}")
            print(f"exclude strategy: {exclude_strategy}")

            # strategy 1: delete entire row if any of the category is in minor_category
            if exclude_strategy == 1:
                if exclude_least:
                    metadata = metadata[~metadata['category'].apply(lambda x: contains_category(x, least_category))]
                elif exclude_low:
                    metadata = metadata[~metadata['category'].apply(lambda x: contains_category(x, low_category))]
                # eusure 'category' column is a list of integer
                metadata['category'] = metadata['category'].apply(lambda x: [int(i) for i in literal_eval(x)])
            
            # strategy 2: delete only corresponding category in a 'category' column. If nothing left, delete row itself
            elif exclude_strategy == 2:
                if exclude_least:
                    metadata['category'] = metadata['category'].apply(lambda x: filter_and_remove_categories(x, least_category))
                elif exclude_low:
                    metadata['category'] = metadata['category'].apply(lambda x: filter_and_remove_categories(x, low_category))
                # If nothing left, drop the row
                metadata = metadata.dropna(subset=['category'])
            else: 
                raise ValueError("exclude_strategy should be either 1 or 2")

        elif cluster:
            print("# Clustering into groups #")
            # Angry
            self.angry_idx = [1, 2, 4, 6]
            # Happy
            self.happy_idx = [0, 5, 12, 13, 16, 18, 19, 24]
            # Neutral
            self.neutral_idx = [3, 7, 9, 11, 14, 23]
            # Sad
            self.sad_idx = [8, 10, 15, 17, 20, 21, 22, 25]

            metadata['category'] = metadata['category'].apply(lambda x: [int(i) for i in literal_eval(x)])
            metadata['category'] = metadata['category'].apply(
                lambda x: [0 if i in self.angry_idx else 1 if i in self.happy_idx else 2 if i in self.neutral_idx else 3 for i in x])
            metadata['category'] = metadata['category'].apply(lambda x: list(set(x))) # remove duplicates
        else:
            raise ValueError("You should set either exclude_least or exclude_low to True")
        
        return metadata
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        context_image = Image.open(os.path.join(self.coco_data_path, sample['folder'], sample['filename']))
        
        bbox = literal_eval(sample['bbox'])
        try:
            body_image = context_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        except:
            print(f"Error: {sample['folder']}, {sample['filename']}, {bbox[0], bbox[1], bbox[2], bbox[3]}")
            body_image = context_image
        # use transform
        if self.context_transform is not None:
            context_image = context_image.convert("RGB")
            context_image = self.context_transform(context_image)
        if self.body_transform is not None:
            body_image = body_image.convert("RGB")
            body_image = self.body_transform(body_image)
        
        # get VAD
        valence = sample['valence'] / 10.0 if self.normalize else sample['valence']
        arousal = sample['arousal'] / 10.0 if self.normalize else sample['arousal']
        dominance = sample['dominance'] / 10.0 if self.normalize else sample['dominance']

        # get category label torch.tensor
        cat_label_temp = np.zeros(26)
        for cat in sample['category']:
            cat_label_temp[cat] = 1 

        if self.is_exclude:
            # Remove the elements whose index is in minor_category
            cat_label = np.array([item for idx, item in enumerate(cat_label_temp) if idx not in self.minor_category])
            # Ensure this removing must not change the true category 
            # because we already filtered out the minor categories in metadata
            assert np.sum(cat_label_temp) == np.sum(cat_label), "Minor categories are not properly removed"
            assert len(cat_label) == 26 - len(self.minor_category), "Minor categories are not properly removed"
        elif self.cluster:
            cat_label = np.zeros(4) # angry, happy, neutral, sad
            for cat in sample['category']:
                if cat in self.angry_idx:     cat_label[0] = 1
                elif cat in self.happy_idx:   cat_label[1] = 1
                elif cat in self.neutral_idx: cat_label[2] = 1
                elif cat in self.sad_idx:     cat_label[3] = 1
        else:
            cat_label = cat_label_temp
        cat_label = torch.from_numpy(cat_label)
        
        # find subject whose sample[f'subject{1~8}_rep{repeat_index}_beta_idx'] is not -1
        # sample['subject'] can be either 'n' or 'all_n'. 
        # Extract n from it.
        sub_idx = int(re.search(r'(\d+)', sample['subject']).group(1))
        assert sub_idx in range(1, 9)
        repeat_index = np.random.randint(3)
        beta_idx = sample[f'subject{sub_idx}_rep{repeat_index}_beta_idx']
        subj = f'subj{sub_idx}'

        voxel_path = np.load(os.path.join(self.voxel_paths[subj], f"idx{beta_idx}.npy"))
        voxel = torch.from_numpy(voxel_path).unsqueeze(0)
        voxel = (voxel - torch.tensor(self.voxel_means[subj])) / torch.tensor(self.voxel_stds[subj])
        brain_data = F.adaptive_max_pool1d(voxel, self.pool_num).squeeze(0)
                
        return context_image, body_image, valence, arousal, dominance, cat_label, brain_data

        
class BrainDataset(Dataset):
    """
    Dataset for brain data guidance while image => emotion category prediction
    """
    def __init__(self,
                 subjects,
                 split,
                 data_type='brain3d',
                 context_transform=None,
                 body_transform=None,
                 normalize=False,
                 ):

        self.coco_data_path = "/home/dongho/brain2valence/data/emotic"
        self.nsd_data_path="/home/data/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split"
        self.subjects = subjects
        self.split = split
        self.data_type = data_type
        self.context_transform = context_transform
        self.body_transform = body_transform
        self.normalize = normalize

        print(f"Pulling Emotic + COCO + NSD Brain data given split: {split}, subjects: {subjects}")
        emotic_data = utils.get_emotic_df(is_split=False)
        self.metadata = utils.get_emotic_coco_nsd_df(emotic_data=emotic_data, 
                                                     split=split, 
                                                     subjects=subjects)
        

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):

        sample = self.metadata.iloc[idx]
        
        context_image = Image.open(os.path.join(self.coco_data_path, sample['folder'], sample['filename']))
        
        bbox = sample['bbox']
        body_image = context_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        # use transform
        if self.context_transform is not None:
            context_image = context_image.convert("RGB")
            context_image = self.context_transform(context_image)
        if self.body_transform is not None:
            body_image = body_image.convert("RGB")
            body_image = self.body_transform(body_image)

        # get VAD
        valence = sample['valence'] / 10.0 if self.normalize else sample['valence']
        arousal = sample['arousal'] / 10.0 if self.normalize else sample['arousal']
        dominance = sample['dominance'] / 10.0 if self.normalize else sample['dominance']

        # get category label torch.tensor
        cat_label = torch.zeros(26)
        for cat in  sample['category']:
            cat_label[cat] = 1

        # Because there are 3 snapshots of both brain3d and roi
        repeat_index = idx % 3

        data = None
        # kind of a naive way.. but it works
        split = sample['brain3d'].split('_')[0]

        if self.data_type == 'brain3d':
            brain_3d = torch.from_numpy(np.load(os.path.join(self.nsd_data_path, sample['brain3d'])))  # (3, *, *, *)
            brain_3d = brain_3d[repeat_index]
            brain_3d = self.reshape_brain3d(brain_3d)  # (96, 96, 96)

            data = brain_3d
        elif self.data_type == 'roi':
            if len(self.subjects) > 1:
                raise ValueError("Only one subject's roi data is available")
            roi = torch.from_numpy(np.load(os.path.join(self.nsd_data_path, sample['roi'])))  # (3, *)
            roi = roi[repeat_index]

            data = roi
        elif self.data_type == 'emo_vis_roi' or self.data_type == 'emo_roi':
            if len(self.subjects) > 1:
                raise ValueError("Only one subject's roi data is available")
            
            roi_path = f"/home/data/nsd_aws/nsddata/ppdata/subj0{self.subjects[0]}/func1pt8mm/roi"
            hcp_mmp_roi = nib.load(os.path.join(roi_path, 'HCP_MMP1.nii.gz')).get_fdata()
            nsdgeneral_roi = nib.load(os.path.join(roi_path, 'nsdgeneral.nii.gz')).get_fdata()
            
            emotion_related_roi_idx = [104, 106, 109, 111, 112, 126, 127, 155, 167,168, 178]
            
            nsdgeneral_mask = (nsdgeneral_roi == 1)
            emotion_related_mask = np.isin(hcp_mmp_roi, emotion_related_roi_idx)
            
            if self.data_type == 'emo_vis_roi':
                roi = nsdgeneral_mask | emotion_related_mask
            elif self.data_type == 'emo_roi':
                roi = emotion_related_mask
            
            brain_3d = torch.from_numpy(np.load(os.path.join(self.nsd_data_path, sample['brain3d'])))[repeat_index]
            data = brain_3d[roi].flatten()
        else: 
            raise ValueError("data_type should be either 'brain3d' or 'roi'")

        return context_image, body_image, valence, arousal, dominance, cat_label, data

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