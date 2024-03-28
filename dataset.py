import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

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
            self.train_metadata, self.val_metadata = train_test_split(
                self.metadata, test_size=0.1, random_state=fixed_suffle_seed)

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
        nsd_id = self.metadata['nsd_id'].apply(lambda x: np.load(os.path.join(self.data_path, x.split('_')[0], x))[-1])

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
                 model_type="B",
                 context_transform=None,
                 body_transform=None,
                 normalize=False,
                 ):

        self.data_path = data_path
        self.split = split  # train, val, test
        self.metadata = emotic_annotations
        self.model_type = model_type # ['B', 'BI']
        self.context_transform = context_transform
        self.body_transform = body_transform
        self.normalize = normalize

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):

        sample = self.metadata.iloc[idx]
        
        context_image = Image.open(os.path.join(self.data_path, sample['folder'], sample['filename']))
        
        use_body = 'B' in self.model_type # "B"ody 
        
        # crop body from image
        # using bbox
        if use_body:
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

        return context_image, body_image, valence, arousal, dominance, cat_label