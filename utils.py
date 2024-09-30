from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import pandas as pd
import math
from dataset import BrainValenceDataset
import scipy
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

def print_model_info(model):
    total_trainable_params = 0
    total_nontrainable_params = 0

    print("Model's net structure:")
    print("Model name:", model.__class__.__name__)
    for name, param in model.named_parameters():
        # print(f"Layer: {name}, Type: {type(param.data).__name__}, Parameters: {param.numel()}")
        if param.requires_grad:
            total_trainable_params += param.numel()
        else:
            total_nontrainable_params += param.numel()
    print("\nTotal trainable parameters: {:.3f}M".format(total_trainable_params/1e6))
    print("\nTotal Non-trainable parameters: {:.3f}M".format(total_nontrainable_params/1e6))

def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

def get_emotic_data() -> dict:
    """
    return 
    ------
    emotic_annotations
    1. get EMOTIC data from COCO dataset
    2. zip its cocoid with annotations(bbox, valence, arousal, dominance)
    """
    # file_name_emotic_annot = '/home/dongho/brain2valence/data/emotic_annotations.mat'
    file_name_emotic_annot = 'Annotations/Annotations.mat'
    

    # get EMOTIC data
    data = scipy.io.loadmat(file_name_emotic_annot, simplify_cells=True)
    emotic_data = data['train'] + data['test'] + data['val']
    emotic_coco_data = [
        x for x in emotic_data if x['original_database']['name'] == 'mscoco']
    coco_id = [x['original_database']['info']['image_id']
               for x in emotic_coco_data]

    emotic_annotations = []
    for sample in emotic_coco_data:

        coco_id = sample['original_database']['info']['image_id']
        filename = sample['filename']
        person = [sample['person']] if isinstance(
            sample['person'], dict) else sample['person']

        annot_per_person = []
        for p in person:
            annot = dict()

            # add bbox
            annot['bbox'] = p['body_bbox']

            emotions = p['annotations_continuous']
            emotions = [emotions] if isinstance(emotions, dict) else emotions

            if len(emotions) != 1:  # 한 사진에 대해서 여러명이 annotate한 경우
                annot['valence'] = p['combined_continuous']['valence']
                annot['arousal'] = p['combined_continuous']['arousal']
                annot['dominance'] = p['combined_continuous']['dominance']
            else:
                annot['valence'] = p['annotations_continuous']['valence']
                annot['arousal'] = p['annotations_continuous']['arousal']
                annot['dominance'] = p['annotations_continuous']['dominance']

            annot_per_person.append(annot)
        emotic_annotation = dict()
        emotic_annotation['coco_id'] = coco_id
        emotic_annotation['filename'] = filename
        emotic_annotation['people'] = annot_per_person
        emotic_annotations.append(emotic_annotation)

    return emotic_annotations


def get_NSD_data(emotic_annotations):
    """
    return
    ------
    NSD data and target_cocoid
    - target_cocoid: 
        1. equally in both NSD, EMOTIC and COCO dataset 
        2. has only one person in the image
        3. has valence annotation
    """
    # out: target_cocoid
    file_name_nsd_stim = '/home/dongho/brain2valence/data/nsd_stim_info_merged.csv'

    # get NSD data
    df = pd.read_csv(file_name_nsd_stim)
    nsd_id = df['nsdId'].values
    nsd_cocoid = df['cocoId'].values
    nsd_cocosplit = df['cocoSplit'].values
    nsd_isshared = df['shared1000'].values

    cocoid_from_emotic = [e['coco_id'] for e in emotic_annotations]
    joint_cocoid = nsd_cocoid[np.isin(nsd_cocoid, cocoid_from_emotic)]

    # target_cocoid: has only one person in the image
    # currently, not used
    # target_cocoid = [coco_id for coco_id in joint_cocoid if len(emotic_annotations[coco_id]) == 1]

    # target_cocoid: regardless of the number of people in the image
    target_cocoid = joint_cocoid
    # train_cocoid = nsd_cocoid[np.isin(nsd_cocoid, target_cocoid) &  ~nsd_isshared]
    # test_cocoid = nsd_cocoid[np.isin(nsd_cocoid, target_cocoid) &  nsd_isshared]

    return df, target_cocoid

# currently, not used


def get_target_valence(valence: torch.Tensor, task_type, num_classif):
    """
    Get the target valence based on the given valence tensor, task type, and number of classifications.

    Args:
        valence: torch.Tensor, shape (B, 1)
        task_type: str, "reg" or "classif"
        num_classif: int, 3 or 5

    Returns:
        torch.Tensor: The target valence tensor.

    Note:
        valence: each valence is 0~10 float value
        task_type:
        - regression: normalize to be 0~1
        - classification: 3 or 5 class classification
            - 3 classes: each valence [0, 4], (4, 7], (7, 10] maps to 0, 1, 2
            - 5 classes: each valence (0, 2], (2, 4], (4, 6], (6, 8], (8, 10] maps to 0, 1, 2, 3, 4

    Raises:
        ValueError: If the task_type is not 'reg' or 'classif'.
        ValueError: If the num_classif is not 3 or 5.
    """

    if task_type == 'reg':
        # Normalize valence to be in the range 0~1 for regression
        target_valence = valence / 10.0
    elif task_type == 'classif':
        if num_classif == 3:
            # Map valence to 0, 1, 2 for 3 classes
            boundaries = torch.tensor([0, 4, 7, 10]).float()
            target_valence = torch.bucketize(valence, boundaries) - 1
        elif num_classif == 5:
            # Map valence to 0, 1, 2, 3, 4 for 5 classes
            boundaries = torch.tensor([0, 2, 4, 6, 8, 10]).float()
            target_valence = torch.bucketize(valence, boundaries) - 1
        else:
            raise ValueError("num_classif must be either 3 or 5.")
    else:
        raise ValueError("task_type must be either 'reg' or 'classif'.")

    return target_valence


def get_torch_dataloaders(
    batch_size,
    data_path,
    emotic_annotations,
    nsd_df,
    target_cocoid,
    mode='train',
    subjects=[1, 2, 5, 7],
    task_type="reg",
    num_classif=3,
    data: str = 'brain3d',
    use_sampler=False,
    use_body=True,
    transform=None,
):
    """
    Get PyTorch data loaders for training, validation, or testing.

    Args
    -------
        batch_size (int): The batch size for the data loaders.
        data_path (str): The path to the data.
        emotic_annotations (str): The path to the Emotic annotations.
        nsd_df (str): The path to the NSD DataFrame.
        target_cocoid (str): The target COCO ID.
        mode (str, optional): The mode of the data loaders. Defaults to 'train'.
        subjects (list, optional): The list of subject IDs. Defaults to [1, 2, 5, 7].
        task_type (str, optional): The type of task. Defaults to "reg".
        num_classif (int, optional): The number of classifications. Defaults to 3.
        data (str, optional): The type of data. Choices of ['brain3d', 'roi']
        use_sampler (bool, optional): Whether to use the weighted random sampler. Defaults to False.
        use_body (bool, optional): Whether to use cropped image by person. Defaults to True.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Defaults to None.
    Returns
    -------
        tuple: A tuple containing the data loaders and dataset sizes.
            - If mode is 'train', returns (train_dl, val_dl, train_dataset_size, val_dataset_size).
            - If mode is 'test', returns (test_dl, test_dataset_size).
    """

    if mode == 'train':

        train_dataset = BrainValenceDataset(
            data_path=data_path,
            split="train",
            emotic_annotations=emotic_annotations,
            nsd_df=nsd_df,
            target_cocoid=target_cocoid,
            subjects=subjects,
            task_type=task_type,
            num_classif=num_classif,
            data=data,
            use_sampler=use_sampler,
            use_body=use_body,
            transform=transform,
        )
        val_dataset = BrainValenceDataset(
            data_path=data_path,
            split="val",
            emotic_annotations=emotic_annotations,
            nsd_df=nsd_df,
            target_cocoid=target_cocoid,
            subjects=subjects,
            task_type=task_type,
            num_classif=num_classif,
            data=data,
            use_sampler=use_sampler,
            use_body=use_body,
            transform=transform,
        )

        sampler = None
        # using WeightedRandomSampler at classif task
        if task_type == "classif" and use_sampler:
            weights = torch.Tensor(train_dataset.get_weights().values)
            sampler = WeightedRandomSampler(
                weights=weights, num_samples=len(weights), replacement=True)

            # check whether test dataset is well-distirbuted
            verify_distribution(sampler, train_dataset)

        train_dl = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_dl, val_dl, len(train_dataset), len(val_dataset)

    elif mode == 'test':

        test_dataset = BrainValenceDataset(
            data_path=data_path,
            split="test",
            emotic_annotations=emotic_annotations,
            nsd_df=nsd_df,
            target_cocoid=target_cocoid,
            subjects=subjects,
            task_type=task_type,
            num_classif=num_classif,
            data=data,
            use_sampler=use_sampler,
            use_body=use_body,
            transform=transform,
        )

        sampler = None
        print("get_torch_dataloaders: use_sampler", use_sampler)
        print("get_torch_dataloaders: task_type", task_type)
        # using WeightedRandomSampler at classif task
        # NOTE: Originally, test dataset does not need to be weighted, but for the sake of class consistency
        if task_type == "classif" and use_sampler:
            weights = torch.Tensor(test_dataset.get_weights().values)
            sampler = WeightedRandomSampler(
                weights=weights, num_samples=len(weights), replacement=True)

            # check whether test dataset is well-distirbuted
            verify_distribution(sampler, test_dataset)

        test_dl = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

        return test_dl, len(test_dataset)

    else:
        TypeError("Wrong mode for dataloader")


def plot_valence_histogram(true_valences, pred_valences):
    '''
    Plot histogram of true and predicted valences per each true valence
    In order to see there is meaningful difference between true valence section
    '''
    # Calculate correctness
    correctness_count = {}

    for true, pred in zip(true_valences, pred_valences):
        if true in correctness_count:
            if true == pred:
                correctness_count[true] += 1
        else:
            correctness_count[true] = 1 if true == pred else 0
    correctness_percentage = {true: (
        count / true_valences.count(true)) * 100 for true, count in correctness_count.items()}

    plt.bar(correctness_percentage.keys(),
            correctness_percentage.values(), align='center', alpha=0.5)
    plt.xlabel('True Valence')
    plt.ylabel('Correctness (%)')
    plt.xticks(range(max(true_valences) + 1))
    plt.yticks(range(0, 101, 10))
    plt.title('Correctness per Each Valence')

    wandb.log({"plot correctness per each valence": wandb.Image(plt)})
    plt.clf()


def get_num_voxels(subject: int) -> int:
    """
    Get the number of voxels for a given subject.

    Args:
        subject (int): The subject number.

    Returns:
        int: The number of voxels for the given subject.
    """
    if subject == 1:
        num_voxels = 15724
    elif subject == 2:
        num_voxels = 14278
    elif subject == 3:
        num_voxels = 15226
    elif subject == 4:
        num_voxels = 13153
    elif subject == 5:
        num_voxels = 13039
    elif subject == 6:
        num_voxels = 17907
    elif subject == 7:
        num_voxels = 12682
    elif subject == 8:
        num_voxels = 14386
    return num_voxels


def verify_distribution(sampler, dataset):

    # Create a list to store the targets
    targets = []

    # Iterate over the sampled data
    for idx in sampler:
        _, target, _, _ = dataset[idx]
        targets.append(target)

    # Count the number of instances of each class
    class_counts = Counter(targets)

    # Print the counts
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count}")


def get_transforms_emotic():
    image_size = 224
    body_size = 112
    context_mean = [0.4690646, 0.4407227, 0.40508908]
    context_std = [0.2514227, 0.24312855, 0.24266963]
    body_mean = [0.43832874, 0.3964344, 0.3706214]
    body_std = [0.24784276, 0.23621225, 0.2323653]

    train_context_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=context_mean, std=context_std),
    ])
    train_body_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((body_size, body_size)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=body_mean, std=body_std),
    ])
    test_context_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=context_mean, std=context_std),
    ])
    test_body_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((body_size, body_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=body_mean, std=body_std),
    ])

    return train_context_transform, train_body_transform, test_context_transform, test_body_transform

def get_emotic_df(is_split=True):
    """
    Parameters
    ---
    is_split=True; 0.7, 0.15, 0.15 train, val, test
    is_split=False; total dataframe

    Return
    ---
    emotic_annotations: pd.DataFrame

    Description
    ---
    - EMOTIC dataset: Multiple people annotated in one image.
    - return metadata: Based on emotional annot not an image, i.e., multiple rows(annots) can be for one image.
    """
    file_name_emotic_annot = '/home/dongho/brain2valence/data/emotic_annotations.mat'
    # file_name_emotic_annot = '/home/juhyeon/Brain2Valence/Annotations/Annotations.mat'
    data = scipy.io.loadmat(file_name_emotic_annot, simplify_cells=True)
    emotic_annotations = data['train'] + data['test'] + data['val']

    folders = []
    filenames = []
    image_ids = []
    bboxes = []
    valences = []
    arousals = []
    dominances = []
    categories = [] # for emotic discrete categories, as index
    emotic_split = []

    cat2idx, _ = get_emotic_categories()
    
    for i, sample in enumerate(emotic_annotations):

        folder = sample['folder']
        filename = sample['filename']
        # Assign -1 to image_id if there is no image_id
        image_id = sample.get('original_database', {}).get('info', {}).get('image_id', -1)
        people = [sample['person']] if isinstance(sample['person'], dict) else sample['person']


        for p in people:
            if i < len(data['train']):
                emotic_split.append('train')
            elif i < len(data['train']) + len(data['test']):
                emotic_split.append('test')
            else:
                emotic_split.append('val')
            folders.append(folder)
            filenames.append(filename)
            image_ids.append(image_id)
            bboxes.append(p['body_bbox'])

            # for valence, arousal, dominance
            continuous_emotions = p['annotations_continuous']
            continuous_emotions = [continuous_emotions] if isinstance(continuous_emotions, dict) else continuous_emotions

            if len(continuous_emotions) != 1:  # 한 사진에 대해서 여러명이 annotate한 경우
                valences.append(p['combined_continuous']['valence'])
                arousals.append(p['combined_continuous']['arousal'])
                dominances.append(p['combined_continuous']['dominance'])
                
            else:
                valences.append(p['annotations_continuous']['valence'])
                arousals.append(p['annotations_continuous']['arousal'])
                dominances.append(p['annotations_continuous']['dominance'])

            # for emotion categories
            if 'combined_categories' in p: 
                # print(p['combined_categories'])
                combined_cat = p['combined_categories']
                combined_cat = np.array([combined_cat]) if type(combined_cat) != np.ndarray else combined_cat
                categories.append([cat2idx[e] for e in combined_cat])
            else:
                annot_cat = p['annotations_categories']['categories']
                annot_cat = np.array([annot_cat]) if type(annot_cat) != np.ndarray else annot_cat
                categories.append([cat2idx[e] for e in annot_cat])

    annotations = pd.DataFrame({'folder': folders,
                                'filename': filenames,
                                'image_id': image_ids,
                                'bbox': bboxes,
                                'valence': valences,
                                'arousal': arousals,
                                'dominance': dominances,
                                'category': categories,
                                'emotic_split': emotic_split})
    if is_split:
        # split data
        # train: 70%, val: 15%, test: 15%
        total_len = len(annotations)
        train_len = int(total_len * 0.7)
        val_len = int(total_len * 0.15)

        train_data = annotations.iloc[:train_len].dropna().reset_index(inplace=False, drop=True)
        val_data = annotations.iloc[train_len:train_len+val_len].dropna().reset_index(inplace=False, drop=True)
        test_data = annotations.iloc[train_len+val_len:].dropna().reset_index(inplace=False, drop=True)

        return train_data, val_data, test_data
    else:
        return annotations.dropna().reset_index(inplace=False, drop=True) 

def get_emotic_coco_nsd_df(emotic_data, split='train', seed=42, subjects=[1, 2, 5, 7]):
    """
    1. emotic data를 가져와서 mscoco 데이터를 남긴다.
    2. metadata를 만든 다음, nsd_df로 부터 coco_id를 가져온다.
    3. emotic+coco+metadata에 해당하는 df를 남긴 뒤, brain3d, roi 데이터를 추가한다.
    """
    ## emotic_df에서 mscoco 데이터만 가져오기
    emotic_coco = emotic_data[emotic_data['folder'] == 'mscoco/images'] #(20702, 8)

    ## bring nsd_df
    file_name_nsd_stim = '/home/dongho/brain2valence/data/nsd_stim_info_merged.csv'
    nsd_df = pd.read_csv(file_name_nsd_stim)

    ## bring metadata
    data_path="/home/data/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split"
    if split in ["all", "one_point"]: # for debugging
        dfs = [pd.read_csv(os.path.join(data_path, f'train_subj0{subj}_metadata.csv')) for subj in subjects]
        dfs += [pd.read_csv(os.path.join(data_path, f'val_subj0{subj}_metadata.csv')) for subj in subjects]
        dfs += [pd.read_csv(os.path.join(data_path, f'test_subj0{subj}_metadata.csv')) for subj in subjects]
        metadata = pd.concat(dfs).reset_index(inplace=False, drop=True)
    elif split in ['train', 'val']:
        dfs = [pd.read_csv(os.path.join(data_path, f'train_subj0{subj}_metadata.csv')) for subj in subjects]
        dfs += [pd.read_csv(os.path.join(data_path, f'val_subj0{subj}_metadata.csv')) for subj in subjects]
        metadata = pd.concat(dfs).reset_index(inplace=False, drop=True)
        train_metadata, val_metadata = train_test_split(metadata, test_size=0.1, random_state=seed)
        
        metadata = train_metadata if split == 'train' else val_metadata
    elif split == 'test':
        dfs = [pd.read_csv(os.path.join(data_path, f'test_subj0{subj}_metadata.csv')) for subj in subjects]
        metadata = pd.concat(dfs).reset_index(inplace=False, drop=True)
    else: raise ValueError("split must be one of ['train', 'val', 'test', 'all', 'one_point']")
    # rename
    metadata = metadata.rename(columns={'coco': 'nsd_id_filename'})

    # get nsd_id from numpy data
    nsd_id = metadata['nsd_id_filename'].apply(lambda x: np.load(os.path.join(data_path, x))[-1])

    # get corresponding image_id from nsd_df
    metadata['image_id'] = nsd_id.apply(lambda x: nsd_df.loc[x, 'cocoId'])

    # emotic_coco 와 metadata의 교집합
    # 5123, 597, 608 (152 * 4)
    emotic_coco_nsd = pd.merge(emotic_coco, metadata, on='image_id', how='inner')\
        .drop(columns=['img', 'trial', 'num_uniques'])\
        .rename(columns={'voxel': 'roi', 'mri': 'brain3d'})
    
    # For debugging
    # Filter out by the length of category and always take the first one
    if split == "one_point":
        emotic_coco_nsd = emotic_coco_nsd[emotic_coco_nsd["category"].apply(lambda x: len(x) == 3)].head(1)

    return emotic_coco_nsd

def get_emotic_df_for_pretraining(subjects):
    """
    Descriptions
    ---
    Return Emotic df excluding images shown to given subjects, for pretraining.
    1. Using `utils.get_emotic_df()` to get emotic_df
    2. Get df given subjects, which is simply using `get_emotic_coco_nsd_df`
    3. Get emotic_df - subject_df (images which are not in subject_df)
    4. Return train, val, test df with split ratio 0.7, 0.15, 0.15
    """
    # If just one subject
    # emotic_df: 33961
    # subj_df: 1614
    # metadata: 32347 = 33961 - 1614
    emotic_df = get_emotic_df(is_split=False)
    subj_df = get_emotic_coco_nsd_df(emotic_data=emotic_df, split='all', subjects=subjects)
    metadata = emotic_df[~emotic_df['image_id'].isin(subj_df['image_id'])]
    total_len = len(metadata)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.15)

    train_data = metadata.iloc[:train_len].dropna().reset_index(inplace=False, drop=True)  # 22642
    val_data = metadata.iloc[train_len:train_len+val_len].dropna().reset_index(inplace=False, drop=True)  # 4852
    test_data = metadata.iloc[train_len+val_len:].dropna().reset_index(inplace=False, drop=True)  # 4853, total 32347

    return train_data, val_data, test_data

def get_emotic_categories():
    categories = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval',
                  'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem',
                  'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness',
                  'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']

    cat2idx = {}
    idx2cat = {}
    for idx, emotion in enumerate(categories):
        cat2idx[emotion] = idx  
        idx2cat[idx] = emotion
    return cat2idx, idx2cat
