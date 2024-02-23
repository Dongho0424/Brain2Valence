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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#TODO: label 수를 낮춰서 negative, neutral, positive 감정 label로 다운시키고, 그것을 classification 하는 task도 생각할 수 있겠다.
#TODO: brain을 다 쓰지 말고, 감정에 잘 반응하는 특정 ROI가 있다.
#TODO: NSD -> 3T로 넘어 가는게 허들. 3T에서 처리를 잘 할 수 있는 디코더를 만들어야 함. 유의미하다. whole brain을 잘 해야하낟.
# 스피치, 동영상에서도 디코딩을 잘 해내야하는 것. 
# Question: NSD dataset의 사진을 보고, 우리가 크게 3개의 감정을 나눠서 분류하는 classificaiton task를 할 수 있을까?
# - 아무튼 NSD의 사진을 보고 image 감정 라벨링을 우리가 하는 것이지.
#TODO: Emoset dataset의 여러 감정 노테이션을 크게 3개의 감정으로 나눠서 그것을 분류하는 classification task를 할 수 있을까?
def print_model_info(model):
    total_params = 0
    print("Model's net structure:")
    print("Model name:", model.__class__.__name__)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Type: {type(param.data).__name__}, Parameters: {param.numel()}")
            total_params += param.numel()
    print(f"\nTotal trainable parameters: {total_params}")

def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')

def get_emotic_data() -> dict:
    """
    return 
    ------
    emotic_annotations
    1. get EMOTIC data from COCO dataset
    2. zip its cocoid with emotic_annotations(valence, arousal, dominance)
    """
    file_name_emotic_annot = './emotic_annotations.mat'

    ## get EMOTIC data
    data = scipy.io.loadmat(file_name_emotic_annot, simplify_cells=True)
    emotic_data = data['train'] + data['test'] + data['val']
    emotic_coco_data = [x for x in emotic_data if x['original_database']['name']=='mscoco']
    coco_id = [x['original_database']['info']['image_id'] for x in emotic_coco_data]

    emotic_annotations = []
    for sample in emotic_coco_data:
        person = [sample['person']] if isinstance(sample['person'], dict) else sample['person']

        valences = []
        arousals = []
        dominances = []
        for p in person:
            emotions = p['annotations_continuous']
            emotions = [emotions] if isinstance(emotions, dict) else emotions

            if len(emotions) != 1: # 한 사진에 대해서 여러명이 annotate한 경우
                valences.append(p['combined_continuous']['valence'])
                arousals.append(p['combined_continuous']['arousal'])
                dominances.append(p['combined_continuous']['dominance'])
            else:
                valences.append(p['annotations_continuous']['valence'])
                arousals.append(p['annotations_continuous']['arousal'])
                dominances.append(p['annotations_continuous']['dominance'])

        emotic_annotations += [{ 'valence':valences, 'arousal':arousals, 'dominance':dominances}]

    emotic_annotations = dict(zip(coco_id, emotic_annotations))

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
    file_name_nsd_stim = './nsd_stim_info_merged.csv'

    ## get NSD data
    df = pd.read_csv(file_name_nsd_stim)
    nsd_id = df['nsdId'].values
    nsd_cocoid = df['cocoId'].values
    nsd_cocosplit = df['cocoSplit'].values
    nsd_isshared = df['shared1000'].values

    joint_cocoid = nsd_cocoid[np.isin(nsd_cocoid, list(emotic_annotations.keys()))]
    target_cocoid = [coco_id for coco_id in joint_cocoid if len(emotic_annotations[coco_id]['valence']) == 1]
    train_cocoid = nsd_cocoid[np.isin(nsd_cocoid, target_cocoid) &  ~nsd_isshared]
    test_cocoid = nsd_cocoid[np.isin(nsd_cocoid, target_cocoid) &  nsd_isshared]

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
            use_sampler=use_sampler
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
            use_sampler=use_sampler
        )

        sampler = None
        # using WeightedRandomSampler at classif task
        if task_type == "classif" and use_sampler:
            weights = torch.Tensor(train_dataset.get_weights().values)
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

        train_dl = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
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
            use_sampler=use_sampler
        )        

        sampler = None
        # using WeightedRandomSampler at classif task
        # NOTE: Originally, test dataset does not need to be weighted, but for the sake of class consistency
        if task_type == "classif" and use_sampler:
            weights = torch.Tensor(test_dataset.get_weights().values)
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

            # temporarily check whether test dataset is well-distirbuted
            verify_distribution(sampler, test_dataset)

        test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

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
    correctness_percentage = {true: (count / true_valences.count(true)) * 100 for true, count in correctness_count.items()}

    plt.bar(correctness_percentage.keys(), correctness_percentage.values(), align='center', alpha=0.5)
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


from collections import Counter

def verify_distribution(sampler, dataset):

    # Create a list to store the targets
    targets = []

    # Iterate over the sampled data
    for idx in sampler:
        brain3d, target = dataset[idx]
        targets.append(target)

    # Count the number of instances of each class
    class_counts = Counter(targets)

    # Print the counts
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count}")