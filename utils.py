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
    annotations = [x['person'] for x in emotic_coco_data] 
    emotic_annotations = []
    for annot in annotations:
        annot = [annot] if type(annot)==dict else annot

        valence = []; arousal = []; dominance = []
        for person in annot:
            person = person['annotations_continuous']
            person = [person] if type(person)==dict else person
            valence += [np.mean([x['valence'] for x in person])]
            arousal += [np.mean([x['arousal'] for x in person])]
            dominance += [np.mean([x['dominance'] for x in person])]
        emotic_annotations += [{ 'valence':valence, 'arousal':arousal, 'dominance':dominance}]

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
        Parameters
        -----
        valence: torch.Tensor, shape (B, 1)
        task_type: str, "reg" or "classif"
        num_classif: int, 3 or 5

        Note
        ------
        valence: each valence is 0~10 float value
        task_type:
        - regression: normalize to be 0~1
        - classification: 3 or 5 class classification
            - 3 classes: each valence [0, 4], (4, 7], (7, 10] maps to 0, 1, 2
            - 5 classes: each valence (0, 2], (2, 4], (4, 6], (6, 8], (8, 10] maps to 0, 1, 2, 3, 4

        return
        ------
        target_valence : torch.Tensor, shape (B, 1)
            - as value(float): 0~1 for regression
            - as label(int): 0~4(or 0~2) for classification
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
):
    
    if mode == 'train':

        train_dataset = BrainValenceDataset(
            data_path=data_path,
            split="train",
            emotic_annotations=emotic_annotations,
            nsd_df=nsd_df,
            target_cocoid=target_cocoid,
            subjects=subjects,
            task_type=task_type,
            num_classif=num_classif
        )
        val_dataset = BrainValenceDataset(
            data_path=data_path,
            split="val",
            emotic_annotations=emotic_annotations,
            nsd_df=nsd_df,
            target_cocoid=target_cocoid,
            subjects=subjects,
            task_type=task_type,
            num_classif=num_classif
        )

        # using WeightedRandomSampler 
        train_weights = torch.Tensor(train_dataset.get_weights().values)
        train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)

        train_dl = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
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
            num_classif=num_classif
        )        

        test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
   
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