import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import pandas as pd
import math
import webdataset as wds
import braceexpand
from dataset import BrainValenceDataset
import scipy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
# def get_dataloaders(
#     batch_size,
#     target_cocoid,
#     image_var='images',
#     num_devices=None,
#     train_data_urls=None,
#     val_data_urls=None,
#     test_data_urls=None,
#     num_data=None,
#     seed=0,
#     voxels_key="nsdgeneral.npy",
#     to_tuple=["voxels", "images", "coco", "brain_3d"],
# ):
#     """
#     train_data_urls: array of train data url
#     val_data_urls: array of val data url
#     test_data_urls: array of test data url
#     target_cocoid(coco id from both NSD & EMOTIC dataset)
    
#     out: three dataloaders that all returns (brain3d, valence)
#     """

#     print("Getting dataloaders...")
#     assert image_var == 'images'
    
#     def my_split_by_node(urls):
#         return urls
    
#     # data_url = list(braceexpand.braceexpand(data_url))

#     if num_devices is None:
#         num_devices = torch.cuda.device_count()
    
        
#     print(f"in utils.py: num_devices: {num_devices}")
#     global_batch_size = batch_size * num_devices
#     num_batches = math.floor(num_data / global_batch_size)
#     # num_worker_batches = math.floor(num_batches / num_workers)
#     # if num_worker_batches == 0: num_worker_batches = 1

#     print("\nnum_data",num_data)
#     print("global_batch_size",global_batch_size)
#     print("batch_size",batch_size)
#     print("num_batches",num_batches)

#     def filter_by_cocoId(sample):
#         # sample: ("voxels", "images", "cocoid", "brain_3d")
#         _, _, cocoid, _ = sample
#         cocoid = cocoid[-1]
#         return (cocoid in target_cocoid)

#     def map_brain_valence_pair(sample):
#         # sample: ("voxels", "images", "cocoid", "brain_3d")
#         # add corresponding valence
#         # make sure all brain_3d shape is same
#         # out: brain_3d, valence

#         _, _, cocoid, brain_3d = sample
#         cocoid = cocoid[-1]
#         valence = get_emotic_annot(cocoid, 'valence')
#         brain_3d = np.mean(brain_3d, axis=0) # (*, *, *)
#         brain_3d = reshape_brain3d(brain_3d, target_shape=(96, 96, 96)) # (96, 96, 96)
        
#         return brain_3d, valence
    
#     ### train ###
    
#     train_target_urls = []
#     for url in train_data_urls:
#         train_target_urls += list(braceexpand.braceexpand(url))
#     print("len(train_target_urls):", len(train_target_urls))

#     # for url in data_urls:
#     train_dataset = wds.WebDataset(train_target_urls, resampled=True, nodesplitter=my_split_by_node)\
#         .shuffle(500, initial=500, rng=random.Random(seed))\
#         .decode("torch")\
#         .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy", brain_3d = "wholebrain_3d.npy")\
#         .to_tuple(*to_tuple)\
#         .select(filter_by_cocoId)\
#         .map(map_brain_valence_pair)\
#         .batched(batch_size, partial=True)\
#         # .with_epoch(num_worker_batches)

#     train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=None, num_workers=1, shuffle=False)       

#     ### val ###
    
#     val_target_urls = []
#     for url in val_data_urls:
#         val_target_urls += list(braceexpand.braceexpand(url))
#     print("len(val_target_urls):", len(val_target_urls))

#     # for url in data_urls:
#     val_dataset = wds.WebDataset(val_target_urls, resampled=True, nodesplitter=my_split_by_node)\
#         .shuffle(500, initial=500, rng=random.Random(seed))\
#         .decode("torch")\
#         .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy", brain_3d = "wholebrain_3d.npy")\
#         .to_tuple(*to_tuple)\
#         .select(filter_by_cocoId)\
#         .map(map_brain_valence_pair)\
#         .batched(batch_size, partial=True)\
#         # .with_epoch(num_worker_batches)

#     val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=None, num_workers=1, shuffle=False)       

#     ### test ###
    
#     test_target_urls = []
#     for url in test_data_urls:
#         test_target_urls += list(braceexpand.braceexpand(url))
#     print("len(test_target_urls):", len(test_target_urls))

#     # for url in data_urls:
#     test_dataset = wds.WebDataset(test_target_urls, resampled=True, nodesplitter=my_split_by_node)\
#         .shuffle(500, initial=500, rng=random.Random(seed))\
#         .decode("torch")\
#         .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy", brain_3d = "wholebrain_3d.npy")\
#         .to_tuple(*to_tuple)\
#         .select(filter_by_cocoId)\
#         .map(map_brain_valence_pair)\
#         .batched(batch_size, partial=True)\
#         # .with_epoch(num_worker_batches)

#     test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=None, num_workers=1, shuffle=False)       
    
#     return train_dl, val_dl, test_dl

def get_torch_dataloaders(
    batch_size,
    data_path,
    emotic_annotations,
    nsd_df,
    target_cocoid, 
    subjects=[1, 2, 5, 7]
):
    train_dataset = BrainValenceDataset(
        data_path=data_path,
        split="train",
        emotic_annotations=emotic_annotations,
        nsd_df=nsd_df,
        target_cocoid=target_cocoid,
        subjects=subjects
    )
    val_dataset = BrainValenceDataset(
        data_path=data_path,
        split="val",
        emotic_annotations=emotic_annotations,
        nsd_df=nsd_df,
        target_cocoid=target_cocoid,
        subjects=subjects
    )
    test_dataset = BrainValenceDataset(
        data_path=data_path,
        split="test",
        emotic_annotations=emotic_annotations,
        nsd_df=nsd_df,
        target_cocoid=target_cocoid,
        subjects=subjects
    )
    
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=300, shuffle=False)
   
    return train_dl, val_dl, test_dl, len(train_dataset), len(val_dataset), len(test_dataset)
