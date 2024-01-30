import wandb
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

class Predictor:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.prepare_dataloader() #TODO
        self.set_wandb_config()
        self.prepare_model() #TODO
        

    def set_wandb_config(self):
        # params for wandb
        if self.local_rank == 0 and self.args.wandb_log: # only use main process for wandb logging
            wandb_project = self.args.wandb_project
            wandb_run = self.args.wandb_name
            wandb_notes = ''
            
            print(f"wandb {wandb_project} run {wandb_run}")
            wandb.login(host='https://api.wandb.ai')#, relogin=True)
            wandb_config = self.args
            wandb.init(
                id = self.args.model_name,
                project=wandb_project,
                name=wandb_run,
                config=wandb_config,
                notes=wandb_notes,
            )

        
    def prepare_dataloader(self):
        num_val = 982
        val_batch_size = 1
        
        val_url = f"{self.args.data_path}/webdataset_avg_split/test/test_subj0{self.args.subj}_" + "{0..1}.tar"
                
        voxels_key = 'nsdgeneral.npy' # 1d inputs

        val_data = wds.WebDataset(val_url, resampled=False)\
            .decode("torch")\
            .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
            .to_tuple("voxels", "images", "coco")\
            .batched(val_batch_size, partial=False)

        val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, shuffle=False)
        
        # check that your data loader is working
        for val_i, (voxel, img_input, coco) in enumerate(val_dl):
            print("idx",val_i)
            print("voxel.shape",voxel.shape)
            print("img_input.shape",img_input.shape)
            break
        
        self.val_dl = val_dl
        self.ind_include = np.arange(num_val)