import wandb
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import utils 
from model import Brain2ValenceModel

class Trainer:
    def __init__(self, args):
        self.args = args

        self.set_wandb_config()
        self.train_dl, self.val_dl, self.num_train, self.num_val = self.prepare_dataloader(args)
        self.model: nn.Module = self.get_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.criterion = self.get_criterion()

    def set_wandb_config(self):
        wandb_project = self.args.wandb_project
        wandb_run = self.args.wandb_name
        
        print(f"wandb {wandb_project} run {wandb_run}")
        wandb.login(host='https://api.wandb.ai')
        wandb_config = {
            "model_name": self.args.model_name,
            
            "batch_size": self.args.batch_size,
            "epochs": self.args.num_epochs,
            "num_train": self.num_train,
            "num_val": self.num_val,
            "seed": self.args.seed,
        }
        print("wandb_config:\n",wandb_config)
    
    # FIXME to prepare all subjects data
        # 1. 일단 train, val, test 다 합쳐서 전체로 만들기
        # 2. 여기서는 다 불러오기만 하고, EMOTIC dataset과 겹치는 것 필터링은 따로 하기.
        # 3. 이 단계에서 train, test 딱 필터링해서 데이터 줄 수 없나? -> func 하나 더 만들어서 해결
    def prepare_dataloader(self): 
        print('Pulling NSD webdataset data...')

        data_path = "/home/juhyeon/fsx/proj-medarc/fmri/natural-scenes-dataset"
        train_url = "{" + f"{data_path}/webdataset_avg_split/train/train_subj0{self.args.subj}_" + "{0..17}.tar," + f"{data_path}/webdataset_avg_split/val/val_subj0{self.args.subj}_0.tar" + "}"
        val_url = f"{data_path}/webdataset_avg_split/test/test_subj0{self.args.subj}_" + "{0..1}.tar"
        print(train_url,"\n",val_url)
        data_url = 
        num_data = 73000

        """
        batch_size,
        image_var='images', 
        num_devices=None,
        data_url=None,
        num_data=None,
        seed=0,
        voxels_key="nsdgeneral.npy",
        to_tuple=["voxels", "images", "coco", "brain_3d"],
        """

        print('Prepping train and validation dataloaders...')
        dataloader = utils.get_dataloaders(
            self.args.batch_size,
            'images',
            num_devices=torch.cuda.device_count(),
            data_url=data_url
            num_data=,
            seed=self.args.seed,
            voxels_key='nsdgeneral.npy',
            to_tuple=["voxels", "images", "coco", "brain_3d"],
        )
        
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.num_train = num_train
        self.num_val = num_val
        self.train_url = train_url
        self.val_url = val_url
    
    def get_model():
        model = Brain2ValenceModel()
        return model

    def get_optimizer(self):
        params = self.model.parameters()
            
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer

    def get_scheduler(self,):
        if self.args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        return scheduler

    
    def get_criterion(self):
        if self.args.criterion == "mse":
            criterion = nn.MSELoss()
        elif self.args.criterion == "mae":
            criterion = nn.L1Loss()
        return criterion
    
    def train(self, args):
        self.model.cuda()
        best_val_loss = float("inf")
        for epoch in range(args.epochs):
            self.model.train() # TODO
            train_loss = 0
            for i, (img, valence_mean, valence_sd, arousal_mean, arousal_sd) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                img = img.cuda()
                valence_mean = valence_mean.cuda()
                arousal_mean = arousal_mean.cuda()
                pred_valence_mean, pred_arousal_mean = self.model(img)
                if args.mode == "both":
                    valence_mean = valence_mean.float()
                    arousal_mean = arousal_mean.float()
                    loss = self.criterion(pred_valence_mean, valence_mean) + self.criterion(pred_arousal_mean, arousal_mean)
                elif args.mode == "valence":
                    valence_mean = valence_mean.float()
                    loss = self.criterion(pred_valence_mean, valence_mean)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * img.size(0)
            
            train_loss /= len(self.train_loader.dataset)
            wandb.log({"train_loss": train_loss, 
                    "lr": self.optimizer.param_groups[0]['lr']}, step=epoch)
                
            self.model.eval()
            val_loss = 0
            for i, (img, valence_mean, valence_sd, arousal_mean, arousal_sd) in enumerate(self.val_loader):
                img = img.cuda()
                valence_mean = valence_mean.cuda()
                arousal_mean = arousal_mean.cuda()
                pred_valence_mean, pred_arousal_mean = self.model(img)
                if args.mode == "both":
                    loss = self.criterion(pred_valence_mean, valence_mean) + self.criterion(pred_arousal_mean, arousal_mean)
                elif args.mode == "valence":
                    loss = self.criterion(pred_valence_mean, valence_mean)
                val_loss += loss.item() * img.size(0)
            val_loss /= len(self.val_loader.dataset)
            wandb.log({"val_loss": val_loss}, step=epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(args, self.model, best=True)
                
            self.scheduler.step()

            print("Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}".format(epoch, train_loss, val_loss))
        
        self.save_model(args, self.model, best=False)
        wandb.log({"best_val_loss": best_val_loss})
        return self.model
    
    def make_log_name(self, args):
        log_name = ""
        log_name += "model={}-".format(args.model).replace("/", "_")
        log_name += "cri={}-".format(args.criterion)
        log_name += "bs={}-".format(args.batch_size)
        log_name += "epoch={}-".format(args.epochs)
        log_name += "n_layers={}-".format(args.n_layers)
        log_name += "optim={}-".format(args.optimizer)
        log_name += "sche={}-".format(args.scheduler)
        log_name += "lr={}-".format(args.lr)
        log_name += "wd={}-".format(args.weight_decay)
        log_name += "momentum={}-".format(args.momentum)
        log_name += "seed={}".format(args.seed)
        return log_name

    def save_model(self, args, model, best):
        log_name = self.make_log_name(args)
        save_dir = os.path.join(args.save_path, args.wandb_run_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if best:
            torch.save(model.state_dict(), os.path.join(save_dir, log_name + "_best_model.pth"))
            # wandb.save(os.path.join(save_dir, log_name + "_best_model.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_dir, log_name + "_last_model.pth"))
            # wandb.save(os.path.join(save_dir, log_name + "_last_model.pth"))