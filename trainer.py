import wandb
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import utils 
from model import Brain2ValenceModel
from tqdm import tqdm

class Trainer:
    def __init__(self, args):
        self.args = args

        self.train_dl, self.val_dl, self.num_train, self.num_val = self.prepare_dataloader()
        self.set_wandb_config()
        self.model: nn.Module = self.get_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.criterion = self.get_criterion()

    def set_wandb_config(self):
        wandb_project = self.args.wandb_project
        wandb_name = self.args.wandb_name
        
        print(f"wandb {wandb_project} run {wandb_name}")
        wandb.login(host='https://api.wandb.ai')
        wandb_config = {
            "model_name": self.args.wandb_name,    
            "batch_size": self.args.batch_size,
            "epochs": self.args.epochs,
            "num_train": self.num_train,
            "num_val": self.num_val,
            "seed": self.args.seed,
            "weight_decay": self.args.weight_decay,
        }
        print("wandb_config:\n",wandb_config)

        wandb.init(
            id = self.args.wandb_name,
            project=wandb_project,
            name=wandb_name,
            config=wandb_config,
            resume="allow",
        )
        
    def prepare_dataloader(self): 
        print("Pulling Brain and Valence pair data...")

        data_path="/home/juhyeon/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split"

        print('Prepping train and validation dataloaders...')

        emotic_annotations = utils.get_emotic_data()
        nsd_df, target_cocoid = utils.get_NSD_data(emotic_annotations)

        train_dl, val_dl, num_train, num_val = utils.get_torch_dataloaders(
            self.args.batch_size,
            data_path = data_path,
            emotic_annotations=emotic_annotations,
            nsd_df=nsd_df,
            target_cocoid=target_cocoid,
            mode='train',
            subjects=[1, 2, 5, 7]
        )

        self.train_dl = train_dl
        self.val_dl = val_dl
        self.num_train = num_train
        self.num_val = num_val

        return train_dl, val_dl, num_train, num_val
    
    def get_model(self):
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
    
    def train(self):
        #### enter Training ###
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:', device)
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())
        #### enter Training ###

        self.model.cuda()
        best_val_loss = float("inf")
        for epoch in range(self.args.epochs):
            self.model.train() 
            train_loss = 0
            for i, (brain_3d, valence) in tqdm(enumerate(self.train_dl)):
                self.optimizer.zero_grad()
                brain_3d = brain_3d.float().cuda()
                valence = valence.float().cuda()
                # valence 0~1로 normalize
                # 해석할 때는 10 곱해서 해석하는 것.
                valence /= 10.0
                pred_valence = self.model(brain_3d)
                # print("brain_3d.shape:", brain_3d.shape)
                # print("valence.shape:", valence.shape)
                # print("pred_valence.shape:", pred_valence.shape)
                
                loss = self.criterion(pred_valence, valence)
                loss.backward()
                self.optimizer.step()
                # loss.item(): batch의 average loss
                # batch size 곱해주면 total loss
                train_loss += loss.item() * brain_3d.shape[0] # multiply by batch size
            
            # 지금까지 train_loss를 총합하였으니, 데이터 개수로 average. 
            train_loss /= float(self.num_train)
            wandb.log(
                {"train_loss": train_loss, 
                 "lr": self.optimizer.param_groups[0]['lr']}, step=epoch)
                
            self.model.eval()
            val_loss = 0
            for i, (brain_3d, valence) in tqdm(enumerate(self.val_dl)):
                brain_3d = brain_3d.float().cuda()
                valence = valence.float().cuda()

                # valence 0~1로 normalize
                valence /= 10.0
                pred_valence = self.model(brain_3d)
                loss = self.criterion(pred_valence, valence)
                val_loss += loss.item() * brain_3d.shape[0] # multiply by batch size
            val_loss /= float(self.num_val)
            wandb.log({"val_loss": val_loss}, step=epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wandb.log({"best_val_loss": best_val_loss})
                self.save_model(self.args, self.model, best=True)
                
            self.scheduler.step()

            print("Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}".format(epoch, train_loss, val_loss))
        
        self.save_model(self.args, self.model, best=False)
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
        model_name = args.model_name # ex) "all_subjects_res18_mae_2"
        save_dir = os.path.join(args.save_path, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if best:
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))

# def prepare_dataloader(self): 
    #     print('Pulling NSD webdataset data...')

    #     data_path = "/home/juhyeon/fsx/proj-medarc/fmri/natural-scenes-dataset"
    #     train_url = "{" + f"{data_path}/webdataset_avg_split/train/train_subj0{self.args.subj}_" + "{0..17}.tar," + f"{data_path}/webdataset_avg_split/val/val_subj0{self.args.subj}_0.tar" + "}"
    #     val_url = f"{data_path}/webdataset_avg_split/test/test_subj0{self.args.subj}_" + "{0..1}.tar"
    #     print(train_url,"\n",val_url)
    #     data_url = 
    #     num_data = 73000

    #     print('Prepping train and validation dataloaders...')
    #     dataloader = utils.get_dataloaders(
    #         self.args.batch_size,
    #         'images',
    #         num_devices=torch.cuda.device_count(),
    #         data_url=data_url
    #         num_data=,
    #         seed=self.args.seed,
    #         voxels_key='nsdgeneral.npy',
    #         to_tuple=["voxels", "images", "coco", "brain_3d"],
    #     )
        
    #     self.train_dl = train_dl
    #     self.val_dl = val_dl
    #     self.num_train = num_train
    #     self.num_val = num_val
    #     self.train_url = train_url
    #     self.val_url = val_url