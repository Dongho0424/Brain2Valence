import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import utils
from model import Image2VADModel
from torchvision.transforms import v2
from tqdm import tqdm
import pandas as pd
import scipy
from dataset import EmoticDataset
from loss import ContinuousLoss_L2, DiscreteLoss, ContinuousLoss_SL1


class EmoticTrainer:
    def __init__(self, args):
        self.args = args

        self.set_device()
        self.train_dl, self.val_dl, self.num_train, self.num_val = self.prepare_dataloader()
        self.set_wandb_config()
        self.model: nn.Module = self.get_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.disc_loss, self.vad_loss = self.get_criterion()
    
    def set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:', self.device)
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    def set_wandb_config(self):
        wandb_project = self.args.wandb_project
        model_name = self.args.model_name
        print(f"wandb {wandb_project} run {model_name}")
        wandb.login(host='https://api.wandb.ai')
        wandb_config = {
            "model": self.args.model,
            "model_name": self.args.model_name,
            "batch_size": self.args.batch_size,
            "epochs": self.args.epochs,
            "num_train": self.num_train,
            "num_val": self.num_val,
            "seed": self.args.seed,
            "weight_decay": self.args.weight_decay,
        }
        print("wandb_config:\n", wandb_config)

        wandb.init(
            id=self.args.wandb_name,
            project=wandb_project,
            name=self.args.wandb_name,
            config=wandb_config,
            resume="allow",
        )

    def prepare_dataloader(self):

        print("Pulling EMOTIC data...")

        data_path = "/home/dongho/brain2valence/data/emotic"

        print('Prepping train and validation dataloaders...')

        train_data, val_data, _ = utils.get_emotic_df()

        train_context_transform, train_body_transform, test_context_transform, test_body_transform =\
            utils.get_transforms_emotic()

        train_dataset = EmoticDataset(data_path=data_path,
                                      split='train',
                                      emotic_annotations=train_data,
                                      model_type="B",
                                      context_transform=train_context_transform,
                                      body_transform=train_body_transform,
                                      normalize=True,
                                      )

        val_dataset = EmoticDataset(data_path=data_path,
                                    split='val',
                                    emotic_annotations=val_data,
                                    model_type="B",
                                    context_transform=test_context_transform,
                                    body_transform=test_body_transform,
                                    normalize=True,
                                    )

        train_dl = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
        print('# train data:', len(train_dataset))
        print('# val data:', len(val_dataset))

        return train_dl, val_dl, len(train_dataset), len(val_dataset)

    def get_model(self):
        model = Image2VADModel(
            backbone=self.args.model,
            model_type=self.args.model_type,
            pretrained=self.args.pretrain,
            backbone_freeze=self.args.backbone_freeze,
        )

        utils.print_model_info(model)
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

    def get_scheduler(self):
        if self.args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=7, gamma=0.1)
        elif self.args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.epochs)
        return scheduler

    def get_criterion(self):
        disc = DiscreteLoss(weight_type='dynamic', device=self.device)
        if self.args.criterion == "emotic_L2":
            cont = ContinuousLoss_L2(margin=0.1)
        elif self.args.criterion == "emotic_SL1":
            cont = ContinuousLoss_SL1(margin=0.1)
        else: raise NotImplementedError(f'criterion {self.args.criterion} is not implemented')

        return disc, cont
    
    def train(self):
        print("#### enter Training ####")
        print("#### enter Training ####")
        print("#### enter Training ####")

        best_val_loss = float("inf")
        # now set equal
        cat_loss_param = 0.5
        vad_loss_param = 0.5

        for epoch in range(self.args.epochs):
            self.model.cuda()
            self.model.train()
            train_loss = 0

            for i, (context_image, body_image, valence, arousal, dominance, category) in tqdm(enumerate(self.train_dl)):

                context_image = context_image.float().cuda()
                body_image = body_image.float().cuda()
                gt_cat = category.float().cuda()
                gt_vad = torch.stack([valence, arousal, dominance], dim=1).float().cuda()

                pred_cat, pred_vad = self.model(body_image, context_image)

                loss_cat = self.disc_loss(pred_cat, gt_cat)
                loss_vad = self.vad_loss(pred_vad, gt_vad)
                loss = cat_loss_param * loss_cat + vad_loss_param * loss_vad

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= self.num_train

            wandb.log(
                {"train_loss": train_loss,
                 "lr": self.optimizer.param_groups[0]['lr']}, step=epoch)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, (context_image, body_image, valence, arousal, dominance, category) in tqdm(enumerate(self.val_dl)):

                    context_image = context_image.float().cuda()
                    body_image = body_image.float().cuda()
                    gt_cat = category.float().cuda()
                    gt_vad = torch.stack([valence, arousal, dominance], dim=1).float().cuda()

                    pred_cat, pred_vad = self.model(body_image, context_image)
                    loss_cat = self.disc_loss(pred_cat, gt_cat)
                    loss_vad = self.vad_loss(pred_vad, gt_vad)

                    loss = cat_loss_param * loss_cat + vad_loss_param * loss_vad
                    val_loss += loss.item()
                
                val_loss /= self.num_val
                wandb.log({"val_loss": val_loss}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print ('saving model at epoch: %d' %(epoch))
                wandb.log({"best_val_loss": best_val_loss}, step=epoch)
                self.save_model(self.args, self.model, best=True)

            self.scheduler.step()

            print("Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}".format(
                epoch, train_loss, val_loss))

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
        model_name = args.model_name  # ex) "all_subjects_res18_mae_2"
        save_dir = os.path.join(args.save_path, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model.cpu()
        if best:
            torch.save(model.state_dict(), os.path.join(
                save_dir, "best_model.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(
                save_dir, "last_model.pth"))

