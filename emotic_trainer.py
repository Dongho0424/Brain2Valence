import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import utils
from model import EmoticModel
from torchvision.transforms import v2
from tqdm import tqdm
import pandas as pd
import scipy
from dataset import EmoticDataset
from loss import ContinuousLoss_L2, DiscreteLoss, ContinuousLoss_SL1
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, average_precision_score

class EmoticTrainer:
    def __init__(self, args):
        self.args = args

        self.set_device()
        self.train_dl, self.val_dl, self.num_train, self.num_val = self.prepare_dataloader()\
              if not args.pretraining else self.prepare_dataloader_for_pretraining()
        if self.args.wandb_log: self.set_wandb_config()
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
        wandb_config = vars(self.args)
        print("wandb_config:\n", wandb_config)

        wandb_name = self.args.wandb_name if self.args.wandb_name != None else self.args.model_name
        wandb.init(
            id=wandb_name+self.args.notes,
            entity="donghochoi",
            project=wandb_project,
            name=wandb_name,
            group=self.args.group,
            config=wandb_config,
            resume="allow",
            notes=self.args.notes
        )

    def prepare_dataloader(self):

        print("Pulling EMOTIC data...")

        data_path = "/home/dongho/brain2valence/data/emotic"

        print('Prepping train and validation dataloaders...')

        train_data, val_data, _ = utils.get_emotic_df()

        train_context_transform, train_body_transform, test_context_transform, test_body_transform =\
            utils.get_transforms_emotic()
        
        if self.args.coco_only:
            train_data = train_data[train_data['folder'] == 'mscoco/images']
            val_data = val_data[val_data['folder'] == 'mscoco/images']

        if self.args.with_nsd:
            self.subjects = [1, 2, 5, 7] if self.args.all_subjects else self.args.subj
            print(f"Do EMOTIC task sing NSD data given subject: {self.subjects}")
            emotic_data = utils.get_emotic_df(is_split=False)
            train_data = utils.get_emotic_coco_nsd_df(emotic_data=emotic_data, 
                                                      split='train', 
                                                      subjects=self.subjects)
            val_data = utils.get_emotic_coco_nsd_df(emotic_data=emotic_data, 
                                                      split='val', 
                                                      subjects=self.subjects)

        train_dataset = EmoticDataset(data_path=data_path,
                                      split='train',
                                      emotic_annotations=train_data,
                                      context_transform=train_context_transform,
                                      body_transform=train_body_transform,
                                      normalize=True,
                                      )

        val_dataset = EmoticDataset(data_path=data_path,
                                    split='val',
                                    emotic_annotations=val_data,
                                    context_transform=test_context_transform,
                                    body_transform=test_body_transform,
                                    normalize=True,
                                    )

        train_dl = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
        print('# train data:', len(train_dataset))
        print('# val data:', len(val_dataset))

        return train_dl, val_dl, len(train_dataset), len(val_dataset)
    
    def prepare_dataloader_for_pretraining(self):

        print("Pulling EMOTIC dataset for pretraining: Excludes images shown to subjects...")
        print('Prepping train and validation dataloaders...')

        data_path = "/home/dongho/brain2valence/data/emotic"
        self.subjects = [1, 2, 5, 7] if self.args.all_subjects else self.args.subj
        train_data, val_data, _ = utils.get_emotic_df_for_pretraining(subjects=self.subjects)
        train_context_transform, train_body_transform, test_context_transform, test_body_transform =\
            utils.get_transforms_emotic()
        
        train_dataset = EmoticDataset(data_path=data_path,
                                      split='train',
                                      emotic_annotations=train_data,
                                      context_transform=train_context_transform,
                                      body_transform=train_body_transform,
                                      normalize=True,
                                      )

        val_dataset = EmoticDataset(data_path=data_path,
                                    split='val',
                                    emotic_annotations=val_data,
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
        model = EmoticModel(
            image_backbone=self.args.image_backbone,
            image_model_type=self.args.model_type,
            pretrained=self.args.pretrained,
            wgt_path=self.args.wgt_path,
            backbone_freeze=self.args.backbone_freeze,
            cat_only=self.args.cat_only
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
        elif self.args.scheduler == "cycle":
            total_steps=int(self.args.epochs * (self.num_train // self.args.batch_size))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=self.args.lr,
                total_steps=total_steps,
                steps_per_epoch=self.num_train // self.args.batch_size,
                final_div_factor=1000,
                last_epoch=-1,
                pct_start=2/self.args.epochs
            )
        return scheduler

    def get_criterion(self):
        if self.args.cat_criterion == "emotic":
            disc = DiscreteLoss(weight_type='dynamic', device=self.device)
        elif self.args.cat_criterion == "softmargin":
            disc = nn.MultiLabelSoftMarginLoss(reduction='sum') # logit 을 넣어줘야 한다.
        else: raise NotImplementedError(f'criterion {self.args.cat_criterion} is not implemented')

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
        best_val_mAP = -float("inf")
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

                if self.args.cat_only:

                    pred_cat = self.model(body_image, context_image)

                    loss = self.disc_loss(pred_cat, gt_cat)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                else:
                    pred_cat, pred_vad = self.model(body_image, context_image)

                    loss_cat = self.disc_loss(pred_cat, gt_cat)
                    loss_vad = self.vad_loss(pred_vad, gt_vad)
                    loss = cat_loss_param * loss_cat + vad_loss_param * loss_vad

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

            train_loss /= self.num_train

            self.model.eval()
            val_loss = 0
            pred_cats = np.zeros((self.args.batch_size, 26))
            gt_cats = np.zeros((self.args.batch_size, 26))
            with torch.no_grad():
                for i, (context_image, body_image, valence, arousal, dominance, category) in tqdm(enumerate(self.val_dl)):

                    context_image = context_image.float().cuda()
                    body_image = body_image.float().cuda()
                    gt_cat = category.float().cuda()
                    gt_vad = torch.stack([valence, arousal, dominance], dim=1).float().cuda()
                    if self.args.cat_only:
                        pred_cat = self.model(body_image, context_image)
                        loss = self.disc_loss(pred_cat, gt_cat)

                        val_loss += loss.item()
                        pred_cats = np.vstack((pred_cats, pred_cat.cpu().numpy())) if not np.all(pred_cats == 0) else pred_cat.cpu().numpy()
                        gt_cats = np.vstack((gt_cats, gt_cat.cpu().numpy())) if not np.all(gt_cats == 0) else gt_cat.cpu().numpy()
                    else:   
                        pred_cat, pred_vad = self.model(body_image, context_image)
                        loss_cat = self.disc_loss(pred_cat, gt_cat)
                        loss_vad = self.vad_loss(pred_vad, gt_vad)

                        loss = cat_loss_param * loss_cat + vad_loss_param * loss_vad
                        val_loss += loss.item()
                
            val_loss /= self.num_val
            ap_scores = [average_precision_score(gt_cats[:, i], pred_cats[:, i]) for i in range(26)]
            mAP = np.mean(ap_scores)

            if self.args.wandb_log:
                wandb.log({"train_loss": train_loss,
                        "lr": self.optimizer.param_groups[0]['lr'],
                        "val_loss": val_loss,
                        "val_mAP": mAP
                        }, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print ('saving model at epoch: %d' %(epoch))
                if self.args.wandb_log:
                    wandb.log({"best_val_loss": best_val_loss}, step=epoch)
                self.save_model(self.args, self.model, best=True)

            if mAP > best_val_mAP:
                best_val_mAP = mAP
                if self.args.wandb_log:
                    wandb.log({"best_val_mAP": mAP}, step=epoch)
                
            self.scheduler.step()

            print("Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}".format(
                epoch, train_loss, val_loss))

        self.save_model(self.args, self.model, best=False)

        return self.model

    def save_model(self, args, model, best):
        model_name = args.model_name  # ex) "all_subjects_res18_mae_2"
        save_dir = os.path.join(args.save_path, model_name + args.notes)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model.cpu()
        if best:
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))

