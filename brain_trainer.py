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
from dataset import BrainDataset, BrainDataset2
from loss import ContinuousLoss_L2, DiscreteLoss, ContinuousLoss_SL1
from emotic_trainer import EmoticTrainer
from model import BrainModel
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, average_precision_score
import warnings
import h5py

class BrainTrainer(EmoticTrainer):
    def __init__(self, args):
        super().__init__(args)

            
    def prepare_dataloader(self):

        print('Prepping train and validation dataloaders...')

        self.subjects = self.args.subj

        train_context_transform, train_body_transform, test_context_transform, test_body_transform =\
            utils.get_transforms_emotic()

        if self.args.dataset_ver == 2:
            train_dataset = BrainDataset2(subjects=self.subjects,
                                          split='train',
                                          data_type=self.args.data,
                                          context_transform=train_context_transform,
                                          body_transform=train_body_transform,
                                          normalize=True,
                                          )
            
            val_dataset = BrainDataset2(subjects=self.subjects,
                                        split='val',
                                        data_type=self.args.data,
                                        context_transform=test_context_transform,
                                        body_transform=test_body_transform,
                                        normalize=True,
                                        )
            self.voxels = {}
            self.num_voxels = {}
            self.voxel_means = {}
            self.voxel_stds = {}

            basedir = '/home/juhyeon/Brain2Valence'

            for s in self.subjects:
                if self.args.data == 'roi':
                    betas = np.load(os.path.join(basedir, f'vis_subj{s}_all_beta.npy'))
                    betas = torch.tensor(betas).to("cpu").to(torch.float16)
                    mean = np.load(os.path.join(basedir, f'vis_subj{s}_train_beta_mean.npy'))
                    std = np.load(os.path.join(basedir, f'vis_subj{s}_train_beta_std.npy'))
                elif self.args.data == 'emo_roi':
                    betas = np.load(os.path.join(basedir, f'emo_subj{s}_all_beta.npy'))
                    betas = torch.tensor(betas).to("cpu").to(torch.float16)
                    mean = np.load(os.path.join(basedir, f'emo_subj{s}_train_beta_mean.npy'))
                    std = np.load(os.path.join(basedir, f'emo_subj{s}_train_beta_std.npy'))
                elif self.args.data == 'emo_vis_roi':
                    betas = np.load(os.path.join(basedir, f'emo_vis_subj{s}_all_beta.npy'))
                    betas = torch.tensor(betas).to("cpu").to(torch.float16)
                    mean = np.load(os.path.join(basedir, f'emo_vis_subj{s}_train_beta_mean.npy'))
                    std = np.load(os.path.join(basedir, f'emo_vis_subj{s}_train_beta_std.npy'))
                    
                self.num_voxels[f'subj0{s}'] = betas.shape[1]
                self.voxels[f'subj0{s}'] = betas
                self.voxel_means[f'subj0{s}'] = mean
                self.voxel_stds[f'subj0{s}'] = std

        elif self.args.dataset_ver == 1:
            train_split = 'one_point' if self.args.one_point else 'train'
            train_dataset = BrainDataset(subjects=self.subjects,
                                        split=train_split,
                                        data_type=self.args.data,
                                        context_transform=train_context_transform,
                                        body_transform=train_body_transform,
                                        normalize=True,
                                        )
            val_split = 'one_point' if self.args.one_point else 'val'
            val_dataset = BrainDataset(subjects=self.subjects,
                                    split=val_split,
                                    data_type=self.args.data,
                                    context_transform=test_context_transform,
                                    body_transform=test_body_transform,
                                    normalize=True,
                                    )
        else: raise ValueError('dataset_ver should be 1 or 2, who are you?')

        train_dl = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
        print('# train data:', len(train_dataset))
        print('# val data:', len(val_dataset))

        return train_dl, val_dl, len(train_dataset), len(val_dataset)
    
    def get_model(self):
        model = BrainModel(
            image_backbone=self.args.image_backbone,
            image_model_type=self.args.model_type,
            brain_backbone=self.args.brain_backbone,
            brain_data_type=self.args.data,
            brain_in_dim=self.args.pool_num,
            brain_out_dim=512,
            subjects=self.subjects,
            backbone_freeze=self.args.backbone_freeze,
            pretrained=self.args.pretrained,
            wgt_path=self.args.wgt_path,
            cat_only=self.args.cat_only,
            fusion_ver=self.args.fusion_ver
        )

        utils.print_model_info(model)
        self.max_pool = nn.AdaptiveMaxPool1d(self.args.pool_num) 

        return model
    
    def train(self):
        print("#### enter Training ####")
        print("#### enter Training ####")
        print("#### enter Training ####")

        best_val_loss = float("inf")
        best_val_ap_mean = -float("inf")
        # now set equal
        cat_loss_param = 0.5
        vad_loss_param = 0.5

        for epoch in range(self.args.epochs):
            self.model.cuda()
            self.model.train()
            train_loss = 0

            for i, (context_image, body_image, valence, arousal, dominance, category, brain_data) in tqdm(enumerate(self.train_dl)):

                context_image = context_image.float().cuda()
                body_image = body_image.float().cuda()
                gt_cat = category.float().cuda()
                gt_vad = torch.stack([valence, arousal, dominance], dim=1).float().cuda()
                
                if self.args.dataset_ver == 2:
                    subj, beta_idx = brain_data
                    brain_data = []
                    for s, b in zip(subj, beta_idx):
                        v = torch.tensor(self.voxels[s][b]).unsqueeze(0).float().cuda()
                        v = (v - torch.tensor(self.voxel_means[s]).float().cuda()) / torch.tensor(self.voxel_stds[s]).float().cuda()
                        brain_data.append(self.max_pool(v))
                    brain_data = torch.stack(brain_data).cuda().squeeze(1)
                elif self.args.dataset_ver == 1:
                    brain_data = brain_data.float().cuda()
                
                if self.args.cat_only: # only category
                    # brain_data: brain3d or roi
                    pred_cat= self.model(body_image, context_image, brain_data)

                    loss = self.disc_loss(pred_cat, gt_cat)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()    
                else: # both category and vad 

                    # brain_data: brain3d or roi
                    pred_cat, pred_vad = self.model(body_image, context_image, brain_data)

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
            pred_cats = np.zeros((self.args.batch_size, 26))
            gt_cats = np.zeros((self.args.batch_size, 26))
            with torch.no_grad():
                for i, (context_image, body_image, valence, arousal, dominance, category, brain_data) in tqdm(enumerate(self.val_dl)):

                    context_image = context_image.float().cuda()
                    body_image = body_image.float().cuda()
                    gt_cat = category.float().cuda()
                    gt_vad = torch.stack([valence, arousal, dominance], dim=1).float().cuda()
                    
                    if self.args.dataset_ver == 2:
                        subj, beta_idx = brain_data
                        brain_data = []
                        for s, b in zip(subj, beta_idx):
                            v = torch.tensor(self.voxels[s][b]).unsqueeze(0).float().cuda()
                            v = (v - torch.tensor(self.voxel_means[s]).float().cuda()) / torch.tensor(self.voxel_stds[s]).float().cuda()
                            brain_data.append(self.max_pool(v))
                        brain_data = torch.stack(brain_data).cuda().squeeze(1)
                    elif self.args.dataset_ver == 1:
                        brain_data = brain_data.float().cuda()
                    
                    if self.args.cat_only: # only category
                        # brain_data: brain3d or roi
                        pred_cat= self.model(body_image, context_image, brain_data)

                        loss = self.disc_loss(pred_cat, gt_cat)

                        val_loss += loss.item()    

                        pred_cats = np.vstack((pred_cats, pred_cat.cpu().numpy())) if not np.all(pred_cats == 0) else pred_cat.cpu().numpy()
                        gt_cats = np.vstack((gt_cats, gt_cat.cpu().numpy())) if not np.all(gt_cats == 0) else gt_cat.cpu().numpy()

                    else: # both category and vad
                        pred_cat, pred_vad = self.model(body_image, context_image, brain_data)
                        loss_cat = self.disc_loss(pred_cat, gt_cat)
                        loss_vad = self.vad_loss(pred_vad, gt_vad)

                        loss = cat_loss_param * loss_cat + vad_loss_param * loss_vad
                        val_loss += loss.item()
                
                val_loss /= self.num_val
                wandb.log({"val_loss": val_loss}, step=epoch)

                # evaluation for categorical emotion

                # Filter out the specific UserWarning
                warnings.filterwarnings("ignore", message="No positive class found in y_true, recall is set to one for all thresholds")

                ap_scores = [average_precision_score(gt_cats[:, i], pred_cats[:, i]) for i in range(26)]
                ap_mean = np.mean(ap_scores)
                wandb.log({"val_AP_mean": ap_mean}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print ('saving model at epoch: %d' %(epoch))
                wandb.log({"best_val_loss": best_val_loss}, step=epoch)
                self.save_model(self.args, self.model, best=True)

            if ap_mean > best_val_ap_mean:
                best_val_ap_mean = ap_mean
                wandb.log({"best_val_AP_mean": ap_mean}, step=epoch)

            self.scheduler.step()

            print("Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}".format(epoch, train_loss, val_loss))

        self.save_model(self.args, self.model, best=False)
        # wandb.log({"best_val_loss": best_val_loss})
        return self.model