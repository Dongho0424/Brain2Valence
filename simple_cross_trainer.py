import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import utils
from torchvision.transforms import v2
from tqdm import tqdm
import pandas as pd
import scipy
from dataset import BrainDataset
from loss import ContinuousLoss_L2, DiscreteLoss, ContinuousLoss_SL1
from model import BrainModel
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, average_precision_score
import warnings

class SimpleCrossTrainer:
    def __init__(self, args):
        self.args = args

        self.set_device()
        self.prepare_dataloader()
        if self.args.wandb_log:
            self.set_wandb_config()
        self.get_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.cat_loss, _ = self.get_criterion()

    def set_device(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
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

    def get_single_dl(self, subj, batch_size, split):

        train_context_transform, train_body_transform, test_context_transform, test_body_transform =\
            utils.get_transforms_emotic()

        split = 'one_point' if self.args.one_point else split
        ctx_transform = train_context_transform if split == 'train' else test_context_transform
        body_transform = train_body_transform if split == 'train' else test_body_transform
        shuffle = (split == 'train')

        dataset = BrainDataset(subjects=[subj],
                               split=split,
                               data_type=self.args.data,
                               context_transform=ctx_transform,
                               body_transform=body_transform,
                               normalize=True,
                               )

        dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dl, len(dataset)

    def prepare_dataloader(self):

        print('Prepping multi-subject train and validation dataloaders...')

        self.subjects = self.args.subj
        print("Training for subjects:", self.subjects)

        train_dls = []
        num_train_total = 0
        val_dls = []
        num_val_total = 0

        # dividing batch size by subj_list, which will then be concatenated across subj during training...
        batch_size = self.args.batch_size // len(self.subjects)
        print("batch_size per subject:", batch_size)
        print("total batch_size:", self.args.batch_size)
        
        for subj in self.subjects:
            train_dl, num_train = self.get_single_dl(subj, batch_size, 'train')
            train_dls.append(train_dl)
            num_train_total += num_train

        # When validation, we use just single subject's validation set as self.subjects[0]
        # self.val_subj = self.subjects[0]
        # val_dl, num_val = self.get_single_dl(self.val_subj, batch_size, 'val')

        for subj in self.subjects:
            val_dl, num_val = self.get_single_dl(subj, batch_size, 'val')
            val_dls.append(val_dl)
            num_val_total += num_val

        print('# train data:', num_train_total)
        # print('# val data:', num_val)
        print('# val data:', num_val_total)

        self.train_dls = train_dls
        self.num_train = num_train_total

        # self.val_dl = val_dl
        # self.num_val = num_val
        self.val_dls = val_dls
        self.num_val = num_val_total

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
        self.model = model

    def get_criterion(self):
        if self.args.cat_criterion == "emotic":
            cat_loss = DiscreteLoss(weight_type='dynamic', device=self.device)
        elif self.args.cat_criterion == "softmargin":
            # logit 을 넣어줘야 한다.
            cat_loss = nn.MultiLabelSoftMarginLoss(reduction='sum')
        else:
            raise NotImplementedError(
                f'criterion {self.args.cat_criterion} is not implemented')

        if self.args.criterion == "emotic_L2":
            vad_loss = ContinuousLoss_L2(margin=0.1)
        elif self.args.criterion == "emotic_SL1":
            vad_loss = ContinuousLoss_SL1(margin=0.1)
        else:
            raise NotImplementedError(
                f'criterion {self.args.criterion} is not implemented')

        return cat_loss, vad_loss

    def get_optimizer(self):
        params = self.model.parameters()

        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer

    def get_scheduler(self):
        if self.args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=7, gamma=0.1)
        elif self.args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == "cycle":
            total_steps = int(self.args.epochs * (self.num_train // self.args.batch_size))
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

    def step(self, voxel, ctx_img, body_img, vad, cat, subj_list):
        # to device
        ctx_img = ctx_img.to(self.device, torch.float)
        body_img = body_img.to(self.device, torch.float)
        cat = cat.to(self.device, torch.float)
        vad = vad.to(self.device, torch.float)
        voxel = voxel.to(self.device, torch.float)

        pred_cat = self.model(body_img, ctx_img, voxel, subj_list)

        cat_loss = self.cat_loss(pred_cat, cat)
        
        return cat_loss, pred_cat

    def train_epoch(self, epoch):
        self.model.to(device=self.device)
        self.model.train()

        train_loss = 0.
        for train_i, data in tqdm(enumerate(zip(*self.train_dls))):
            # repeat_index = train_i % 3 # randomly choose the one in the repeated three
            # TODO: 이미지 하나 당 brain_data가 3개씩인데,
            # random하게 1개 선택하지 말고 이미지 하나 당 3개의 brain_data를 모두 사용하도록 하면 사실상 3배 data augmentation 아닌가?

            voxel_list, ctx_img_list, body_img_list, vad_list, cat_list = [], [], [], [], []
            for ctx_img, body_img, v, a, d, cat, voxel in data:
                vad = torch.stack([v, a, d], dim=1)

                # adaptive max pool
                voxel = F.adaptive_max_pool1d(voxel.float(), self.args.pool_num)

                voxel_list.append(voxel)
                ctx_img_list.append(ctx_img)
                body_img_list.append(body_img)
                vad_list.append(vad)
                cat_list.append(cat)
            
            voxel = torch.cat(voxel_list, dim=0)
            ctx_img = torch.cat(ctx_img_list, dim=0)
            body_img = torch.cat(body_img_list, dim=0)
            vad = torch.cat(vad_list, dim=0)
            cat = torch.cat(cat_list, dim=0)

            loss, _ = self.step(voxel, ctx_img, body_img, vad, cat, self.subjects)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss

    def single_subj_eval_epoch(self, epoch):
        self.model.eval()

        val_loss = 0.
        pred_cats = np.zeros((self.args.batch_size, 26))
        gt_cats = np.zeros((self.args.batch_size, 26))
        with torch.no_grad():
            for val_i, data in tqdm(enumerate(self.val_dl)):
                # repeat_index = train_i % 3 # randomly choose the one in the repeated three
                # TODO: 이미지 하나 당 brain_data가 3개씩인데,
                # random하게 1개 선택하지 말고 이미지 하나 당 3개의 brain_data를 모두 사용하도록 하면 사실상 3배 data augmentation 아닌가?
                ctx_img, body_img, v, a, d, gt_cat, voxel = data
                vad = torch.stack([v, a, d], dim=1)
                voxel = F.adaptive_max_pool1d(voxel.float(), self.args.pool_num)
                
                loss, pred_cat = self.step(voxel, ctx_img, body_img, vad, gt_cat, [self.val_subj])
                val_loss += loss.item()
                
                pred_cats = np.vstack((pred_cats, pred_cat.cpu().numpy())) if not np.all(
                    pred_cats == 0) else pred_cat.cpu().numpy()
                gt_cats = np.vstack((gt_cats, gt_cat.cpu().numpy())) if not np.all(
                    gt_cats == 0) else gt_cat.cpu().numpy()

        ap_scores = [average_precision_score(gt_cats[:, i], pred_cats[:, i]) for i in range(26)]
        val_mAP = np.mean(ap_scores)

        return val_loss, val_mAP

    def eval_epoch(self, epoch):
        self.model.eval()

        val_loss = 0.
        pred_cats = np.zeros((self.args.batch_size, 26))
        gt_cats = np.zeros((self.args.batch_size, 26))
        with torch.no_grad():
            for val_i, data in tqdm(enumerate(zip(*self.val_dls))):
                # repeat_index = val_i % 3 # randomly choose the one in the repeated three
                # TODO: 이미지 하나 당 brain_data가 3개씩인데,
                # random하게 1개 선택하지 말고 이미지 하나 당 3개의 brain_data를 모두 사용하도록 하면 사실상 3배 data augmentation 아닌가?

                voxel_list, ctx_img_list, body_img_list, vad_list, cat_list = [], [], [], [], []
                for ctx_img, body_img, v, a, d, gt_cat, voxel in data:
                    vad = torch.stack([v, a, d], dim=1)

                    # adaptive max pool
                    voxel = F.adaptive_max_pool1d(voxel.float(), self.args.pool_num)

                    voxel_list.append(voxel)
                    ctx_img_list.append(ctx_img)
                    body_img_list.append(body_img)
                    vad_list.append(vad)
                    cat_list.append(gt_cat)
                
                voxel = torch.cat(voxel_list, dim=0)
                ctx_img = torch.cat(ctx_img_list, dim=0)
                body_img = torch.cat(body_img_list, dim=0)
                vad = torch.cat(vad_list, dim=0)
                gt_cat = torch.cat(cat_list, dim=0)

                loss, pred_cat = self.step(voxel, ctx_img, body_img, vad, gt_cat, self.subjects)
                val_loss += loss.item()

                pred_cats = np.vstack((pred_cats, pred_cat.cpu().numpy())) if not np.all(
                    pred_cats == 0) else pred_cat.cpu().numpy()
                gt_cats = np.vstack((gt_cats, gt_cat.cpu().numpy())) if not np.all(
                    gt_cats == 0) else gt_cat.cpu().numpy()
            
        ap_scores = [average_precision_score(gt_cats[:, i], pred_cats[:, i]) for i in range(26)]
        val_mAP = np.mean(ap_scores)

        return val_loss, val_mAP

    def train(self):
        print("#### enter Training ####")
        print("#### enter Training ####")
        print("#### enter Training ####")

        best_val_loss = float("inf")
        best_val_mAP = -float("inf")

        # Filter out the specific UserWarning
        warnings.filterwarnings(
            "ignore", message="No positive class found in y_true, recall is set to one for all thresholds")

        for epoch in range(self.args.epochs):
            ## Train ##
            train_loss = self.train_epoch(epoch)
            train_loss /= self.num_train

            ## Eval ##
            val_loss, val_mAP = self.eval_epoch(epoch)
            val_loss /= self.num_val

            ## Wandb Log & save models ##
            if self.args.wandb_log:
                wandb.log(
                    {"train_loss": train_loss,
                     "lr": self.optimizer.param_groups[0]['lr'],
                     "val_loss": val_loss,
                     "val_mAP": val_mAP}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print('saving model at epoch: %d' % (epoch))
                if self.args.wandb_log:
                    wandb.log({"best_val_loss": best_val_loss}, step=epoch)
                self.save_model(self.args, self.model, best=True)

            if val_mAP > best_val_mAP:
                best_val_mAP = val_mAP
                if self.args.wandb_log:
                    wandb.log({"best_val_mAP": val_mAP}, step=epoch)

            self.scheduler.step()

            print("Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}".format(epoch, train_loss, val_loss))

        self.save_model(self.args, self.model, best=False)

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

