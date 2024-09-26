import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import utils
from model import BrainModel
from tqdm import tqdm
from dataset import BrainDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, average_precision_score

class CrossPredictor():
    def __init__(self, args):
        self.args = args

        self.set_device()
        self.prepare_dataloader()
        if self.args.wandb_log: self.set_wandb_config()
        self.load_model(args, self.args.best)     

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
        print("wandb_config:\n",wandb_config)
        wandb_name = self.args.wandb_name if self.args.wandb_name != None else self.args.model_name
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=wandb_config,
        )

    def get_single_dl(self, subj):

        _, _, test_context_transform, test_body_transform = utils.get_transforms_emotic()

        test_split = 'one_point' if self.args.one_point else 'test'
        test_dataset = BrainDataset(subjects=[subj],
                                    split=test_split,
                                    data_type=self.args.data,
                                    context_transform=test_context_transform,
                                    body_transform=test_body_transform,
                                    normalize=True,
                                    )

        # always batch size is 1
        test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        return test_dl, len(test_dataset)

    def prepare_dataloader(self): 
        
        print('Prepping multi-subject test dataloaders...')

        self.subjects = self.args.subj_tgt # for model backbone defining, not preparing test data

        print("Predicting for subjects:", self.subjects)
        
        self.test_dl, self.num_test = self.get_single_dl(self.subjects[0])
        print('# test data:', self.num_test)
    
    def load_model(self, args, use_best=True):
        model = BrainModel(
            image_backbone=self.args.image_backbone,
            image_model_type=self.args.model_type,
            brain_backbone="single_subj", # for predicting
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
        model.to(self.device)
        
        model_name = args.model_name # ex) "all_subjects_res18_mae_2"
        save_dir = os.path.join(args.save_path, model_name)
        if use_best:
            best_path = os.path.join(save_dir, "best_model.pth")
            model.load_state_dict(torch.load(best_path), strict=False)
        else:
            last_path = os.path.join(save_dir, "last_model.pth")
            model.load_state_dict(torch.load(last_path), strict=False)
        
        self.model = model

    def step(self, voxel, ctx_img, body_img, vad, cat):
        # to device
        ctx_img = ctx_img.to(self.device, torch.float)
        body_img = body_img.to(self.device, torch.float)
        cat = cat.to(self.device, torch.float)
        vad = vad.to(self.device, torch.float)
        voxel = voxel.to(self.device, torch.float)

        pred_cat = self.model(body_img, ctx_img, voxel, self.subjects)

        return pred_cat
    
    def predict(self):
        print("#### enter Predicting ####")
        print("#### enter Predicting ####")
        print("#### enter Predicting ####")

        self.model.eval()

        pred_cats = np.zeros((self.num_test, 26))
        gt_cats = np.zeros((self.num_test, 26))
        pred_vads = np.zeros((self.num_test, 3))
        gt_vads = np.zeros((self.num_test, 3))

        with torch.no_grad():
            for test_i, data in tqdm(enumerate(self.test_dl)):

                ctx_img, body_img, v, a, d, gt_cat, voxel = data
                gt_vad = torch.stack([v, a, d], dim=1)

                voxel = F.adaptive_max_pool1d(voxel.float(), self.args.pool_num)

                pred_cat = self.step(voxel, ctx_img, body_img, gt_vad, gt_cat)

                pred_cats[test_i, :] = pred_cat.cpu().numpy()
                gt_cats[test_i, :] = gt_cat.cpu().numpy()

        # evaluation for categorical emotion
        ap_scores = [average_precision_score(gt_cats[:, i], pred_cats[:, i]) for i in range(26)]
        mAP = np.mean(ap_scores)

        _, idx2cat = utils.get_emotic_categories()
        for i, ap in enumerate(ap_scores):
            print(f"AP for {i}. {idx2cat[i]}: {ap:.4f}")
        print(f"mAP: {mAP:.4f}")

        # plot AP per category
        plt.figure(figsize=(10, 8))
        plt.yscale('log')
        plt.xticks(range(26), [f"{i}. {idx2cat[i]}" for i in range(26)], rotation=-90)
        for i, ap in enumerate(ap_scores):
            plt.bar(i, ap)
            plt.text(i, ap, f'{ap:.4f}', ha='center', va='bottom')

        if self.args.wandb_log:
            wandb.log({"mAP": mAP,
                       "Average Precision per category": wandb.Image(plt)})
            
        plt.clf()

