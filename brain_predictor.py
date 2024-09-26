import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import utils
from model import BrainModel
from tqdm import tqdm
from dataset import BrainDataset, BrainAllDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, average_precision_score
import h5py

class BrainPredictor():
    def __init__(self, args):
        self.args = args

        self.test_dl, self.num_test = self.prepare_dataloader()
        if self.args.wandb_log: self.set_wandb_config()
        self.model: nn.Module = self.load_model(args, use_best=self.args.best)     

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

    def prepare_dataloader(self): 
        
        print('Prepping test dataloaders...')

        self.subjects = [1, 2, 5, 7] if self.args.all_subjects else self.args.subj

        _, _, test_context_transform, test_body_transform = utils.get_transforms_emotic()
        
        if self.args.all_subjects:
            test_dataset = BrainAllDataset(
                split ='test',
                data_type=self.args.data,
                context_transform=test_context_transform,
                body_transform=test_body_transform,
                normalize=True,
            )
        else:
            test_split = 'one_point' if self.args.one_point else 'test'
            test_dataset = BrainDataset(subjects=self.subjects,
                                        split=test_split,
                                        data_type=self.args.data,
                                        context_transform=test_context_transform,
                                        body_transform=test_body_transform,
                                        normalize=True,
                                        )

        # always batch size is 1
        test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print('# test data:', len(test_dataset))
        
        
                
        if self.args.all_subjects:
            self.voxels = {}
            self.num_voxels = {}

            for s in range(1, 9):
                f = h5py.File(f'/home/data/mindeyev2/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
                betas = f['betas'][:]
                betas = torch.Tensor(betas).to("cpu").to(torch.float16)
                print(betas.shape)
                self.num_voxels[f'subj0{s}'] = betas.shape[1]
                self.voxels[f'subj0{s}'] = betas
                
            print(self.voxels.keys())

            print('Loaded all subjects')

        return test_dl, len(test_dataset)
    
    def load_model(self, args, use_best=True) -> nn.Module :
        model = BrainModel(
            image_backbone=self.args.image_backbone,
            image_model_type=self.args.model_type,
            brain_backbone=self.args.brain_backbone,
            brain_data_type=self.args.data,
            brain_in_dim=self.args.pool_num,
            pretrained=self.args.pretrained,
            wgt_path=self.args.wgt_path,
            subjects=self.subjects,
            backbone_freeze=self.args.backbone_freeze,
            cat_only=self.args.cat_only,
            fusion_ver=self.args.fusion_ver
        )
        
        model_name = args.model_name # ex) "all_subjects_res18_mae_2"
        save_dir = os.path.join(args.save_path, model_name)
        if use_best:
            best_path = os.path.join(save_dir, "best_model.pth")
            print(f"Loading best model from {best_path}")
            model.load_state_dict(torch.load(best_path))
        else:
            last_path = os.path.join(save_dir, "last_model.pth")
            model.load_state_dict(torch.load(last_path))
            
        self.max_pool = nn.AdaptiveMaxPool1d(self.args.pool_num) 

        return model
    
    def predict(self):
        #### enter Predicting ###
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:', device)
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())
        #### enter Predicting ###

        self.model.cuda()
        self.model.eval()

        pred_cats = np.zeros((self.num_test, 26))
        gt_cats = np.zeros((self.num_test, 26))
        pred_vads = np.zeros((self.num_test, 3))
        gt_vads = np.zeros((self.num_test, 3))

        with torch.no_grad():
            for i, (context_image, body_image, valence, arousal, dominance, category, brain_data) in tqdm(enumerate(self.test_dl)):

                context_image = context_image.float().cuda()
                body_image = body_image.float().cuda()
                gt_cat = category.float().cuda()
                gt_vad = torch.stack([valence, arousal, dominance], dim=1).float().cuda()
                
                if not self.args.all_subjects:
                    brain_data = brain_data.float().cuda()
                else:
                    subj, beta_idx= brain_data
                    brain_data = []
                    for s,b in zip(subj, beta_idx):
                        brain_data.append(self.max_pool(torch.tensor(self.voxels[s][b]).unsqueeze(0).float().cuda()))
                    brain_data = torch.stack(brain_data).cuda().squeeze(1) # FIXME: max pool 하고 model에 넘기는 걸로 변경만 하면 될 듯
            
                if self.args.cat_only: # only category
                    pred_cat = self.model(body_image, context_image, brain_data)

                    pred_cats[i, :] = pred_cat.cpu().numpy()
                    gt_cats[i, :] = gt_cat.cpu().numpy()

                else: # category + VAD
                    pred_cat, pred_vad = self.model(body_image, context_image, brain_data) # (1, 26), (1, 3)

                    pred_cats[i, :] = pred_cat.cpu().numpy()
                    gt_cats[i, :] = gt_cat.cpu().numpy()
                    pred_vads[i, :] = pred_vad.cpu().numpy()
                    gt_vads[i, :] = gt_vad.cpu().numpy()

        # evaluation for categorical emotion
        ap_scores = [average_precision_score(gt_cats[:, i], pred_cats[:, i]) for i in range(26)]
        mAP = np.mean(ap_scores)

        _, idx2cat = utils.get_emotic_categories()
        for i, ap in enumerate(ap_scores):
            print(f"AP for {i}. {idx2cat[i]}: {ap:.4f}")
        print("mAP: {:.4f}".format(mAP))

        # plot AP per category
        plt.figure(figsize=(10, 8))
        plt.title('Average Precision per category')
        plt.yscale('log')
        plt.xticks(range(26), [f"{i}. {idx2cat[i]}" for i in range(26)], rotation=-90)
        for i, ap in enumerate(ap_scores):
            plt.bar(i, ap)
            plt.text(i, ap, f'{ap:.4f}', ha='center', va='bottom')
        if self.args.wandb_log:
            wandb.log({"mAP": mAP,
                       "Average Precision per category": wandb.Image(plt)})
        plt.clf()

        if not self.args.cat_only: # when predict vad only
            # evalutation for VAD
            vad_mae = [np.mean(np.abs(pred_vads[:, i] - gt_vads[:, i])) for i in range(3)]
            v_mae = vad_mae[0]
            a_mae = vad_mae[1]
            d_mae = vad_mae[2]
            total_mae = np.mean(vad_mae)

            v_corr = r2_score(gt_vads[:, 0], pred_vads[:, 0])
            a_corr = r2_score(gt_vads[:, 1], pred_vads[:, 1])
            d_corr = r2_score(gt_vads[:, 2], pred_vads[:, 2])

            print("valence mae: {:.4f}, arousal mae: {:.4f}, dominance mae: {:.4f}, total mae: {:.4f}".format(v_mae, a_mae, d_mae, total_mae))
            print("valence corr: {:.4f}, arousal corr: {:.4f}, dominance corr: {:.4f} ".format(v_corr, a_corr, d_corr))
            
            wandb.log({"valence_mae": v_mae, "arousal_mae": a_mae, "dominance_mae": d_mae, "total_mae": total_mae})
            wandb.log({"valence_corr": v_corr, "arousal_corr": a_corr, "dominance_corr": d_corr})

            for index, vad in enumerate(['valence', 'arousal', 'dominance']):
                # Plot true vs pred valence 
                plt.scatter(gt_vads[:, index], pred_vads[:, index], alpha=0.2)
                plt.xlabel(f"True {vad}")
                plt.ylabel(f"Pred {vad}")
                plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                        
                wandb.log({f"plot true {vad} vs pred {vad}": wandb.Image(plt)})
                plt.clf()
        
