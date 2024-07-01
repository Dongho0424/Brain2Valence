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

class BrainPredictor():
    def __init__(self, args):
        self.args = args

        self.test_dl, self.num_test = self.prepare_dataloader()
        self.set_wandb_config()
        self.model: nn.Module = self.load_model(args, use_best=self.args.best)     

    def set_wandb_config(self):
        wandb_project = self.args.wandb_project
        model_name = self.args.model_name
        print(f"wandb {wandb_project} run {model_name}")
        wandb.login(host='https://api.wandb.ai')
        wandb_config = {
            "model_name": self.args.model_name,
            "image_backbone": self.args.image_backbone,
            "brain_backbone": self.args.brain_backbone,
            "batch_size": self.args.batch_size,
            "epochs": self.args.epochs,
            "num_test": self.num_test,
            "seed": self.args.seed,
            "weight_decay": self.args.weight_decay,
        }
        print("wandb_config:\n",wandb_config)

        wandb.init(
            id=self.args.model_name+self.args.notes,
            project=wandb_project,
            name=self.args.model_name,
            group=self.args.group,
            config=wandb_config,
            resume="allow",
            notes=self.args.notes
        )

    def prepare_dataloader(self): 
        
        print('Prepping test dataloaders...')

        self.subjects = [1, 2, 5, 7] if self.args.all_subjects else [self.args.subj]

        _, _, test_context_transform, test_body_transform = utils.get_transforms_emotic()

        test_dataset = BrainDataset(subjects=self.subjects,
                                    split='test',
                                    data_type=self.args.data,
                                    context_transform=test_context_transform,
                                    body_transform=test_body_transform,
                                    normalize=True,
                                    )

        # always batch size is 1
        test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print('# test data:', len(test_dataset))

        return test_dl, len(test_dataset)
    
    def load_model(self, args, use_best=True) -> nn.Module :
        model = BrainModel(
            image_backbone=self.args.image_backbone,
            image_model_type=self.args.model_type,
            brain_backbone=self.args.brain_backbone,
            brain_data_type=self.args.data,
            subjects=self.subjects,
            backbone_freeze=True,
            cat_only=self.args.cat_only,
            fusion_ver=self.args.fusion_ver
        )
        
        model_name = args.model_name # ex) "all_subjects_res18_mae_2"
        save_dir = os.path.join(args.save_path, model_name + args.notes)
        if use_best:
            best_path = os.path.join(save_dir, "best_model.pth")
            model.load_state_dict(torch.load(best_path))
        else:
            last_path = os.path.join(save_dir, "last_model.pth")
            model.load_state_dict(torch.load(last_path))
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
                brain_data = brain_data.float().cuda()

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
        ap_mean = np.mean(ap_scores)

        wandb.log({"AP_mean": ap_mean})
        _, idx2cat = utils.get_emotic_categories()
        for i, ap in enumerate(ap_scores):
            print(f"AP for {i}. {idx2cat[i]}: {ap:.4f}")

        # plot AP per category
        plt.figure(figsize=(10, 8))
        plt.title('Average Prevision per category')
        plt.yscale('log')
        plt.xticks(range(26), [f"{i}. {idx2cat[i]}" for i in range(26)], rotation=-90)
        for i, ap in enumerate(ap_scores):
            plt.bar(i, ap)
            plt.text(i, ap, f'{ap:.4f}', ha='center', va='bottom')
        wandb.log({f"Average Prevision per category": wandb.Image(plt)})
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
        
