import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import utils
from model import Image2VADModel
from tqdm import tqdm
from dataset import EmoticDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, average_precision_score

class EmoticPredictor:
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
            "model": self.args.model,
            "model_name": self.args.model_name,    
            "batch_size": self.args.batch_size,
            "epochs": self.args.epochs,
            "num_test": self.num_test,
            "seed": self.args.seed,
            "weight_decay": self.args.weight_decay,
        }
        print("wandb_config:\n",wandb_config)

        wandb.init(
            id = self.args.wandb_name,
            project=wandb_project,
            name=self.args.wandb_name,
            config=wandb_config,
            resume="allow",
        )
        
    def prepare_dataloader(self): 
        print("Pulling EMOTIC data...")

        data_path = "/home/dongho/brain2valence/data/emotic"

        print('Prepping test dataloaders...')

        _, _, test_data = utils.get_emotic_df()

        _, _, test_context_transform, test_body_transform =\
            utils.get_transforms_emotic()

        test_dataset = EmoticDataset(data_path=data_path,
                                    split='test',
                                    emotic_annotations=test_data,
                                    model_type="B",
                                    context_transform=test_context_transform,
                                    body_transform=test_body_transform,
                                    normalize=True,
                                    )

        # always batch size is 1
        test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print('# test data:', len(test_dataset))

        return test_dl, len(test_dataset)
    
    def load_model(self, args, use_best=True) -> nn.Module :
        model = Image2VADModel(
            backbone=self.args.model,
            model_type=self.args.model_type,
            pretrained=self.args.pretrain,
            backbone_freeze=self.args.backbone_freeze,
        )
        
        model_name = args.model_name # ex) "all_subjects_res18_mae_2"

        if use_best:
            best_path = os.path.join(self.args.save_path, model_name, "best_model.pth")
            model.load_state_dict(torch.load(best_path))
        else:
            last_path = os.path.join(self.args.save_path, model_name, "last_model.pth")
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

        pred_cat_tensor = torch.Tensor().cuda()
        gt_cat_tensor = torch.Tensor().cuda()

        pred_vad_tensor = torch.Tensor().cuda()
        gt_vad_tensor = torch.Tensor().cuda()

        with torch.no_grad():
            for i, (context_image, body_image, valence, arousal, dominance, category) in tqdm(enumerate(self.test_dl)):

                context_image = context_image.float().cuda()
                body_image = body_image.float().cuda()
                gt_cat = category.float().cuda()
                gt_vad = torch.stack([valence, arousal, dominance], dim=1).float().cuda()
                pred_cat, pred_vad = self.model(body_image, context_image) # (1, 26), (1, 3)

                pred_cat_tensor = torch.cat((pred_cat_tensor, pred_cat), 0)
                gt_cat_tensor = torch.cat((gt_cat_tensor, gt_cat), 0)

                pred_vad_tensor = torch.cat((pred_vad_tensor, pred_vad), 0)
                gt_vad_tensor = torch.cat((gt_vad_tensor, gt_vad), 0)

        # evaluation for categorical emotion
        ap = self.calculate_AP_score(pred_cat_tensor.cpu().numpy(), gt_cat_tensor.cpu().numpy())
        ap_mean = np.mean(ap)
        ap_dict = {f"AP_class_{i+1}": ap for i, ap in enumerate(ap)}

        wandb.log({"AP": ap_dict, "AP_mean": ap_mean})
        
        # evalutation for VAD
        vad_mae = F.l1_loss(pred_vad_tensor, gt_vad_tensor, reduction='none')
        v_mae = vad_mae[:, 0].mean().item()
        a_mae = vad_mae[:, 1].mean().item()
        d_mae = vad_mae[:, 2].mean().item()
        total_mae = vad_mae.mean().item()

        v_corr = r2_score(gt_vad_tensor[:, 0].cpu().numpy(), pred_vad_tensor[:, 0].cpu().numpy())
        a_corr = r2_score(gt_vad_tensor[:, 1].cpu().numpy(), pred_vad_tensor[:, 1].cpu().numpy())
        d_corr = r2_score(gt_vad_tensor[:, 2].cpu().numpy(), pred_vad_tensor[:, 2].cpu().numpy())

        print("valence mae: {:.4f}, arousal mae: {:.4f}, dominance mae: {:.4f}, total mae: {:.4f}".format(v_mae, a_mae, d_mae, total_mae))
        print("valence corr: {:.4f}, arousal corr: {:.4f}, dominance corr: {:.4f} ".format(v_corr, a_corr, d_corr))
        
        wandb.log({"valence_mae": v_mae, "arousal_mae": a_mae, "dominance_mae": d_mae, "total_mae": total_mae})
        wandb.log({"valence_corr": v_corr, "arousal_corr": a_corr, "dominance_corr": d_corr})

        for index, vad in enumerate(['valence', 'arousal', 'dominance']):
            # Plot true vs pred valence 
            plt.scatter(gt_vad_tensor[:, index].cpu().numpy(), pred_vad_tensor[:, index].cpu().numpy(), alpha=0.2)
            plt.xlabel(f"True {vad}")
            plt.ylabel(f"Pred {vad}")
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                    
            wandb.log({f"plot true {vad} vs pred {vad}": wandb.Image(plt)})
            plt.clf()
    
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
    
    def calculate_AP_score(self, cat_preds, cat_labels):
        ap = [average_precision_score(cat_labels[i, :], cat_preds[i, :]) for i in range(26)]
        return ap