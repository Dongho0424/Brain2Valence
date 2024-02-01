import wandb
import torch
import torch.nn as nn
import os
import utils
from model import Brain2ValenceModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Predictor:
    def __init__(self, args):
        self.args = args

        self.test_dl, self.num_test = self.prepare_dataloader()
        self.set_wandb_config()
        self.model: nn.Module = self.get_model()

    def set_wandb_config(self):
        wandb_project = self.args.wandb_project
        wandb_run = self.args.wandb_name
        
        print(f"wandb {wandb_project} run {wandb_run}")
        wandb.login(host='https://api.wandb.ai')
        wandb_config = {
            "model_name": self.args.model_name,    
            "batch_size": self.args.batch_size,
            "epochs": self.args.epochs,
            "num_test": self.num_test,
            "seed": self.args.seed,
        }
        print("wandb_config:\n",wandb_config)

        wandb.init(
            id = self.args.model_name,
            project=wandb_project,
            name=wandb_run,
            config=wandb_config,
            resume="allow",
        )
        
    def prepare_dataloader(self): 
        print("Pulling Brain and Valence pair data...")

        data_path="/home/juhyeon/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split"

        print('Prepping test dataloaders...')

        emotic_annotations = utils.get_emotic_data()
        nsd_df, target_cocoid = utils.get_NSD_data(emotic_annotations)

        test_dl, num_test = utils.get_torch_dataloaders(
            self.args.batch_size,
            data_path = data_path,
            emotic_annotations=emotic_annotations,
            nsd_df=nsd_df,
            target_cocoid=target_cocoid,
            mode='test',
            subjects=[1, 2, 5, 7]
        )

        self.test_dl = test_dl
        self.num_test = num_test

        return test_dl, num_test
    
    def get_model(self):
        model = Brain2ValenceModel()
        best_path = os.path.join(self.args.save_path, self.args.wandb_run_name, self.make_log_name(self.args) + "_best_model.pth")
        model = model.load_state_dict(torch.load(best_path))
        return model
    
    def predict(self, args):
        #### enter Predicting ###
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:', device)
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())
        #### enter Predicting ###

        self.model.cuda()
        self.model.eval()

        true_valences = []
        pred_valences = []

        with torch.no_grad():
            for i, (brain_3d, valence) in enumerate(self.test_dl):
                brain_3d = brain_3d.float().cuda()
                valence = valence.float().cuda()
                # valence 0~1로 normalize
                valence /= 10.0
                pred_valence = self.model(brain_3d)

                true_valences.append(valence)
                pred_valences.append(pred_valence.item())

        print("RMSE: {:.4f}, R squared: {:.4f}".format(np.sqrt(mean_squared_error(true_valences, pred_valences)), r2_score(true_valences, pred_valences)))

        # Plot true vs pred valence 
        plt.scatter(true_valences, pred_valences)
        plt.xlabel("True valence")
        plt.ylabel("Pred valence")
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
            
        wandb.log({"plot true valence vs pred valence": wandb.Image(plt)})
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