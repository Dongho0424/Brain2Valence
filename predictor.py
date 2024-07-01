import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import os
import utils
from model import Brain2ValenceModel, Image2VADModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, average_precision_score

# \TODO: 1. ROI 추출한 15000개짜리 시그널에 MLP 붙여서 똑같이 regression
# - subject-wise: subject1만 해보기
# \TODO: 2. Brainformer로 참고해서 regression
# - subject-wise: subject1만 해보기
# 그래도 안되면???
# \TODO: 3. valence를 3(0~4,4~7,7~10) or 5구간으로 나눠서 classification
# dataloader에서 각 구간에 대해서 고르게 class를 뽑아주는 설정해서! weighted_random sampler
# \TODO: 4. float precision 32 -> 16으로 낮추고 batchsize 16 -> 32
# using pytorch autocast
# 3 -> 4 -> 1 -> 2
# \TODO: 5. valence 반올림해서 10개 classification 으로 바꾸기

class Predictor:
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
        print("Pulling Brain and Valence pair data...")

        data_path="/home/data/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split"

        print('Prepping test dataloaders...')

        emotic_annotations = utils.get_emotic_data()
        nsd_df, target_cocoid = utils.get_NSD_data(emotic_annotations)

        self.subjects = [1, 2, 5, 7] if self.args.all_subjects else [self.args.subj]
        print('Train for current subjects:,', [f"subject{sub}" for sub in self.subjects])

        transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_dl, num_test = utils.get_torch_dataloaders(
            batch_size=1, # 1 for test data loader
            data_path = data_path,
            emotic_annotations=emotic_annotations,
            nsd_df=nsd_df,
            target_cocoid=target_cocoid,
            mode='test', # test mode
            subjects=self.subjects,
            task_type=self.args.task_type,
            num_classif=self.args.num_classif,
            data=self.args.data,
            use_sampler=self.args.sampler,
            use_body=True,
            transform=transform,
        )

        # for debugging
        # test_dl, _, num_test, _ = utils.get_torch_dataloaders(
        # # _, test_dl, _, num_test = utils.get_torch_dataloaders(
        #     batch_size=1, # 1 for test data loader
        #     data_path = data_path,
        #     emotic_annotations=emotic_annotations,
        #     nsd_df=nsd_df,
        #     target_cocoid=target_cocoid,
        #     mode='train', # test mode
        #     subjects=self.subjects,
        #     task_type=self.args.task_type,
        #     num_classif=self.args.num_classif,
        #     data=self.args.data,
        # )

        print('# test data:', num_test)

        return test_dl, num_test
    
    def load_model(self, args, use_best=True) -> nn.Module :
        if self.args.task_type == 'img2vad':
            model = Image2VADModel(
                image_backbone=self.args.model,
                num_classif=self.args.num_classif,
                pretrained=self.args.pretrain
            )
        elif self.args.task_type in ['reg', 'classif']:
            model = Brain2ValenceModel(
                model_name=self.args.model,
                task_type=self.args.task_type,
                num_classif=self.args.num_classif,
                subject=self.subjects,
            )
        else: 
            raise NotImplementedError(f"task type {self.args.task_type} is not implemented")
        
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

        if self.args.task_type == "reg":
            self.predict_regression()
        elif self.args.task_type == "classif":
            self.predict_classification()
        elif self.args.task_type == "img2vad":
            self.predict_img2vad()
        else: 
            raise NotImplementedError(f"task type {self.args.task_type} is not implemented")

    def predict_classification(self):
        test_loss = 0.0
        correct = 0
        criterion = nn.CrossEntropyLoss()

        true_valences = []
        pred_valences = []

        # data: brain3d or roi
        with torch.no_grad():
            for i, (data, image, valence, arousal, dominance) in enumerate(self.test_dl):
                if i == 0:
                    print("data type:", self.args.data)
                    print("task type:", self.args.task_type)
                    print("data shape:", data.shape)
                    print("valence shape:", valence.shape)
                data = data.float().cuda()
                valence = valence.long().cuda()

                output = self.model(data)

                # (B, num_classif)
                # Reshape data and target to handle varying batch sizes
                output = output.view(data.size(0), -1)
                valence = valence.view(valence.size(0))
                loss = criterion(output, valence)
                # select the index of the highest value
                # (B, num_classif) -> (B, 1)   
                pred_valence = output.argmax(dim=1, keepdim=True)

                test_loss += loss.item()
                print("g.t. valence:", valence)
                print("pred_valence:", pred_valence)
                correct += pred_valence.eq(valence.view_as(pred_valence)).sum().item()

                true_valences.append(valence.item())
                pred_valences.append(pred_valence.item())
            
        test_loss /= self.num_test
        print(f'\nTest set: Average loss: {test_loss:.4f},\
                   Accuracy: {correct}/{self.num_test} ({100. * correct / self.num_test:.0f}%)\n')
        wandb.log({"test_loss": test_loss, "accuracy": 100. * correct / self.num_test})

        utils.plot_valence_histogram(true_valences, pred_valences)

        # scatter pred and g.t. valence when only 10 class classfication
        if self.args.num_classif == 10:
            plt.scatter(true_valences, pred_valences)
            plt.xlabel("True valence")
            plt.ylabel("Pred valence")
            plt.plot([0, 10], [0, 10], color='red', linestyle='--')
            wandb.log({"plot true valence vs pred valence": wandb.Image(plt)})
            plt.clf()

    def predict_regression(self):
        true_valences = []
        pred_valences = []

        # data: brain3d or roi
        with torch.no_grad():
            for i, (data, image, valence, arousal, dominance) in enumerate(self.test_dl):
                if i == 0:
                    print("data type:", self.args.data)
                    print("task type:", self.args.task_type)
                    print("data shape:", data.shape)
                    print("valence shape:", valence.shape)
                data = data.float().cuda()
                valence = valence.float().cuda()
                
                pred_valence = self.model(data)

                print("g.t. valence:", valence)  
                print("predicted valence:", pred_valence)  

                true_valences.append(valence.item())
                pred_valences.append(pred_valence.item())
                    
        mae = np.mean(np.abs(np.array(true_valences) - np.array(pred_valences)))
        rmse = np.sqrt(mean_squared_error(true_valences, pred_valences))
        r2 = r2_score(true_valences, pred_valences)
        print("MAE: {:.4f}, RMSE: {:.4f}, R squared: {:.4f}".format(mae, rmse, r2))
        print({"true_valence mean": np.mean(true_valences), "true_valence std": np.std(true_valences)})
        print({"pred_valence mean": np.mean(pred_valences), "pred_valence std": np.std(pred_valences)})

        wandb.log({"MAE": mae, "RMSE": rmse, "r2_score": r2})
        wandb.log({"true_valence mean": np.mean(true_valences), "true_valence std": np.std(true_valences)})
        wandb.log({"pred_valence mean": np.mean(pred_valences), "pred_valence std": np.std(pred_valences)})

        # Plot true vs pred valence 
        plt.scatter(true_valences, pred_valences)
        plt.xlabel("True valence")
        plt.ylabel("Pred valence")
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                
        wandb.log({"plot true valence vs pred valence": wandb.Image(plt)})
        plt.clf()

    def predict_img2vad(self):
        pred_vad = torch.Tensor().cuda()
        gt_vad = torch.Tensor().cuda()

        with torch.no_grad():
            for i, (data, image, valence, arousal, dominance) in enumerate(self.test_dl):
                data = data.float().cuda()
                image = image.float().cuda()
                valence = valence.float().cuda()
                arousal = arousal.float().cuda()
                dominance = dominance.float().cuda()

                pred = self.model(image) # (1, 3)
                pred_vad = torch.cat((pred_vad, pred), dim=0)
                
                gt = torch.stack([valence, arousal, dominance], dim=1).float().cuda() # (1, 3)
                gt_vad = torch.cat((gt_vad, gt), dim=0)

        vad_mae = F.l1_loss(pred_vad, gt_vad, reduction='none')
        v_mae = vad_mae[:, 0].mean().item()
        a_mae = vad_mae[:, 1].mean().item()
        d_mae = vad_mae[:, 2].mean().item()
        total_mae = vad_mae.mean().item()

        print("valence mae: {:.4f}, arousal mae: {:.4f}, dominance mae: {:.4f}, total mae: {:.4f}".format(v_mae, a_mae, d_mae, total_mae))
        
        wandb.log({"valence_mae": v_mae, "arousal_mae": a_mae, "dominance_mae": d_mae, "total_mae": total_mae})

        for index, vad in enumerate(['valence', 'arousal', 'dominance']):
            # Plot true vs pred valence 
            plt.scatter(gt_vad[:, index].cpu().numpy(), pred_vad[:, index].cpu().numpy(), alpha=0.4)
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
    
    def calculate_AP_score(cat_preds, cat_labels):
        ap = [average_precision_score(cat_labels[i, :], cat_preds[i, :]) for i in range(26)]
        return ap
    