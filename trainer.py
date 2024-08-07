import wandb
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import os
import utils 
from model import Brain2ValenceModel, EmoticModel
from torchvision.transforms import v2
from tqdm import tqdm

class Trainer:
    def __init__(self, args):
        self.args = args

        self.train_dl, self.val_dl, self.num_train, self.num_val = self.prepare_dataloader()
        self.set_wandb_config()
        self.model: nn.Module = self.get_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.criterion = self.get_criterion()

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

        print('Prepping train and validation dataloaders...')

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

        train_dl, val_dl, num_train, num_val = utils.get_torch_dataloaders(
            batch_size=self.args.batch_size,
            data_path=data_path,
            emotic_annotations=emotic_annotations,
            nsd_df=nsd_df,
            target_cocoid=target_cocoid,
            mode='train', # train mode
            subjects=self.subjects,
            task_type=self.args.task_type,
            num_classif=self.args.num_classif,
            data=self.args.data,
            use_sampler=self.args.sampler,
            use_body=True,
            transform=transform,
        )

        print('# train data:', num_train)
        print('# val data:', num_val)

        return train_dl, val_dl, num_train, num_val
    
    def get_model(self):
        if self.args.task_type == 'img2vad':
            model = EmoticModel(
                image_backbone=self.args.model,
                num_classif=self.args.num_classif,
                pretrained=self.args.pretrained,
                backbone_freeze=self.args.backbone_freeze,
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
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        return scheduler

    
    def get_criterion(self):
        if self.args.criterion == "mse":
            criterion = nn.MSELoss()
        elif self.args.criterion == "mse_sum":
            criterion = nn.MSELoss(reduction='sum')
        elif self.args.criterion == "mae":
            criterion = nn.L1Loss()
        elif self.args.criterion == "ce":
            criterion = nn.CrossEntropyLoss()
        return criterion
    
    def train(self):
        #### enter Training ###
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:', device)
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())
        #### enter Training ###

        self.model.cuda()
        best_val_loss = float("inf")
        for epoch in range(self.args.epochs):
            self.model.train() 
            train_loss = 0

            scaler = GradScaler()

            # data: brain3d or roi
            for i, (data, image, valence, arousal, dominance) in tqdm(enumerate(self.train_dl)):
                
                self.optimizer.zero_grad()
                data = data.float().cuda()
                image = image.float().cuda()
                valence = valence.float().cuda()
                arousal = arousal.float().cuda()
                dominance = dominance.float().cuda()
                if self.args.task_type in ['reg', 'classif']:

                    with autocast():
                        pred_valence = self.model(data)
                        # Reshape data and target to handle varying batch sizes
                        pred_valence = pred_valence.view(data.size(0), -1)
                        valence = valence.view(valence.size(0))
                        loss = self.criterion(pred_valence, valence)
                    # Scales loss and calls backward() to create scaled gradients
                    scaler.scale(loss).backward()
                    # Unscales gradients and calls optimizer.step()
                    scaler.step(self.optimizer)
                    # Updates the scale for next iteration
                    scaler.update()

                    # loss.item(): the average loss of the batch
                    # multiply by batch size to get total loss
                    train_loss += loss.item() * self.args.batch_size # multiply by batch size

                elif self.args.task_type == 'img2vad':
                    pred = self.model(image)
                    gt = torch.stack([valence, arousal, dominance], dim=1).float().cuda()
                    loss = self.criterion(pred, gt) # always mse
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item() * self.args.batch_size
                else: 
                    raise NotImplementedError(f"task type {self.args.task_type} is not implemented")
            
            # get average total loss by dividing by the number of train data
            train_loss /= float(self.num_train)
            wandb.log(
                {"train_loss": train_loss, 
                 "lr": self.optimizer.param_groups[0]['lr']}, step=epoch)    

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, (data, image, valence, arousal, dominance) in tqdm(enumerate(self.val_dl)):
                    
                    data = data.float().cuda()
                    image = image.float().cuda()
                    valence = valence.float().cuda()
                    arousal = arousal.float().cuda()
                    dominance = dominance.float().cuda()

                    if self.args.task_type in ['reg', 'classif']:
                        pred_valence = self.model(data)
                        loss = self.criterion(pred_valence, valence)
                        val_loss += loss.item() * self.args.batch_size # multiply by batch size

                    elif self.args.task_type == 'img2vad':
                        pred = self.model(image)
                        gt = torch.stack([valence, arousal, dominance], dim=1).float().cuda()
                        loss = self.criterion(pred, gt)
                        val_loss += loss.item() * self.args.batch_size # multiply by batch size
                    else: 
                        raise NotImplementedError(f"task type {self.args.task_type} is not implemented")
                val_loss /= float(self.num_val)
                wandb.log({"val_loss": val_loss}, step=epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wandb.log({"best_val_loss": best_val_loss}, step=epoch)
                self.save_model(self.args, self.model, best=True)
                
            self.scheduler.step()

            print("Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}".format(epoch, train_loss, val_loss))
        
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
        model_name = args.model_name # ex) "all_subjects_res18_mae_2"
        save_dir = os.path.join(args.save_path, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if best:
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))
