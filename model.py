import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from typing import List
from monai.networks.nets import resnet
from resnet import ResNetwClf
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, __dict__, ResNet
import random

class ResMLP(nn.Module):
    def __init__(self, h, n_blocks, dropout=0.15):
        super().__init__()
        self.n_blocks = n_blocks
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(n_blocks)
        ])

    def forward(self, x):
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        return x

class BrainModel(nn.Module):
    def __init__(self,
                 image_backbone: str = "resnet18",
                 image_model_type: str = "BI",
                 brain_backbone: str = "resnet18",
                 brain_data_type: str = 'brain3d',
                 brain_in_dim = 2048, # pool_num for cross_subj 
                 brain_out_dim = 512,
                 pretrained: str = "None",
                 wgt_path: str = None,
                 backbone_freeze=False,
                 subjects = [1, 2, 5, 7], # for subject specific mlp model
                 cat_only = False,
                 fusion_ver=1, # 1 or 2, 999 for one-point
                 ):
        super().__init__()

        ## For Image
        assert image_backbone in ["resnet18", "resnet50"], f"backbone {image_backbone} is not implemented"
        self.image_backbone = image_backbone
        assert image_model_type in ["B", "BI", "I", "brain_only"], f"model type {image_model_type} is not implemented"
        self.image_model_type = image_model_type
        assert pretrained in ["None", "default", "EMOTIC", "simple_cross_subj", "cross_subj"], f"pretrain {pretrained} is not implemented"
        self.pretrained = pretrained
        if pretrained == "EMOTIC":
            assert(wgt_path is not None), "wgt_path is required for EMOTIC pretrained model"

        ## For Brain
        assert brain_backbone in ["resnet18", "resnet50", "mlp1", "mlp2", "mlp3", "simple_cross_subj", "single_subj", "cross_subj"],\
              f"backbone {brain_backbone} is not implemented"
        self.brain_backbone = brain_backbone
        assert brain_data_type in ["brain3d", "roi"], f"data type {brain_data_type} is not implemented"
        self.brain_data_type = brain_data_type
        self.brain_in_dim = brain_in_dim
        self.subjects = subjects
        self.cat_only = cat_only

        print("#############################")
        print("### Initialize BrainModel ###")
        print("Image Model backbone:", image_backbone)
        print("Image Model type:", image_model_type)
        print("Pretrain Type:", pretrained)
        print("Brain Model backbone:", brain_backbone)
        print("Brain Data Type:", brain_data_type)
        print("Data type:", brain_data_type)
        print("Category Prediction Only:", cat_only)
        print("#############################")

        ## Image Model ##
        if self.image_backbone == "resnet18":

            # context model
            model_context: ResNet = __dict__[self.image_backbone](num_classes=365)
            # body model
            model_body = resnet18()
            # features
            self.context_last_feature = list(model_context.children())[-1].in_features
            self.body_last_feature = list(model_body.children())[-1].in_features

        # TODO : implement resnet50
        elif self.image_backbone == "resnet50":
            raise NotImplementedError("resnet50 is not implemented for image model")
            
        ## Brain Model ##
        if self.brain_backbone == "resnet18":
            self.res_model = ResNetwClf(backbone_type='resnet_18', num_classes=brain_out_dim)
        elif self.brain_backbone == "resnet50":
            self.res_model = ResNetwClf(backbone_type='resnet_50', num_classes=brain_out_dim)
        elif self.brain_backbone == "mlp1":
            assert len(subjects) == 1, "mlp model is only for subject specific model"

            features = utils.get_num_voxels(subjects[-1])
            h = 4096
            self.lin0 = nn.Sequential(
                nn.Linear(features, h, bias=False),
                nn.BatchNorm1d(h), 
                nn.GELU(),
                nn.Dropout(0.5),
            )
            self.mlp = nn.ModuleList([ 
                nn.Sequential(
                    nn.Linear(h, h, bias=False),
                    nn.LayerNorm(h),
                    nn.GELU(), 
                    nn.Dropout(0.15)
                ) for _ in range (4)])
            self.proj = nn.Linear(h, brain_out_dim, bias=True)

        elif self.brain_backbone == "mlp2": # lightweight version of "mlp1"
            assert len(subjects) == 1, "mlp2 model is only for subject specific model"

            features = utils.get_num_voxels(subjects[-1])
            h = 4096
            self.lin0 = nn.Linear(features, h, bias=False)
            self.mlp = nn.ModuleList([ 
                nn.Sequential(
                    nn.LayerNorm(h),
                    nn.Linear(h, h, bias=False),
                    nn.GELU(), 
                    nn.Dropout(0.15),
                    nn.Linear(h, h, bias=False),
                ) for _ in range (2)])
            self.proj = nn.Sequential(
                nn.Linear(h, 1024, bias=True),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Linear(1024, brain_out_dim, bias=True),
            )
        elif self.brain_backbone == "mlp3": # using AdaptiveMaxPool1d
            assert len(subjects) == 1, "mlp3 model is only for subject specific model"

            features = utils.get_num_voxels(subjects[-1])
            h = 2048
            self.max_pool = nn.AdaptiveMaxPool1d(h) 
            self.mlp = nn.ModuleList([ 
                nn.Sequential(
                    nn.Linear(h, h, bias=False),
                    nn.LayerNorm(h),
                    nn.GELU(), 
                    nn.Dropout(0.15),
                ) for _ in range (2)])
            self.proj = nn.Sequential(
                nn.Linear(h, 1024, bias=True),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Linear(1024, brain_out_dim, bias=True),
            )
        elif self.brain_backbone == "simple_cross_subj": # for training
            # We assume that the number of voxels is adaptively max pooled to brain_in_dim (2048)
            # before feeding into the model 
            
            # MindBridge-like embedder and builder
            h = brain_in_dim
            self.embedder = nn.ModuleDict({
                str(subj): nn.Sequential( 
                    ResMLP(h, 2),
                    nn.Linear(h, 1024, bias=True),
                    nn.LayerNorm(1024),
                    nn.GELU(),
                    nn.Linear(1024, brain_out_dim, bias=True),
                ) for subj in subjects
            })
        elif self.brain_backbone == "cross_subj": # for training
            assert len(subjects) > 1, "cross_subj model is only for cross_subject model"
            # We assume that the number of voxels is adaptively max pooled to brain_in_dim (2048)
            # before feeding into the model 
            
            # MindBridge-like embedder and builder
            h = brain_in_dim
            self.embedder = nn.ModuleDict({
                str(subj): nn.Sequential( 
                    ResMLP(h, 2),
                    nn.Linear(h, brain_out_dim, bias=True),
                    nn.LayerNorm(brain_out_dim),
                    nn.GELU(),
                ) for subj in subjects
            })
            self.builder = nn.ModuleDict({
                str(subj): nn.Sequential( 
                    nn.Linear(brain_out_dim, h, bias=True),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    ResMLP(h, 2)
                )
                for subj in subjects
            })
        elif self.brain_backbone == "single_subj": # for predicting
            assert len(subjects) == 1, "As we finetune the model for single subject, only one subject is allowed for predicting"
            # We assume that the number of voxels is adaptively max pooled to brain_in_dim (2048) 
            # before feeding into the model 
            
            # MindBridge-like 
            h = brain_in_dim
            self.embedder = nn.ModuleDict({
                str(subj): nn.Sequential( 
                    ResMLP(h, 2),
                    nn.Linear(h, brain_out_dim, bias=True),
                    nn.LayerNorm(brain_out_dim),
                    nn.GELU(),
                ) for subj in subjects
            })

        ## fusion model ##
        # three backbones corresponding to model type
        fuse_in_dim = 0
        if self.image_model_type == "B": fuse_in_dim = self.body_last_feature
        elif self.image_model_type == "I": fuse_in_dim = self.context_last_feature
        elif self.image_model_type == "BI": fuse_in_dim = self.context_last_feature + self.body_last_feature
        elif self.image_model_type == "brain_only": fuse_in_dim = 0
        else: raise NotImplementedError(f"model type {image_model_type} is not implemented")

        fuse_in_dim += brain_out_dim # 512 or 1024 or 1536
        fuse_out_dim = 256

        if fusion_ver == 1: # EMOTIC paper ver.
            self.model_fusion = nn.Sequential(
                nn.Linear(fuse_in_dim, fuse_out_dim),
                nn.BatchNorm1d(fuse_out_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )
        elif fusion_ver == 2:
            self.model_fusion = nn.Sequential(
                nn.Linear(fuse_in_dim, fuse_in_dim),
                nn.BatchNorm1d(fuse_in_dim),
                nn.GELU(),
                nn.Linear(fuse_in_dim, fuse_in_dim),
                nn.BatchNorm1d(fuse_in_dim),
                nn.GELU(),
                nn.Linear(fuse_in_dim, fuse_in_dim),
                nn.BatchNorm1d(fuse_in_dim),
                nn.GELU(),
                nn.Linear(fuse_in_dim, fuse_out_dim),
            )
        elif fusion_ver == 999: # Replace BatchNorm to LayerNorm
            self.model_fusion = nn.Sequential(
                nn.Linear(fuse_in_dim, fuse_out_dim),
                nn.LayerNorm(fuse_out_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )
        else: raise NotImplementedError(f"fusion version {fusion_ver} is not implemented")

        ## Final Layers ##
        if self.cat_only:
            self.fc_cat = nn.Linear(fuse_out_dim, 26)
        else:
            self.fc_cat = nn.Linear(fuse_out_dim, 26)
            self.fc_vad = nn.Linear(fuse_out_dim, 3)

        ## Using Pretrained Weights ## 
        if self.pretrained == "None":
            print("Context & Body model: train from scratch")
            # remove last layer
            self.model_context = nn.Sequential(*list(model_context.children())[:-1])
            self.model_body = nn.Sequential(*list(model_body.children())[:-1])

        elif self.pretrained == "default": 
            # context model pretrained by Places365 dataset
            print("Context model: Use pretrained model by Places365 dataset")
            context_state_dict = torch.load('/home/dongho/brain2valence/data/places/resnet18_state_dict.pth')
            model_context.load_state_dict(context_state_dict)
            # body model pretrained by ImageNet dataset
            print("Body model: Use pretrained model by ImageNet dataset")
            model_body = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

            # remove last layer
            self.model_context = nn.Sequential(*list(model_context.children())[:-1])
            self.model_body = nn.Sequential(*list(model_body.children())[:-1])

        elif self.pretrained == "EMOTIC":
            assert fusion_ver == 1, "EMOTIC pretrained model is only available for fusion_ver=1"
            print("Context model: Use pretrained model by EMOTIC dataset")
            print("Body model: Use pretrained model by EMOTIC dataset")

            # The pretrained weights of emotic model using EMOTIC dataset
            print("pretrained weight dir:", wgt_path)
            pretrained_weights = torch.load(wgt_path)
            # We now want to feed pretrained weights into image_model, fusion network, final layers, excluding brain_model.
            # However, the input sizes of the fusion network are different between brain_model and emotic_model
            # As only the first layer's size is different, only adjusting the first layer of fusion network is needed
            first_fusion_layer_key = "model_fusion.0.weight"
            prev_weight = pretrained_weights[first_fusion_layer_key]
            # input: fuse_in = torch.cat([x_body, x_context, x_brain], 1)
            # n * 1536 fuse_in; left 1024 from image data, right 512 from brain data
            # 1536 * 256 weight; upper 1024 * 256 is fed pretrained, lower 512 * 256 is zero initialized
            # But the weight of FFN in Sequential() is transposed.
            # Therefore, according to weight size 256 * 1536, left 256 * 1024 is fed pretrained, right 256 * 512 is zero initialized
            # new_weight = torch.cat([prev_weight, torch.zeros(256, 512)], dim=1)
            new_weight = torch.cat([prev_weight, torch.zeros(fuse_out_dim, brain_out_dim)], dim=1)

            pretrained_weights[first_fusion_layer_key] = new_weight

            # remove last layer
            self.model_context = nn.Sequential(*list(model_context.children())[:-1])
            self.model_body = nn.Sequential(*list(model_body.children())[:-1])

            self.load_state_dict(pretrained_weights, strict=False)
            # print(self.state_dict()[first_fusion_layer_key])
        elif self.pretrained in ["cross_subj", "simple_cross_subj"]:
            assert brain_data_type == "roi", "cross_subj is only available for roi data type" 
            print("Img Context model: Use pretrained model by EMOTIC dataset")
            print("Img Body model: Use pretrained model by EMOTIC dataset")
            print("brain model + fusion net(for ROI data): Use pretrained model by other subjects")

            print("pretrained weight dir:", wgt_path)
            # Use pretrained weight to image model and fusion net
            pretrained_weights = torch.load(wgt_path)
            self.load_state_dict(pretrained_weights, strict=False)

            # remove last layer
            self.model_context = nn.Sequential(*list(model_context.children())[:-1])
            self.model_body = nn.Sequential(*list(model_body.children())[:-1])
        
        # freeze pretrained parameters
        if backbone_freeze:
            for param in self.model_body.parameters():
                param.requires_grad = False
            for param in self.model_context.parameters():
                param.requires_grad = False

    def forward(self, 
                x_body: torch.Tensor = None,
                x_context: torch.Tensor = None,
                x_brain: torch.Tensor = None,
                subj_list = None, # for cross_subj, 
                ):
        """
        - x_context: (B, 3, 224, 224), 
        - x_body: (B, 3, 112, 112)
        - x_brain: brain3d or roi
        - subj_list: list of subj
            - different from self.subjects, which is used for initializing embedder and builder
            - this subj_list is used for selecting specific subject's embedder
        - out: (B, 26), (B, 3)
            - 26 for emotion categories
            - 3 for vad
        """
        assert not (self.image_model_type in "B" and x_body is None), "body is required for model type B"
        assert not (self.image_model_type in "I" and x_context is None), "context is required for model type I"

        # for brain
        if self.brain_backbone in ["resnet18", "resnet50"]:
            # add channel dimension
            x_brain = x_brain.unsqueeze(dim=1)  # (B, 96, 96, 96) -> (B, 1, 96, 96, 96)
            x_brain = self.res_model(x_brain)  # (B, brain_out_feature)
        elif self.brain_backbone == "mlp1":
            x_brain = self.lin0(x_brain)
            residual = x_brain
            for block in range(len(self.mlp)):
                x_brain = self.mlp[block](x_brain)
                x_brain += residual   
                residual = x_brain
            x_brain = self.proj(x_brain) # (B, brain_out_feature)
        elif self.brain_backbone == "mlp2":
            x_brain = self.lin0(x_brain)
            residual = x_brain
            for block in range(len(self.mlp)):
                x_brain = self.mlp[block](x_brain)
                x_brain += residual   
                residual = x_brain
            x_brain = self.proj(x_brain) # (B, brain_out_feature)
        elif self.brain_backbone == "mlp3":
            x_brain = self.max_pool(x_brain) # adaptive max pool
            residual = x_brain
            for block in range(len(self.mlp)):
                x_brain = self.mlp[block](x_brain)
                x_brain += residual   
                residual = x_brain
            x_brain = self.proj(x_brain) # (B, brain_out_feature)
        elif self.brain_backbone == "simple_cross_subj": # training
            assert x_brain.shape[1] == self.brain_in_dim, f"We assume that # voxels is adaptively max pooled to {self.brain_in_dim}"
            assert subj_list is not None, "subj_list is required for cross_subj"

            x_subj_list = torch.chunk(x_brain, len(subj_list), dim=0) # let each element: (B, 2048)
            x = []
            for i, subj_i in enumerate(subj_list): # pretraining: [2, 5, 7], # fine-tuning: [1]
                x_i = self.embedder[str(subj_i)](x_subj_list[i]) # subj_i semantic embedding by embedder
                x.append(x_i)
            x_brain = torch.concat(x, dim=0) # (3 * B, brain_out_dim)

        elif self.brain_backbone == "cross_subj": # training
            assert x_brain.shape[1] == self.brain_in_dim, f"We assume that # voxels is adaptively max pooled to {self.brain_in_dim}"
            # pretraining: self.subjects == subj_list
            # adapting(fine-tuning): self.subjects != subj_list
            # then subj_list is used for forward
            assert subj_list is not None, "subj_list is required for cross_subj"
            
            x_subj_list = torch.chunk(x_brain, len(subj_list), dim=0) # each element: (B, 2048)
            x = []
            x_rec = []
            if self.pretrained == 'cross_subj': # choose subj_a (source subject) and subj_b (target subject)
                subj_a, subj_b = subj_list[0], subj_list[-1]
            else: # random sample 2 subjects
                subj_a, subj_b = random.sample(subj_list, 2) 

            for i, subj_i in enumerate(subj_list): # pretraining: [2, 5, 7] or # fine-tuning: [2(src), 1(tgt)]
 
                x_i = self.embedder[str(subj_i)](x_subj_list[i]) # subj_i semantic embedding by embedder
                
                if subj_i == subj_a: 
                    x_a = x_i                # subj_a seman embedding are choosen
                x.append(x_i)

                x_i_rec = self.builder[str(subj_i)](x_i) # subj_i recon brain signals by builder
                x_rec.append(x_i_rec)

            x_brain = torch.concat(x, dim=0) # (3 * B, brain_out_dim)
            x_rec = torch.concat(x_rec, dim=0) # (3 * B, brain_out_dim)

            # forward cycling
            x_b = self.builder[str(subj_b)](x_a)  # subj_b recon brain signal using subj_a seman embedding
            x_b = self.embedder[str(subj_b)](x_b) # subj_b semantic embedding (pseudo)
        elif self.brain_backbone == 'single_subj': # predicting
            # Predicting for single subject, which used to finetune this model
            assert x_brain.shape[1] == self.brain_in_dim, f"We assume that # voxels is adaptively max pooled to {self.brain_in_dim}"
            assert subj_list is not None, "subj_list is required for cross_subj"
            assert len(subj_list) == 1, "subj_list should have only one subject"

            x_brain = self.embedder[str(subj_list[0])](x_brain)

        else: raise NotImplementedError(f"backbone {self.brain_backbone} is not implemented")

        # for image
        if self.image_model_type == 'B':
            x_body = self.model_body(x_body)
            x_body = x_body.view(-1, self.body_last_feature)

            if x_brain.dim() == 1:
                x_brain = x_brain.unsqueeze(0)
            fuse_in = torch.cat((x_body, x_brain), 1)
            fuse_out = self.model_fusion(fuse_in)
        elif self.image_model_type == 'I':
            x_context = self.model_context(x_context)
            x_context = x_context.view(-1, self.context_last_feature)

            if x_brain.dim() == 1:
                x_brain = x_brain.unsqueeze(0)
            fuse_in = torch.cat((x_context, x_brain), 1)
            fuse_out = self.model_fusion(fuse_in)
        elif self.image_model_type == 'BI':
            x_body = self.model_body(x_body)
            x_body = x_body.view(-1, self.body_last_feature)

            x_context = self.model_context(x_context)
            x_context = x_context.view(-1, self.context_last_feature)

            if x_brain.dim() == 1:
                x_brain = x_brain.unsqueeze(0)
            fuse_in = torch.cat([x_body, x_context, x_brain], 1)
            fuse_out = self.model_fusion(fuse_in)
        elif self.image_model_type == "brain_only":
            if x_brain.dim() == 1:
                x_brain = x_brain.unsqueeze(0)
            fuse_out = self.model_fusion(x_brain)

        if self.brain_backbone == 'cross_subj':
            cat_out = self.fc_cat(fuse_out)
            return cat_out, x_rec, x_a, x_b
        elif self.brain_backbone == 'single_subj':
            cat_out = self.fc_cat(fuse_out)
            return cat_out
        elif self.cat_only:
            # return logit
            cat_out = self.fc_cat(fuse_out)
            return cat_out
        else:
            # return logit
            cat_out = self.fc_cat(fuse_out)
            vad_out = self.fc_vad(fuse_out)
            return cat_out, vad_out

class EmoticModel(nn.Module):
    def __init__(self,
                 image_backbone: str = "resnet18",
                 image_model_type: str = "BI",
                 pretrained="None",
                 wgt_path: str = None,
                 backbone_freeze=False,
                 cat_only=False,
                 ):
        super().__init__()

        self.backbone = image_backbone
        assert image_model_type in ["B", "BI", "I"], f"model type {image_model_type} is not implemented"
        self.model_type = image_model_type
        self.cat_only = cat_only
        assert pretrained in ["None", "default", "EMOTIC"], f"pretrain {pretrained} is not implemented"
        self.pretrained = pretrained
        if pretrained == "EMOTIC":
            assert(wgt_path is not None), "wgt_path is required for EMOTIC pretrained model"

        print("#############################")
        print("### Initialize Image2VADModel ###")
        print("Image Model backbone:", image_backbone)
        print("Image Model type:", image_model_type)
        print("Category Prediction Only:", cat_only)
        print("#############################")

        if self.backbone == "resnet18":

            # context model
            model_context = __dict__[self.backbone](num_classes=365)
            self.context_last_feature = list(model_context.children())[-1].in_features

            # body model
            model_body = resnet18()
            self.body_last_feature = list(model_body.children())[-1].in_features

            if self.pretrained == "None":
                print("Context & Body model: train from scratch")
            elif self.pretrained == "default": 
                # context model
                context_state_dict = torch.load('/home/dongho/brain2valence/data/places/resnet18_state_dict.pth')
                model_context.load_state_dict(context_state_dict)
                
                self.model_context = nn.Sequential(*list(model_context.children())[:-1])

                # body model
                model_body = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)   
                self.model_body = nn.Sequential(*list(model_body.children())[:-1])
            elif self.pretrained == "EMOTIC":
                print("Context model: Use pretrained model by EMOTIC dataset")
                print("Body model: Use pretrained model by EMOTIC dataset")

                # The pretrained weights of emotic model using EMOTIC dataset
                print("pretrained weight dir:", wgt_path)
                pretrained_weights = torch.load(wgt_path)
                
                # remove last layer
                self.model_context = nn.Sequential(*list(model_context.children())[:-1])
                self.model_body = nn.Sequential(*list(model_body.children())[:-1])

                self.load_state_dict(pretrained_weights, strict=False)
            
            # fusion two backbones corresponding to model type
            in_features = 0
            if self.model_type == "B": in_features = self.body_last_feature
            elif self.model_type == "I": in_features = self.context_last_feature
            elif self.model_type == "BI": in_features = self.context_last_feature + self.body_last_feature
            else: raise NotImplementedError(f"model type {image_model_type} is not implemented")

            self.model_fusion = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )
            
            if self.cat_only:
                self.fc_cat = nn.Linear(256, 26)
            else:
                self.fc_cat = nn.Linear(256, 26)
                self.fc_vad = nn.Linear(256, 3)
            
            # freeze parameters
            if backbone_freeze:
                for param in self.model_body.parameters():
                    param.requires_grad = False
                for param in self.model_context.parameters():
                    param.requires_grad = False

        elif self.backbone == "resnet50":
            assert(False)
            # # temporarily
            # assert self.model_type == 'B', "resnet50 is only for model_type: B"
            # # load pretrained model
            # self.model_body = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) if pretrained else resnet50()
            # # freeze parameters
            # if backbone_freeze:
            #     for param in self.model_body.parameters():
            #         param.requires_grad = False
            # # add layer for fine tuning
            # num_ftrs = self.model_body.fc.in_features
            # # self.model.fc = nn.Linear(num_ftrs, num_classes)
            # self.model_body.fc = nn.Sequential(
            #     nn.Linear(num_ftrs, 256),
            #     nn.BatchNorm1d(256),
            #     nn.ReLU(),
            #     nn.Dropout(0.5),
            #     nn.Linear(256, 3)
            # )
        else:
            raise NotImplementedError(f"model {image_backbone} is not implemented")
        
    def forward(self, x_body: torch.Tensor = None, x_context: torch.Tensor = None):
        """
        - body: (B, 3, 224, 224), (B, 3, 112, 112)
        - out: (B, 26), (B, 3)
            - 26 for emotion categories
            - 3 for vad
        """
        assert not (self.model_type in "B" and x_body is None), "body is required for model type B"
        assert not (self.model_type in "I" and x_context is None), "context is required for model type I"

        if self.model_type == 'B':
            x_body = self.model_body(x_body)
            x_body = x_body.view(-1, self.body_last_feature)
            fuse_out = self.model_fusion(x_body)
        elif self.model_type == 'I':
            x_context = self.model_context(x_context)
            x_context = x_context.view(-1, self.context_last_feature)
            fuse_out = self.model_fusion(x_context)
        elif self.model_type == 'BI':
            x_body = self.model_body(x_body)
            x_body = x_body.view(-1, self.body_last_feature)

            x_context = self.model_context(x_context)
            x_context = x_context.view(-1, self.context_last_feature)
            fuse_in = torch.cat((x_body, x_context), 1)
            fuse_out = self.model_fusion(fuse_in)
            
        if self.cat_only:
            cat_out = self.fc_cat(fuse_out)
            return cat_out
        else:
            cat_out = self.fc_cat(fuse_out)
            vad_out = self.fc_vad(fuse_out)
            return cat_out, vad_out

class Brain2ValenceModel(nn.Module):
    def __init__(self,
                 model_name: str = "resnet18",
                 task_type: str = "reg",
                 num_classif: int = 3,
                 subject: List[int] = [1], # for subject specific mlp model
                 ):
        super().__init__()

        self.model_name = model_name
        print("current model backbone:", model_name)

        # when regression, num_classes = 1
        num_classes = num_classif if task_type == "classif" else 1
        print("num_classes:", num_classes)

        if model_name == "resnet18":
            self.res_model = ResNetwClf(backbone_type='resnet_18', num_classes=num_classes)
            # self.model = resnet.resnet18(
            #     n_input_channels=1, num_classes=num_classes, feed_forward=True)
        elif model_name == "resnet50":
            self.res_model = ResNetwClf(backbone_type='resnet_50', num_classes=num_classes)
            # self.model = resnet.resnet50(
            #     n_input_channels=1, num_classes=num_classes, feed_forward=True)
        elif model_name == "mlp":
            assert len(subject) == 1, "mlp model is only for subject specific model"

            features = utils.get_num_voxels(subject[-1])
            self.lin1 = nn.Sequential(
                nn.Linear(features, 4096),
                nn.BatchNorm1d(4096), 
                nn.GELU(),
                nn.Dropout(0.5),
            )
            # self.mlp = nn.ModuleList([
            #     nn.Sequential(
            #         nn.Linear(4096, 4096),
            #         nn.BatchNorm1d(4096), 
            #         nn.GELU(),
            #         nn.Dropout(0.15),
            #     ) for _ in range(3)
            # ])
            self.last = nn.Sequential(
                nn.Linear(4096, 768),
                nn.BatchNorm1d(768), 
                nn.GELU(),
                nn.Linear(768, num_classes),
            )
        else:
            raise NotImplementedError(f"model {model_name} is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        - x: 
            - brain3d: (B, 96, 96, 96)
            - roi: (B, *)
        - out: (B, num_classif)
        """

        if self.model_name in ["resnet18", "resnet50"]:
            # add channel dimension
            x = x.unsqueeze(dim=1)  # (B, 96, 96, 96) -> (B, 1, 96, 96, 96)
            valence = self.res_model(x)  # (B, 1) or (B, num_classif)
        elif self.model_name == "mlp":
            x = self.lin1(x)
            # residual = x
            # for block in range(len(self.mlp)):
            #     x = self.mlp[block](x)
            #     x += residual   
            #     residual = x
            valence = self.last(x)
        return valence.float().squeeze(dim=-1)