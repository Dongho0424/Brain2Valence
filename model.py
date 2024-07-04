import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from typing import List
from monai.networks.nets import resnet
from resnet import ResNetwClf
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, __dict__, ResNet

class BrainModel(nn.Module):
    def __init__(self,
                 image_backbone: str = "resnet18",
                 image_model_type: str = "BI",
                 brain_backbone: str = "resnet18",
                 brain_data_type: str = 'brain3d',
                 brain_out_feature = 512,
                 pretrain: str = "None",
                 backbone_freeze=False,
                 subjects = [1, 2, 5, 7], # for subject specific mlp model
                 cat_only = False,
                 fusion_ver=1, # 1 or 2
                 ):
        super().__init__()

        ## For Image
        assert image_backbone in ["resnet18", "resnet50"], f"backbone {image_backbone} is not implemented"
        self.image_backbone = image_backbone
        assert image_model_type in ["B", "BI", "I", "brain_only"], f"model type {image_model_type} is not implemented"
        self.image_model_type = image_model_type
        assert pretrain in ["None", "default", "EMOTIC"], f"pretrain {pretrain} is not implemented"
        self.pretrain = pretrain

        ## For Brain
        assert brain_backbone in ["resnet18", "resnet50", "mlp1", "mlp2"], f"backbone {brain_backbone} is not implemented"
        self.brain_backbone = brain_backbone
        assert brain_data_type in ["brain3d", "roi"], f"data type {brain_data_type} is not implemented"
        self.brain_data_type = brain_data_type

        self.cat_only = cat_only

        print("#############################")
        print("### Initialize BrainModel ###")
        print("Image Model backbone:", image_backbone)
        print("Image Model type:", image_model_type)
        print("Pretrain Type:", pretrain)
        print("Brain Model backbone:", brain_backbone)
        print("Brain Data Type:", brain_data_type)
        print("Data type:", brain_data_type)
        print("Category Prediction Only:", cat_only)
        print("#############################")

        ## Image Model
        if self.image_backbone == "resnet18":

            # context model
            model_context: ResNet = __dict__[self.image_backbone](num_classes=365)
            # body model
            model_body = resnet18()

            # Pretrain Setting
            if self.pretrain == "None":
                print("Context & Body model: train from scratch")
                self.context_last_feature = list(model_context.children())[-1].in_features
                self.model_context = nn.Sequential(*list(model_context.children())[:-1])
                self.body_last_feature = list(model_body.children())[-1].in_features
                self.model_body = nn.Sequential(*list(model_body.children())[:-1])
            elif self.pretrain == "default": 
                # context model pretrained by Places365 dataset
                print("Context model: Use pretrained model by Places365 dataset")
                context_state_dict = torch.load('/home/dongho/brain2valence/data/places/resnet18_state_dict.pth')
                model_context.load_state_dict(context_state_dict)
                # body model pretrained by ImageNet dataset
                print("Body model: Use pretrained model by ImageNet dataset")
                model_body = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

                self.context_last_feature = list(model_context.children())[-1].in_features
                self.model_context = nn.Sequential(*list(model_context.children())[:-1])
                self.body_last_feature = list(model_body.children())[-1].in_features
                self.model_body = nn.Sequential(*list(model_body.children())[:-1])

            elif self.pretrain == "EMOTIC":
                print("Context model: Use pretrained model by EMOTIC dataset")
                print("Body model: Use pretrained model by EMOTIC dataset")

                self.context_last_feature = list(model_context.children())[-1].in_features
                self.model_context = nn.Sequential(*list(model_context.children())[:-1])
                self.body_last_feature = list(model_body.children())[-1].in_features
                self.model_body = nn.Sequential(*list(model_body.children())[:-1])

                target_model_dir = "/home/dongho/brain2valence/trained_models/pretrain_whole_emotic_dataset_0702_not_freeze/best_model.pth"

                pretrained_weights = torch.load(target_model_dir)
                self.load_state_dict(pretrained_weights, strict=False)  # load both context and body

        # TODO : implement resnet50
        elif self.image_backbone == "resnet50":
            raise NotImplementedError("resnet50 is not implemented for image model")
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

        ## Brain Model
        if self.brain_backbone == "resnet18":
            self.res_model = ResNetwClf(backbone_type='resnet_18', num_classes=brain_out_feature)
        elif self.brain_backbone == "resnet50":
            self.res_model = ResNetwClf(backbone_type='resnet_50', num_classes=brain_out_feature)
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
            self.proj = nn.Linear(h, brain_out_feature, bias=True)

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
                nn.Linear(1024, brain_out_feature, bias=True),
            )

        # fusion model 
        # three backbones corresponding to model type
        fuse_in_features = 0
        if self.image_model_type == "B": fuse_in_features = self.body_last_feature
        elif self.image_model_type == "I": fuse_in_features = self.context_last_feature
        elif self.image_model_type == "BI": fuse_in_features = self.context_last_feature + self.body_last_feature
        elif self.image_model_type == "brain_only": fuse_in_features = 0
        else: raise NotImplementedError(f"model type {image_model_type} is not implemented")

        fuse_in_features += brain_out_feature # 512 or 1024 or 1536
        fuse_out_features = 256

        # TODO: 여기가 별로 맘에 들지 않음.
        if fusion_ver == 1:
            self.model_fusion = nn.Sequential(
                nn.Linear(fuse_in_features, fuse_out_features),
                nn.BatchNorm1d(fuse_out_features),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )
        elif fusion_ver == 2:
            self.model_fusion = nn.Sequential(
                nn.Linear(fuse_in_features, fuse_in_features),
                nn.BatchNorm1d(fuse_in_features),
                nn.GELU(),
                nn.Linear(fuse_in_features, fuse_in_features),
                nn.BatchNorm1d(fuse_in_features),
                nn.GELU(),
                nn.Linear(fuse_in_features, fuse_in_features),
                nn.BatchNorm1d(fuse_in_features),
                nn.GELU(),
                nn.Linear(fuse_in_features, fuse_out_features),
            )
        else: raise NotImplementedError(f"fusion version {fusion_ver} is not implemented")

        if self.cat_only:
            self.fc_cat = nn.Linear(fuse_out_features, 26)
        else:
            self.fc_cat = nn.Linear(fuse_out_features, 26)
            self.fc_vad = nn.Linear(fuse_out_features, 3)
        
        # freeze pretrained parameters
        if backbone_freeze:
            for param in self.model_body.parameters():
                param.requires_grad = False
            for param in self.model_context.parameters():
                param.requires_grad = False

    def forward(self, 
                x_body: torch.Tensor = None,
                x_context: torch.Tensor = None,
                x_brain: torch.Tensor = None):
        """
        - x_context: (B, 3, 224, 224), 
        - x_body: (B, 3, 112, 112)
        - x_brain: brain3d or roi
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

        if self.cat_only:
            cat_out = F.sigmoid(self.fc_cat(fuse_out))
            return cat_out
        else:
            cat_out = F.sigmoid(self.fc_cat(fuse_out))
            vad_out = self.fc_vad(fuse_out)
            return cat_out, vad_out

class Image2VADModel(nn.Module):
    def __init__(self,
                 image_backbone: str = "resnet18",
                 image_model_type: str = "BI",
                 pretrain="None",
                 backbone_freeze=False,
                 cat_only=False,
                 ):
        super().__init__()

        self.backbone = image_backbone
        assert image_model_type in ["B", "BI", "I"], f"model type {image_model_type} is not implemented"
        self.model_type = image_model_type
        self.cat_only = cat_only
        assert pretrain in ["None", "default"], f"pretrain {pretrain} is not implemented"
        self.pretrain = pretrain

        print("#############################")
        print("### Initialize Image2VADModel ###")
        print("Image Model backbone:", image_backbone)
        print("Image Model type:", image_model_type)
        print("Category Prediction Only:", cat_only)
        print("#############################")

        if self.backbone == "resnet18":

            # context model
            model_context = __dict__[self.backbone](num_classes=365)
            
            # body model
            model_body = resnet18()

            if self.pretrain == "None":
                print("Context & Body model: train from scratch")
            elif self.pretrain == "default": 
                # context model
                context_state_dict = torch.load('/home/dongho/brain2valence/data/places/resnet18_state_dict.pth')
                model_context.load_state_dict(context_state_dict)
                self.context_last_feature = list(model_context.children())[-1].in_features
                self.model_context = nn.Sequential(*list(model_context.children())[:-1])

                # body model
                model_body = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)   
                self.body_last_feature = list(model_body.children())[-1].in_features
                self.model_body = nn.Sequential(*list(model_body.children())[:-1])
            
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
            cat_out = F.sigmoid(self.fc_cat(fuse_out))
            return cat_out
        else:
            cat_out = F.sigmoid(self.fc_cat(fuse_out))
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