import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from typing import List
from monai.networks.nets import resnet
from resnet import ResNetwClf
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, __dict__
# freeze, res18
# TODO: cover context image given task type (B, BI)
class Image2VADModel(nn.Module):
    def __init__(self,
                 backbone: str = "resnet18",
                 model_type: str = "BI",
                 pretrained=True,
                 backbone_freeze=False,
                 ):
        super().__init__()

        self.backbone = backbone
        assert model_type in ["B", "BI", "I"], f"model type {model_type} is not implemented"
        self.model_type = model_type
        print("Model backbone:", backbone)
        print("Model type:", model_type)


        if self.backbone == "resnet18":

            # context model
            model_context = __dict__[self.backbone](num_classes=365)
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
            else: raise NotImplementedError(f"model type {model_type} is not implemented")

            self.model_fusion = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )
            self.fc_cat = nn.Linear(256, 26)
            self.fc_cont = nn.Linear(256, 3)
            
            # freeze parameters
            if backbone_freeze:
                for param in self.model_body.parameters():
                    param.requires_grad = False
                for param in self.model_context.parameters():
                    param.requires_grad = False

        elif self.backbone == "resnet50":
            # temporarily
            assert self.model_type == 'B', "resnet50 is only for model_type: B"
            # load pretrained model
            self.model_body = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) if pretrained else resnet50()
            # freeze parameters
            if backbone_freeze:
                for param in self.model_body.parameters():
                    param.requires_grad = False
            # add layer for fine tuning
            num_ftrs = self.model_body.fc.in_features
            # self.model.fc = nn.Linear(num_ftrs, num_classes)
            self.model_body.fc = nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 3)
            )
        else:
            raise NotImplementedError(f"model {backbone} is not implemented")
        
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
            
        cat_out = F.sigmoid(self.fc_cat(fuse_out))
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out

class Emotic(nn.Module):
    def __init__(self, num_context_features, num_body_features):
        super(Emotic,self).__init__()

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((self.num_context_features + num_body_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.d1 = nn.Dropout(p=0.5)

        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)


    def forward(self, x_context, x_body):
        context_features = x_context.view(-1, self.num_context_features)
        body_features = x_body.view(-1, self.num_body_features)
        fuse_features = torch.cat((context_features, body_features), 1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)   
            
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out


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