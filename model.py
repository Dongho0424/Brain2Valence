import torch
import torch.nn as nn
import utils
from typing import List
from monai.networks.nets import resnet
from resnet import ResNetwClf
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
# freeze, res18
# TODO: cover context image given task type (B, BI)
class Image2VADModel(nn.Module):
    def __init__(self,
                 model_name: str = "resnet18",
                 num_classif: int = 3,
                 pretrained=True,
                 backbone_freeze=False,
                 ):
        super().__init__()

        self.model_name = model_name
        print("current model backbone:", model_name)

        # the number of class for VAD regression: 3
        # valence, arousal, dominance, respectively
        # num_classes = num_classif if task_type == "classif" else 3
        num_classes = 3
        print("num_classes:", num_classes)

        if self.model_name == "resnet18":
            # load pretrained model
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) if pretrained else resnet18()
            # freeze parameters
            if backbone_freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
            # add layer for fine tuning
            num_ftrs = self.model.fc.in_features
            # self.model.fc = nn.Linear(num_ftrs, num_classes)
            self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        elif self.model_name == "resnet50":
            # load pretrained model
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) if pretrained else resnet50()
            # freeze parameters
            if backbone_freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
            # add layer for fine tuning
            num_ftrs = self.model.fc.in_features
            # self.model.fc = nn.Linear(num_ftrs, num_classes)
            self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        else:
            raise NotImplementedError(f"model {model_name} is not implemented")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        - x: (B, 3, 224, 224)
        - out: (B, num_classif)
            - # valence, arousal, dominance, respectively
        """
        return self.model(x)

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