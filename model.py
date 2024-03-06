import torch
import torch.nn as nn
import utils
from typing import List
from monai.networks.nets import resnet
from resnet import ResNetwClf

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

        if model_name == "resnet18" or model_name == "resnet50":
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