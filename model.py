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
            self.lin = nn.Sequential(
                nn.Linear(features, 4096, bias=False),
                nn.LayerNorm(4096), 
                nn.GELU(inplace=True),
                nn.Dropout(0.5),
            )
            self.mlp = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(4096, 4096, bias=False),
                    nn.LayerNorm(1024), 
                    nn.GELU(inplace=True),
                    nn.Dropout(0.15),
                ),
                nn.Sequential(
                    nn.Linear(4096, 1024, bias=False),
                    nn.LayerNorm(1024), 
                    nn.GELU(inplace=True),
                    nn.Dropout(0.15),
                ),
                nn.Sequential(
                    nn.Linear(1024, 128, bias=False),
                    nn.LayerNorm(128), 
                    nn.GELU(inplace=True),
                    nn.Dropout(0.15),
                ),
                ]
            )
            self.proj = nn.Linear(128, num_classes, bias=False),
        else:
            raise NotImplementedError(f"model {model_name} is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        - x: 
            - brain3d: (B, 96, 96, 96)
            - roi: (B, *)
        - out: (B, 1)
        """

        if self.model_name in ["resnet18", "resnet50"]:
            # add channel dimension
            x = x.unsqueeze(dim=1)  # (B, 96, 96, 96) -> (B, 1, 96, 96, 96)
            valence = self.res_model(x)  # (B, 1) or (B, num_classif)
        elif self.model_name == "mlp":
            x = self.lin(x)
            residual = x
            for block in range(len(self.mlp)):
                x = self.mlp[block](x) + residual
                residual = x
            valence = self.proj(x)
        return valence.float().squeeze(dim=-1)