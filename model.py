import torch
import torch.nn as nn
from monai.networks.nets import resnet   


class Brain2ValenceModel(nn.Module):
    def __init__(self, model_name: str = "resnet18", task_type: str = "reg", num_classif: int = 3):
        super().__init__()

        self.model_name = model_name
        print("current model backbone:", model_name)

        num_classes = num_classif if task_type == "classif" else 1 # when regression, num_classes = 1

        if model_name == "resnet18":
            self.model = resnet.resnet18(n_input_channels=1, num_classes=num_classes, feed_forward=True)
        elif model_name == "resnet50":
            self.model = resnet.resnet50(n_input_channels=1, num_classes=num_classes, feed_forward=True)
        else:
            raise NotImplementedError(f"model {model_name} is not implemented")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 96, 96, 96)
        out: (B, 1)
        """
        # add channel dimension
        x = x.unsqueeze(dim=1)  # (B, 96, 96, 96) -> (B, 1, 96, 96, 96)
        valence = self.model(x) 

        return valence.float().squeeze(dim=-1)