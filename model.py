import torch
import torch.nn as nn
from monai.networks.nets import resnet   


class Brain2ValenceModel(nn.Module):
    def __init__(self, model: str = "resnet18"):
        super().__init__()

        print("current model backbone:", model)

        if model == "resnet18":
            self.model = resnet.resnet18(n_input_channels=1, num_classes=1, feed_forward=True)
        elif model == "resnet50":
            self.model = resnet.resnet50(n_input_channels=1, num_classes=1, feed_forward=True)
        else:
            raise NotImplementedError(f"model {model} is not implemented")
        
        self.model_name = model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 96, 96, 96)
        out: (B, 1)
        """
        # add channel dimension
        x = x.unsqueeze(dim=1)  # (B, 96, 96, 96) -> (B, 1, 96, 96, 96)
        valence = self.model(x) 

        return valence.float().squeeze(dim=-1)