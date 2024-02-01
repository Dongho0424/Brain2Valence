import torch 
import torch.nn as nn
from monai.networks.nets import resnet


class Brain2ValenceModel(nn.Module):
    def __init__(self):
        super().__init__()    
        
        self.model = resnet.resnet18(n_input_channels=1,  num_classes=1, feed_forward=True)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 96, 96, 96)
        out: (B, 1)
        """
        # add channel dimension
        x = x.unsqueeze(dim=1) # (B, 96, 96, 96) -> (B, 1, 96, 96, 96)
        valence = self.model(x)
        
        return valence.float().squeeze(dim=-1)