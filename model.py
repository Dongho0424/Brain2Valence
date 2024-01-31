import torch 
import torch.nn as nn
from monai.networks.nets import resnet


class Brain2ValenceModel(nn.Module):
    def __init__(self):
        super().__init__()    
        
        # TODO: using MONAI
        # self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=pretrained)
        """
        def __init__(
        self,
        block: type[ResNetBlock | ResNetBottleneck] | str,
        layers: list[int],
        block_inplanes: list[int],
        """
        self.model = resnet.ResNet(
            block='basic',
            layers=[3,3,3,3],
            block_inplanes=[3,3,3,3],
            num_classes=400
        )
        self.backbone = nn.Sequential(*list(self.model.children())[:-1])
        self.fc_valence = nn.Linear(self.model.fc.in_features, 1)
            
    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1).float()

        valence = self.fc_valence(x)
        
        return valence.float().squeeze()