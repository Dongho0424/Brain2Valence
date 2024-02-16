import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from os.path import join, expanduser

#TODO: dataloader 등 뿌려주는 거 다시 봐야함. (디버깅)

## EMOTIC dataset
## NSD와 겹치는 데이터로
#TODO later 1. 이미지 -> 감정 예측
#TODO later 2. 이미지 + 뇌사진 -> 감정 예측 (멀티 모달)
# 이미지 -> cnn -> 512
# roi -> mlp -> 512
# 1024 -> mlp -> classification 

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if 'cuda' in out.data.type():
        # if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = torch.cat([out.data, zero_pads], dim=1)

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], 'B', **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], 'A', **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], 'A', **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], 'B', **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], 'B', **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], 'B', **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], 'B', **kwargs)
    return model


backbone_dict = {
    'resnet_10': resnet10,
    'resnet_18': resnet18,
    'resnet_34': resnet34,
    'resnet_50': resnet50,
}


class ResNetwClf(nn.Module):
    def __init__(self, backbone_type, num_classes, pretrained=False, mri_modelpath=None, decouple=False,
                 target='ad_transition', backbone_freeze=False, **kwargs):
        super(ResNetwClf, self).__init__()
        self.backbone = backbone_dict[backbone_type](**kwargs)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        num_outputs = 1 if num_classes == 2 else num_classes
        self.decouple = decouple
        self.target = target
        
        self.backbone_freeze = backbone_freeze
        print(f'backbone_freeze: {backbone_freeze}')
        if pretrained:
            self._load_pretrained_models(backbone_type, mri_modelpath=mri_modelpath)

        if decouple or target == 'ad_transition_w_time':
            if decouple and target =='ad_transition_w_time':
                self.clf = nn.ModuleList()
                for i in range(2):
                    if i == 0:
                        transition_clf = nn.ModuleList()
                        for _ in range(2):
                            transition_clf.append(nn.Linear(512, num_outputs))
                        self.clf.append(transition_clf)
                    else:
                        self.clf.append(nn.Linear(512, num_outputs))
            else:
                self.clf = nn.ModuleList()
                for i in range(2):
                    self.clf.append(nn.Linear(512, num_outputs))
        else:
            self.clf = nn.Linear(512, num_outputs)


    def _load_pretrained_models(self, backbone_type, mri_modelpath=None):

        
        if mri_modelpath is not None:
            mri_emb_weight = torch.load(mri_modelpath)
            # 1. delete items in mri_emb_weight whose keys start with 'clf'
            for k in list(mri_emb_weight.keys()):
                if k.startswith('clf'):
                    del mri_emb_weight[k]
            print(mri_emb_weight.keys())
            # self.backbone.load_state_dict(mri_emb_weight)
            self.load_state_dict(mri_emb_weight)
            if self.backbone_freeze:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            # for n, p in self.backbone.named_parameters():
            #     assert p.data.shape == mri_emb_weight['backbone.' + n].shape
            #     p.data = mri_emb_weight['backbone.' + n]
            #     if self.backbone_freeze:
            #         p.requires_grad = False
        else:
            home = expanduser("~")
            weight_path = join(home,
                               f"Data/BigBrain/ADNI/MedicalNet_pytorch_files2/pretrain/{backbone_type}_23dataset.pth")
            pretrained_weight = torch.load(weight_path)['state_dict']
            for k in pretrained_weight.copy().keys():
                if k.startswith('module.'):
                    pretrained_weight[k[7:]] = pretrained_weight.pop(k)
            self.backbone.load_state_dict(pretrained_weight)

    def forward(self, x, demo_feature=None, get_inter=False):
        fea = self.backbone(x)
        fea = self.avgpool(fea)
        fea = fea.view(fea.size(0), -1)

        if self.target == 'ad_transition_w_time':
            output_list = []

            for i in range(2):
                if i == 0:
                    if self.decouple:
                        _, age, _ = demo_feature
                        age_mask = (age >= 75).cuda()
                        tmp = 0
                        for j in range(2):
                            tmp += self.clf[i][j](fea).squeeze() * (j == age_mask)
                        out = tmp.squeeze()
                    else:
                        out = self.clf[i](fea).squeeze()
                else:
                    out = self.clf[i](fea).squeeze() # head 0 for ad_transition head 1 for ad_time
                output_list.append(out)
            y_hat = output_list

        else:
            if self.decouple:
                _, age, _ = demo_feature
                age_mask = (age >= 75).cuda()
                tmp = 0
                for i in range(2):
                    tmp += self.clf[i](fea).squeeze() * (i == age_mask)
                y_hat = tmp.squeeze()
            else:
                y_hat = self.clf(fea).squeeze()

        if get_inter:
            return fea, y_hat
        else:
            return y_hat