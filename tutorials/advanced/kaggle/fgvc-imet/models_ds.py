from collections import OrderedDict
from functools import partial

import torch
import torchvision.models as M
from torch import nn
from torch.nn import functional as F

import se_resnet as SEM
from utils import ON_KAGGLE


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def create_net(net_cls, pretrained: bool):
    if ON_KAGGLE and pretrained:
        net = net_cls()
        model_name = net_cls.__name__
        weights_path = f'../input/{model_name}/{model_name}.pth'
        net.load_state_dict(torch.load(weights_path))
    else:
        net = net_cls(pretrained=pretrained)
    return net


class ResNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.resnet101, dropout=0.1):
        super().__init__()
        self.resnet = create_net(net_cls, pretrained=pretrained)
        self.fc3 = nn.Sequential(OrderedDict(
            [
                # ('bn1', nn.BatchNorm1d(2 * (1024))),
                ('dropout1', nn.Dropout(dropout)),
                ('fc1', nn.Linear(2 * (1024), num_classes))
            ]
        ))

        self.fc4 = nn.Sequential(OrderedDict(
            [
                # ('bn1', nn.BatchNorm1d(2 * (2048))),
                ('dropout1', nn.Dropout(dropout)),
                ('fc1', nn.Linear(2 * (2048), num_classes))
            ]
        ))
        self.logits = nn.Sequential(OrderedDict(
            [
                # ('bn1', nn.BatchNorm1d(2 * num_classes)),
                ('relu1', nn.ELU(inplace=True)),
                ('dropout1', nn.Dropout(dropout)),
                ('fc1', nn.Linear(2 * (num_classes), num_classes)),
            ]
        ))


    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)


        x = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        # x2 = torch.cat([F.adaptive_avg_pool2d(x2, 1), F.adaptive_max_pool2d(x2, 1)], -1)
        # x2 = x2.view(-1, 512 * 2)
        # x2 = self.fc2(x2)
        #
        x3 = torch.cat([F.adaptive_avg_pool2d(x3, 1), F.adaptive_max_pool2d(x3, 1)], -1)
        x3 = x3.view(-1, 1024 * 2)
        x3 = self.fc3(x3)
        #
        x4 = torch.cat([F.adaptive_avg_pool2d(x4, 1), F.adaptive_max_pool2d(x4, 1)], -1)
        x4 = x4.view(-1, 2048 * 2)
        x4 = self.fc4(x4)
        #
        x = torch.cat([x3, x4], -1)
        x = self.logits(x)
        return x

    def freeze(self):
        for p in self.resnet.parameters():
            p.requires_grad = False
        return True

    def unfreeze(self, name=None):
        unfreezed_parameters = []
        if name is not None and name != 'all':
            for c_name, child in self.resnet.named_children():
                if c_name == name:
                    for p in child.parameters():
                        if not p.requires_grad:
                            unfreezed_parameters.append(p)
                        p.requires_grad = True
        elif name == 'all':
            for p in self.resnet.parameters():
                if not p.requires_grad:
                    unfreezed_parameters.append(p)
                p.requires_grad = True
        else:
            return unfreezed_parameters
        return unfreezed_parameters

def init_weights(module):
    if type(module) == nn.Linear:
        torch.nn.init.kaiming_normal_(module.weight)
        module.bias.data.fill_(0.01)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class SeResNext(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=SEM.se_resnext50_32x4d, dropout=0.3):
        super().__init__()
        self.resnet = create_net(net_cls, pretrained=pretrained)
        # self.avgpool = F.adaptive_avg_pool2d()
#        self.fc2 = nn.Sequential(OrderedDict(
#            [
#                # ('bn1', nn.BatchNorm1d(2 * (512 + 1024 + 2048))),
#                ('dropout1', nn.Dropout(dropout)),
#                ('fc1', nn.Linear(2 * (512), num_classes)),
#                # ('relu1', nn.ReLU(inplace=True)),
#            ]
#        ))

        self.fc3 = nn.Sequential(OrderedDict(
            [
                # ('bn1', nn.BatchNorm1d(2 * (512 + 1024 + 2048))),
                ('dropout1', nn.Dropout(dropout)),
                ('fc1', nn.Linear(2 * (1024), num_classes)),
                # ('relu1', nn.ReLU(inplace=True)),
            ]
        ))

        self.fc4 = nn.Sequential(OrderedDict(
            [
                # ('bn1', nn.BatchNorm1d(2 * (512 + 1024 + 2048))),
                ('dropout1', nn.Dropout(dropout)),
                ('fc1', nn.Linear(2 * (2048), num_classes)),
                # ('relu1', nn.ReLU(inplace=True)),
            ]
        ))

        self.logits = nn.Sequential(OrderedDict(
            [
                # ('bn1', nn.BatchNorm1d(2 * 2048)),
                ('elu', nn.ELU()),
                ('dropout1', nn.Dropout(dropout)),
                ('fc1', nn.Linear(2 * (num_classes), num_classes)),
            ]
        ))

        # self.fc2.apply(init_weights)
        # self.fc3.apply(init_weights)
        # self.fc4.apply(init_weights)
        # self.logits.apply(init_weights)

    def freeze(self):
        for p in self.resnet.parameters():
            p.requires_grad = False
        return True

    def unfreeze(self, name=None):
        unfreezed_parameters = []
        if name is not None and name != 'all':
            for c_name, child in self.resnet.named_children():
                if c_name == name:
                    for p in child.parameters():
                        if not p.requires_grad:
                            unfreezed_parameters.append(p)
                        p.requires_grad = True
        elif name == 'all':
            for p in self.resnet.parameters():
                if not p.requires_grad:
                    unfreezed_parameters.append(p)
                p.requires_grad = True
        else:
            return unfreezed_parameters
        return unfreezed_parameters

    def forward(self, x):
        # batch_size = x.shape[0]
        x = self.resnet.layer0(x)
        x = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        # x2 = torch.cat([F.adaptive_avg_pool2d(x2, 1), F.adaptive_max_pool2d(x2, 1)], -1)
        # x2 = x2.view(-1, 512 * 2)
        # x2 = self.fc2(x2)
        #
        x3 = torch.cat([F.adaptive_avg_pool2d(x3, 1), F.adaptive_max_pool2d(x3, 1)], -1)
        x3 = x3.view(-1, 1024 * 2)
        x3 = self.fc3(x3)
        #
        x4 = torch.cat([F.adaptive_avg_pool2d(x4, 1), F.adaptive_max_pool2d(x4, 1)], -1)
        x4 = x4.view(-1, 2048 * 2)
        x4 = self.fc4(x4)
        #
        x = torch.cat([x3, x4], -1)
        x = self.logits(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.densenet121):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.avg_pool = AvgPool()
        self.net.classifier = nn.Linear(
            self.net.classifier.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.net.classifier(out)
        return out


resnet18 = partial(ResNet, net_cls=M.resnet18)
resnet34 = partial(ResNet, net_cls=M.resnet34)
resnet50 = partial(ResNet, net_cls=M.resnet50)
resnet101 = partial(ResNet, net_cls=M.resnet101)
resnet152 = partial(ResNet, net_cls=M.resnet152)

densenet121 = partial(DenseNet, net_cls=M.densenet121)
densenet169 = partial(DenseNet, net_cls=M.densenet169)
densenet201 = partial(DenseNet, net_cls=M.densenet201)
densenet161 = partial(DenseNet, net_cls=M.densenet161)

seresnext50 = partial(SeResNext, net_cls=SEM.se_resnext50_32x4d)
seresnext101 = partial(SeResNext, net_cls=SEM.se_resnext101_32x4d)
senet = partial(SeResNext, net_cls=SEM.senet154)

