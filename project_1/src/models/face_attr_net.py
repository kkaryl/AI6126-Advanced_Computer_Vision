import os

import torch.nn as nn
import torch
import shutil
import torchvision.models as tvmodels

__all__ = ['FaceAttrResNet']

class FC_Block(nn.Module):
    def __init__(self, inplanes, planes, drop_rate=0.15):
        super(FC_Block, self).__init__()
        self.fc = nn.Linear(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.relu(x)
        return x


class FaceAttrResNet(nn.Module):
    def __init__(self, resnet_layers, num_attributes=40):
        super(FaceAttrResNet, self).__init__()
        assert resnet_layers in [18, 34, 50, 101, 152]
        if resnet_layers == 18:
            pt_model = tvmodels.resnet18(pretrained=True)
        elif resnet_layers == 34:
            pt_model = tvmodels.resnet34(pretrained=True)
        elif resnet_layers == 50:
            pt_model = tvmodels.resnet50(pretrained=True)
        elif resnet_layers == 101:
            pt_model = tvmodels.resnet101(pretrained=True)
        elif resnet_layers == 152:
            pt_model = tvmodels.resnet152(pretrained=True)
        pt_out = pt_model.fc.out_features
        self.pretrained = pt_model
        self.num_attributes = num_attributes
        for i in range(num_attributes):
            setattr(self, 'classifier' + str(i).zfill(2),
                    nn.Sequential(FC_Block(pt_out, pt_out // 2),
                                  nn.Linear(pt_out // 2, 2)))
        self.name = 'FaceAttrResNet'+str(resnet_layers)

    def forward(self, x):
        x = self.pretrained(x)

        y = []
        for i in range(self.num_attributes):
            classifier = getattr(self, 'classifier' + str(i).zfill(2))
            y.append(classifier(x))
        return y

    def save_ckp(self, state, is_best, checkpoint_path, bestmodel_path):
        print("=> saving checkpoint '{}'".format(checkpoint_path))
        torch.save(state, checkpoint_path)
        if is_best:
            best_fpath = bestmodel_path
            # copy that checkpoint file to best path given, best_model_path
            shutil.copyfile(checkpoint_path, best_fpath)

    def load_ckp(self, optimizer, checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        self.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
        return optimizer, start_epoch, best_prec1