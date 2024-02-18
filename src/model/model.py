import torch
import torch.nn as nn
from torchvision import models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias = True,
                 use_bn = False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn

        self.linear = nn.Linear(self.in_features,
                                self.out_features,
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type = 'nonlinear',
                 **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features,self.out_features,False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features,self.hidden_features,True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features,self.out_features,False,True))

    def forward(self,x):
        x = self.layers(x)
        return x

class Model(nn.Module):
    def __init__(self,cfg,base_model,base_out_layer):
        super().__init__()
        self.cfg = cfg
        self.base_model = base_model
        self.base_out_layer = base_out_layer

        #PRETRAINED MODEL
        self.pretrained = models.resnet50(pretrained=True)

        self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.pretrained.maxpool = Identity()

        self.pretrained.fc = Identity()

        for p in self.pretrained.parameters():
            p.requires_grad = True

        self.projector = ProjectionHead(self.cfg.MODEL.in_features,
                                        self.cfg.MODEL.hidden_features,
                                        self.cfg.MODEL.out_features)


    def forward(self,x):
        out = self.pretrained(x)

        xp = self.projector(torch.squeeze(out))

        return xp
    
class Downstream_Model(nn.Module):
    def __init__(self,cfg,premodel,num_classes):
        super().__init__()

        self.cfg = cfg
        self.premodel = premodel

        self.num_classes = num_classes

        #TAKING OUTPUT FROM AN INTERMEDITATE LAYER
        #PREPRAING THE TRUNCATED MODEL

        for p in self.premodel.parameters():
            p.requires_grad = False

        for p in self.premodel.projector.parameters():
            p.requires_grad = False


        self.lastlayer = nn.Linear(self.cfg.MODEL.in_features,self.num_classes)



    def forward(self,x):
        out = self.premodel.pretrained(x)

        out = self.lastlayer(out)
        return out