import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from spikingjelly.activation_based.model import sew_resnet as sewr
import numpy as np
from copy import deepcopy

#Net for Experiment 1. ES la red que propone spikingjelly para DVSNet
class myDVSGestureNet(nn.Module):
    def __init__(self, channels=128, output_size = 11, input_sizexy =(128,128),spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        desired_output = 4
        nconv_blocks = int(np.min(np.log2(np.array(input_sizexy) / desired_output))) #Aquí ponía 5
        print('Number of conv_blocks:',nconv_blocks)
        for i in range(nconv_blocks):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))

        outp_convx = input_sizexy[0] // (2**nconv_blocks)
        outp_convy = input_sizexy[1] // (2**nconv_blocks)

        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            #En caso de (128,128) de entrada. Lo multiplico por 4x4 debido a que los 5 max pooling(2,2) pasan las matrices de (128,128) a (4,4).
            layer.Linear(channels * outp_convx * outp_convy, 512), 
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            #La última capa cambia su tamaño en función del número de clases posibles. De esta manera la capa de votación(VotingLayer) la dejamos fijas con 10 neuronas por voto('maxpooling' de 10)
            layer.Linear(512, output_size * 10),   
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)  
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)
    
class myDVSGestureNetANN(nn.Module):
    def __init__(self, channels=128, output_size = 11, input_sizexy =(128,128),spiking_neuron: callable = None, **kwargs):
        super().__init__()
        conv = []
        desired_output = 4
        nconv_blocks = int(np.min(np.log2(np.array(input_sizexy) / desired_output))) #Aquí ponía 5
        print('Number of conv_blocks:',nconv_blocks)
        for i in range(nconv_blocks):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(nn.BatchNorm2d(channels))
            conv.append(nn.ReLU())
            conv.append(nn.MaxPool2d(2, 2))

        outp_convx = input_sizexy[0] // (2**nconv_blocks)
        outp_convy = input_sizexy[1] // (2**nconv_blocks)

        self.conv_fc = nn.Sequential(
            *conv,

            nn.Flatten(),
            nn.Dropout(0.5),
            #En caso de (128,128) de entrada. Lo multiplico por 4x4 debido a que los 5 max pooling(2,2) pasan las matrices de (128,128) a (4,4).
            nn.Linear(channels * outp_convx * outp_convy, 512), 
            nn.ReLU(),

            nn.Dropout(0.5),
            #La última capa cambia su tamaño en función del número de clases posibles. De esta manera la capa de votación(VotingLayer) la dejamos fijas con 10 neuronas por voto('maxpooling' de 10)
            nn.Linear(512, output_size * 10),   
            nn.ReLU(),
            
            layer.VotingLayer(10)  
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


#I've changed the input channels of the first conv layer from 3 to 2.
class mySEWResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, cnf: str = None, spiking_neuron: callable = None, **kwargs):
        super().__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3,   #Here I change from 3 to 2 channels in the input channels.
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, sewr.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, sewr.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cnf: str=None, spiking_neuron: callable = None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                sewr.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, cnf, spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)



def _mysew_resnet(arch, block, layers, pretrained, progress, cnf, spiking_neuron, **kwargs):
    model = mySEWResNet(block, layers, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
    if pretrained:
        state_dict = sewr.load_state_dict_from_url(sewr.model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def mysew_resnet18(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-18
    :rtype: torch.nn.Module

    The spike-element-wise ResNet-18 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_ modified by the ResNet-18 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """

    return _mysew_resnet('resnet18', sewr.BasicBlock, [2, 2, 2, 2], pretrained, progress, cnf, spiking_neuron, **kwargs)
