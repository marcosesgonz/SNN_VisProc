import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from copy import deepcopy

#Net for Experiment 1. ES la red que propone spikingjelly para DVSNet
class myDVSGestureNet(nn.Module):
    def __init__(self, channels=128, output_size = 11, input_sizexy =(128,128),spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        nconv_blocks = 5
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
        print(outp_convx,outp_convy)

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