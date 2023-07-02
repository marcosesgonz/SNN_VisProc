import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from copy import deepcopy

#Net for Experiment 1. ES la red que propone spikingjelly para DVSNet
class myDVSGestureNet(nn.Module):
    def __init__(self, channels=128, output_size = 11, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            #La última capa cambia su tamaño en función del número de clases posibles. De esta manera la capa de votación(VotingLayer) la dejamos fijas con 10 neuronas por voto('maxpooling' de 10)
            layer.Linear(512, output_size * 10),   
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)  
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)