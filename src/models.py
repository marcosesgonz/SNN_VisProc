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
        nconv_blocks = 5         # = int(np.min(np.log2(np.array(input_sizexy) / desired_output))) , desired_ouput = 4
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

    def forward(self, x: torch.Tensor):  #x.shape = (T, N, C, H, W)
        x = self.conv_fc(x)
        x = x.mean(0)
        return x
    
class myDVSGestureANN(nn.Module):
    def __init__(self, channels=128, output_size = 11, input_sizexy =(128,128),softm = True):
        super().__init__()
        conv = []
        nconv_blocks = 5
        print('Number of conv_blocks:',nconv_blocks)
        for i in range(nconv_blocks):
            if conv.__len__() == 0:
                in_channels = 1
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
            nn.Dropout(0.5)
        )

        
        if softm:
            self.classifier= nn.Sequential(     
                nn.Linear(512, output_size), 
                nn.Softmax(dim = 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512,output_size*10), 
                nn.ReLU(),
                nn.AvgPool1d(10, 10)
            )


    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(2)      #Adding the channel dimension since data from e2vid doesnt include it
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        x = self.conv_fc(x)
        x = x.view(batch_size, num_frames, -1)  # Volver a la forma secuencial
        # Promedio entre todos los pasos temporales
        print('Antes del clasificador',x.size())
        out = self.classifier(x).mean(1)
        print('Después del clasificador promediando T:',out.size())
        return out



class myDVSGestureRANN(nn.Module):
    def __init__(self, channels=128, output_size = 11, input_sizexy =(128,128),softm = True):
        super().__init__()
        conv = []
        nconv_blocks = 5
        print('Number of conv_blocks:',nconv_blocks)
        for i in range(nconv_blocks):
            if conv.__len__() == 0:
                in_channels = 1
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
            nn.Dropout(0.5)
        )

        
        if softm:
            #Recurrent layer with GRU. Inputsize indica el tamaño de entrada y hidden size el tamaño del estado oculto además del tamaño del estado de salida
            self.gru = nn.GRU(input_size=512, hidden_size = output_size, num_layers=1, batch_first=True)
            self.classifier= nn.Sequential(      
                nn.Softmax(dim = 1)
            )
        else:
            self.gru = nn.GRU(input_size=512, hidden_size = output_size * 10, num_layers=1, batch_first=True)
            self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.AvgPool1d(10, 10)
            )


    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(2)      #Adding the channel dimension since data from e2vid doesnt include it
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        x = self.conv_fc(x)
        x = x.view(batch_size, num_frames, -1)  # Volver a la forma secuencial

        # Pasar los frames secuencialmente a la capa GRU. El estado inicial de la celda GRU h_0 será la que da por defecto(torch.zeros(..))
        out, hidden = self.gru(x)
        # Utilizar el último estado oculto como entrada para la capa totalmente conectada final
        out = self.classifier(out[:, -1, :])

        return out


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


from spikingjelly.clock_driven.neuron import MultiStepLIFNode
#from spikingjelly.clock_driven import layer, surrogate
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial



class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.c_hidden = hidden_features
        self.c_output = out_features
    def forward(self, x):
        T,B,C,N = x.shape
        x = self.fc1_conv(x.flatten(0,1))
        x = self.fc1_bn(x).reshape(T,B,self.c_hidden,N).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0,1))
        x = self.fc2_bn(x).reshape(T,B,C,N).contiguous()
        x = self.fc2_lif(x)
        return x

class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)

        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.attn_drop = nn.Dropout(0.2)
        self.res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T,B,C,N))

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + (self.mlp(x))
        return x


class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_conv(x.flatten(0, 1)) # have some fire value
        x = self.proj_bn(x).reshape(T,B,-1,H,W).contiguous()
        x = self.proj_lif(x).flatten(0,1).contiguous()
        x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, 64, 64).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()
        x = self.maxpool1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, 32, 32).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, 16, 16).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        x_rpe = self.rpe_bn(self.rpe_conv(x)).reshape(T,B,-1,H//16,W//16).contiguous()
        x_rpe = self.rpe_lif(x_rpe).flatten(0,1)
        x = x + x_rpe
        x = x.reshape(T, B, -1, (H//16)*(H//16)).contiguous()

        return x

class Spikformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2]
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims)
        num_patches = patch_embed.num_patches
        pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"pos_embed", pos_embed)
        setattr(self, f"block", block)

        self.norm = norm_layer(embed_dims)
        # classification head 这里不需要脉冲，因为输入的是在T时长平均发射值
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()

        pos_embed = getattr(self, f"pos_embed")
        trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        x = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.mean(3)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


@register_model
def spikformer(pretrained=False, **kwargs):
    model = Spikformer(
        patch_size=16, embed_dims=256, num_heads=16, mlp_ratios=4,
        in_channels=2, num_classes=10, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=2, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model