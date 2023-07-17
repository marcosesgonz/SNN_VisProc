import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from Datasets import DVSAnimals, DVSDailyActions, DVSActionRecog, DVS128Gesture
from models import myDVSGestureNet, mysew_resnet18
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from data_augmentation import EventMix

#set the seed for reproducibility
seed = 310
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.mps.deterministic = True
torch.backends.cuda.deterministic = True


def loading_data(input_data,time_step = 16 ,datatype = 'frame', splitmeth = 'number',tr_tst_split = True,tau_factor = 0.8,scale_factor = 50, data_aug_prob = 0):
    relative_root = os.path.basename(input_data)
    if relative_root == 'DVS_Gesture_dataset':
        train_set = DVS128Gesture(root = input_data, train = True, data_type = datatype, frames_number = time_step, 
                                  split_by = splitmeth, factor_tau = tau_factor, scale_factor = scale_factor) 
        test_set = DVS128Gesture(root = input_data, train = False, data_type = datatype, frames_number = time_step, 
                                 split_by = splitmeth, factor_tau = tau_factor, scale_factor = scale_factor) 
    elif relative_root == 'DVS_Animals_Dataset':
        train_set = DVSAnimals(root = input_data, train = True, data_type = datatype, frames_number = time_step,
                                    split_by = splitmeth, factor_tau = tau_factor, scale_factor = scale_factor) 
        test_set = DVSAnimals(root = input_data, train = False, data_type = datatype, frames_number = time_step, 
                                    split_by = splitmeth, factor_tau = tau_factor,scale_factor = scale_factor) 
    elif relative_root == 'DVS_DailyAction_dataset':
        train_set = DVSDailyActions(root = input_data,train = True, data_type = datatype, frames_number = time_step,
                                    split_by = splitmeth, factor_tau = tau_factor,scale_factor = scale_factor) 
        test_set = DVSDailyActions(root = input_data,train = False, data_type = datatype, frames_number = time_step,
                                    split_by = splitmeth, factor_tau = tau_factor,scale_factor = scale_factor) 
    elif relative_root == 'DVS_ActionRecog_dataset':
        train_set = DVSActionRecog(root = input_data,train = True, data_type = datatype, frames_number = time_step,
                                    split_by = splitmeth, factor_tau = tau_factor,scale_factor = scale_factor) 
        test_set = DVSActionRecog(root = input_data,train = False, data_type = datatype, frames_number = time_step,
                                    split_by = splitmeth, factor_tau = tau_factor,scale_factor = scale_factor) 
    else:
        raise ValueError('Unknown dataset. Could check name of the folder.')
    
    num_classes = len(train_set.classes)
    size_xy = train_set.get_H_W()
    if data_aug_prob != 0:
        #OJO, EventMix da ya las etiquetas con one_hot encoding al tener que mixear etiquetas para aumentar datos.
        train_set = EventMix(dataset=train_set, num_class = len(train_set.classes),num_mix = 1,
                             beta = 1, prob = data_aug_prob, noise = 0.05, gaussian_n = 3) 
        print('Using data augmentation with %.2f prob'%data_aug_prob)
    if tr_tst_split:
        return train_set,test_set,num_classes,size_xy
    else: 
        return ConcatDataset([train_set,test_set]), num_classes, size_xy


def load_net(net_name, n_classes, size_xy):
    if net_name == 'DVSG_net':
        net = myDVSGestureNet(channels=128, output_size = n_classes,input_sizexy= size_xy, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    elif net_name == 'resnet18':
        net = mysew_resnet18(spiking_neuron=neuron.LIFNode,num_classes = n_classes, surrogate_function=surrogate.ATan(), detach_reset=True,cnf='ADD',zero_init_residual=True)
    else:
        raise ValueError('Unknown arquitecture. Could check posible names. Names gift: ',net_name)
    #Establecemos las neuronas en modo multipaso
    functional.set_step_mode(net, 'm') 
    return net

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def train_model(net, n_classes, tr_loader, optimizer, device, lr_scheduler,data_augmented = False):
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in tr_loader: 
            optimizer.zero_grad()
            frame = frame.to(device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]  IMPORTANT, COULDN'T BE NECESSARY FOR SOME DATASETS

            label = label.to(device)

            if data_augmented:
                label_onehot = label
                num_batch_samples = len(label_onehot)      
            else:
                label_onehot = F.one_hot(label, n_classes).float()
                num_batch_samples = label.numel()

            out_fr = net(frame).mean(0)
            loss = F.mse_loss(out_fr, label_onehot)
            loss.backward()
            optimizer.step()

            train_samples += num_batch_samples
            train_loss += loss.item() * num_batch_samples

            if data_augmented:
                max_labels, pred_indices = label.topk(k=2, dim=1)  # Índices de los 2 máximos en la dimensión 1
                _, label_indices = out_fr.topk(k=2, dim=1)
                correct = 0.
                for i in range(num_batch_samples):
                    if max_labels[i,0] == 1: #Si esto pasa, significa que la etiqueta de la instancia está bien definida con una etiqueta
                        if pred_indices[i,0] == label_indices[i,0]:
                            correct += 1.
                    else:
                        if torch.all(pred_indices[i,:] == pred_indices[i,:]):
                            correct += 1.

                train_acc += correct
            else:
                train_acc += (out_fr.argmax(1) == label).float().sum().item() #Con argmax en la dimensión 1(.argmax(1)) estoy deshaciendo el onehot para cada etiqueta del batch 

            functional.reset_net(net)           #DUDA ¿Por qué no se resetean los pesos después de cada instancia en lugar después de haber pasado todo el batch?

        train_loss /= train_samples
        train_acc /= train_samples
        lr_scheduler.step()
        
        return train_loss,train_acc


def test_model(net, n_classes,tst_loader,
             optimizer,device,lr_scheduler):
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in tst_loader:
                frame = frame.to(device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.to(device)
                label_onehot = F.one_hot(label, n_classes).float()
                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_loss /= test_samples
        test_acc /= test_samples
        return test_loss,test_acc