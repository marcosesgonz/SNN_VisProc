import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from Datasets import DVSAnimals, DVSDailyActions, DVSActionRecog, DVS128Gesture, MAD
from models import myDVSGestureNet, mysew_resnet18, myDVSGestureRANN, myDVSGestureANN, myDVSGesture3DANN
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from data_augmentation import EventMix

#set the seed for reproducibility
seed = 310
try:
    import cupy as cp
    cp.random.seed(seed)
except:
    cp = None
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.mps.deterministic = True
torch.backends.cuda.deterministic = True

def scan_corrupted_files(frames_root,min_value,max_value):
    """
    Important scanning function to check outlayers in FRAMES PROCESSED WITH EXP_DECAY splitby method. This function scan folders with the same composition of datasets folders 'train' and 'test'.
    """
    corrupted_files = []
    for class_ in [it.name for it in os.scandir(frames_root) if it.is_dir()]:
        for sample in os.listdir(os.path.join(frames_root,class_)):
            if sample.endswith('.npz'):
                sample_root = os.path.join(frames_root,class_,sample)
                l = np.load(sample_root, allow_pickle=True)['frames'].astype(np.float32)
                unique_values = np.unique(l)
                if not (np.all(unique_values >= min_value) and np.all(unique_values <= max_value)):
                    print(f'Sample {sample_root} possibly corrupted.')
                    corrupted_files.append(sample_root)
                elif len(unique_values) == 1:
                    print(f'Warning. Only {unique_values} encountered in {sample_root}.')
                    corrupted_files.append(sample_root)
                else:
                    for individual_frame in l:
                        if len(np.unique(individual_frame)) == 1:
                            print(f'Warning. Only one value encountered in frame {individual_frame} of {sample_root}.')
                            corrupted_files.append(sample_root)
        
        return (corrupted_files if len(corrupted_files) > 0 else  None)



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
    elif relative_root == 'MAD_dataset':
        train_set = MAD(root = input_data, train = True, test_subj_id = 1, data_type = datatype, frames_number = time_step,
                        split_by = splitmeth, factor_tau = tau_factor, scale_factor = scale_factor)
        test_set = MAD(root = input_data, train = False, test_subj_id = 1, data_type = datatype, frames_number = time_step,
                        split_by = splitmeth, factor_tau = tau_factor, scale_factor = scale_factor)
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
        if relative_root == 'MAD_dataset':   #En verdad, es equivalente usar ConcatDataset o llamar de nuevo a la función MAD sin especificar 'test_subj_id'
            data_set = MAD(root = input_data, data_type = datatype, frames_number = time_step,
                        split_by = splitmeth, factor_tau = tau_factor, scale_factor = scale_factor)
            return data_set, num_classes, size_xy
        else:
            return ConcatDataset([train_set,test_set]), num_classes, size_xy


def load_net(net_name: str, n_classes: int, size_xy: tuple, neuron_type: str = 'LIF' ,cupy: bool = False, num_frames: int = 16, softm: bool = True):

    possible_nets = ['DVSG_net','resnet18','DVSG_RANN','DVSG_ANN', 'DVSG_3DANN']
    assert (net_name in possible_nets), 'Unknown arquitecture. Could check posible names.'

    if not net_name.endswith('ANN'):
        if neuron_type == 'IF':
            neuron_model = neuron.IFNode
        elif neuron_type == 'LIF':
            neuron_model = neuron.LIFNode
        elif neuron_type == 'PLIF':
            neuron_model = neuron.ParametricLIFNode
        else:
            raise NotImplementedError('Possible values implemented: IF, LIF, PLIF')
        print(f'Using {neuron_type} neurons')

        if net_name == 'DVSG_net':
            net = myDVSGestureNet(channels=128, output_size = n_classes,input_sizexy= size_xy, spiking_neuron = neuron_model, surrogate_function=surrogate.ATan(), detach_reset=True)
        elif net_name == 'resnet18':
            net = mysew_resnet18(spiking_neuron = neuron_model,num_classes = n_classes, surrogate_function=surrogate.ATan(), detach_reset=True,cnf='ADD',zero_init_residual=True)

        #Establecemos las neuronas en modo multipaso
        functional.set_step_mode(net, 'm')
        if cupy:
            functional.set_backend(net, 'cupy', instance = neuron_model)
            print('Using cupy in backend')
    else:
        if net_name == 'DVSG_RANN':
            net = myDVSGestureRANN(output_size = n_classes,input_sizexy=size_xy, softm = softm)
        elif net_name == 'DVSG_ANN':
            net = myDVSGestureANN(output_size = n_classes,input_sizexy=size_xy, softm = softm)
        elif net_name == 'DVSG_3DANN':
            net = myDVSGesture3DANN(output_size = n_classes,input_sizexy = size_xy, num_frames = num_frames,softm = softm)
        
    return net

def num_trainable_params(net):
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return trainable_params
    

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def train_model(net, n_classes, tr_loader, optimizer, device, lr_scheduler, SNNmodel = True, data_augmented = False):
        net.train()
        train_loss, train_acc, train_samples = 0, 0, 0
        for frame, label in tr_loader: 
            optimizer.zero_grad()
            frame = frame.to(device)
            if SNNmodel:
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]  IMPORTANT, COULDN'T BE NECESSARY FOR SOME DATASETS
            label = label.to(device)

            if data_augmented:
                label_onehot = label
                num_batch_samples = len(label_onehot)      
            else:
                label_onehot = F.one_hot(label, n_classes).float()
                num_batch_samples = label.numel()

 
            output = net(frame)       #Firing rate of spiking neurons in all time steps if SNN
            loss = F.mse_loss(output, label_onehot)
            loss.backward()
            optimizer.step()

            train_samples += num_batch_samples
            train_loss += loss.item() * num_batch_samples

            if data_augmented:
                max_labels, pred_indices = label.topk(k=2, dim=1)  # Índices de los 2 máximos en la dimensión 1
                _, label_indices = output.topk(k=2, dim=1)
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
                train_acc += (output.argmax(1) == label).float().sum().item() #Con argmax en la dimensión 1(.argmax(1)) estoy deshaciendo el onehot para cada etiqueta del batch 

            if SNNmodel:
                functional.reset_net(net)           

        train_loss /= train_samples
        train_acc /= train_samples
        lr_scheduler.step()
        
        return train_loss,train_acc


def test_model(net, n_classes,tst_loader,
             device,SNNmodel = True):
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in tst_loader:
                frame = frame.to(device)
                if SNNmodel:
                    frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.to(device)
                label_onehot = F.one_hot(label, n_classes).float()
                out_fr = net(frame)
                loss = F.mse_loss(out_fr, label_onehot)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                if SNNmodel:
                    functional.reset_net(net)
        test_loss /= test_samples
        test_acc /= test_samples
        return test_loss,test_acc


