import torch
import sys
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import wandb
import random
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from Datasets import DVSAnimals, DVSDailyActions, DVSActionRecog, DVS128Gesture
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from models import myDVSGestureNet, mysew_resnet18
from data_augmentation import EventMix
import numpy as np
import time
import os
import datetime

#set the seed for reproducibility
seed = 310
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.mps.deterministic = True
torch.backends.cuda.deterministic = True

data_dir = '/Users/marcosesquivelgonzalez/Desktop/MasterCDatos/TFM/data/DVS_Gesture_dataset'

def loading_data(input_data,time_step = 16 ,splitmeth = 'number',tr_tst_split = True,tau_factor = 0.8,scale_factor = 50, data_aug_prob = 0):
    relative_root = os.path.basename(input_data)
    if relative_root == 'DVS_Gesture_dataset':
        train_set = DVS128Gesture(root = input_data, train = True, data_type = 'frame', frames_number = time_step, 
                                  split_by = splitmeth, factor_tau = tau_factor, scale_factor = scale_factor) 
        test_set = DVS128Gesture(root = input_data, train = False, data_type = 'frame', frames_number = time_step, 
                                 split_by = splitmeth, factor_tau = tau_factor, scale_factor = scale_factor) 
    elif relative_root == 'DVS_Animals_Dataset':
        train_set = DVSAnimals(root = input_data, train = True, data_type = 'frame', frames_number = time_step,
                                    split_by = splitmeth, factor_tau = tau_factor, scale_factor = scale_factor) 
        test_set = DVSAnimals(root = input_data, train = False, data_type = 'frame', frames_number = time_step, 
                                    split_by = splitmeth, factor_tau = tau_factor,scale_factor = scale_factor) 
    elif relative_root == 'DVS_DailyAction_dataset':
        train_set = DVSDailyActions(root = input_data,train = True, data_type = 'frame', frames_number = time_step,
                                    split_by = splitmeth, factor_tau = tau_factor,scale_factor = scale_factor) 
        test_set = DVSDailyActions(root = input_data,train = False, data_type = 'frame', frames_number = time_step,
                                    split_by = splitmeth, factor_tau = tau_factor,scale_factor = scale_factor) 
    elif relative_root == 'DVS_ActionRecog_dataset':
        train_set = DVSActionRecog(root = input_data,train = True, data_type = 'frame', frames_number = time_step,
                                    split_by = splitmeth, factor_tau = tau_factor,scale_factor = scale_factor) 
        test_set = DVSActionRecog(root = input_data,train = False, data_type = 'frame', frames_number = time_step,
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

def execute_experiment_v2(project_ref, name_experim, T = 16, splitby = 'number', batch_size = 8, 
                        epochs = 30,device = 'mps',lr = 0.1, inp_data= data_dir, 
                        net_name = 'DVSG_net',run_id = None, split_tr_tst = True,
                        factor_tau = 0.8 , scale_factor = 50, data_aug_prob = 0,
                        ):
    
    relative_root = os.path.basename(inp_data)
    #Carga de datos en función del dataset que se vaya a usar
    train_set,test_set, nclasses_, sizexy = loading_data(input_data = inp_data,time_step = T,
                                                         splitmeth = splitby,tr_tst_split = split_tr_tst,
                                                         tau_factor = factor_tau,scale_factor= scale_factor,
                                                         data_aug_prob = data_aug_prob)
    train_size_,test_size_ = len(train_set),len(test_set)
    #Arquitectura de red que se va a usar, modo multipaso 'm' por defecto
    net = load_net(net_name = net_name, n_classes = nclasses_, size_xy = sizexy)
    #Registro en wandb para la monitorización
    wandb.login()
    if run_id is not None:
        hyperparameters = None
        name_experim = None
        checkpoint_file = input('Checkpoint file(.kpath) with desired trained model:')
        resume_ = 'must'
    else:
        hyperparameters = dict(epochs=epochs,
            time_step = T,
            nclasses = nclasses_,
            train_size = train_size_,
            test_size = test_size_,
            batch_size=batch_size,
            learning_rate=lr,
            dataset=relative_root,
            DatAug_probability = data_aug_prob,
            architecture=net_name)
        if splitby == 'exp_decay':
            hyperparameters['tau factor'] = factor_tau
            hyperparameters['scale factor'] = scale_factor
        resume_ = None

    with wandb.init(project = project_ref, name = name_experim,
                    config = hyperparameters, id = run_id,resume = resume_): 
        if device == 'cuda':
            #Limpió la cache de la GPU
            torch.cuda.empty_cache()
        net.to(device)
        
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True
        )
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
            pin_memory=True
        )
        print('Tamaño de imágenes',sizexy,'\nNúmero de clases: ',nclasses_,'\nNº instancias train/test:', train_size_,'/', test_size_)
        #Optimizamos con SGD
        optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)     
        #El learning rate irá disminuyendo siguiendo un coseno según pasen las épocas. Luego vuelve a aumentar hasta llegar al valor inicial siguiendo este mismo coseno
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        max_test_acc = -1
        start_epoch = 0
        #Cargamos los datos si se quiere continuar ejecución
        if run_id is not None:
            dicts = torch.load(checkpoint_file)
            lr_scheduler.load_state_dict(dicts['lr_scheduler'])
            optimizer.load_state_dict(dicts['optimizer'])
            net.load_state_dict(dicts['net'])
            max_test_acc = dicts['max_test_acc']
            start_epoch = dicts['epoch'] + 1
            print('Trained model succesfully loaded')

        root_file = os.path.dirname(__file__)
        out_dir = os.path.join(root_file,'result_logs',relative_root, f'T{T}_b{batch_size}_lr{lr}_{name_experim}')

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f'Mkdir {out_dir}.')

        writer = SummaryWriter(out_dir, purge_step=start_epoch)

        for epoch in range(start_epoch, epochs):
            train_loss, train_acc = train_model(net=net, n_classes = nclasses_,tr_loader = train_data_loader,
                                                optimizer = optimizer,device = device, lr_scheduler = lr_scheduler,
                                                data_augmented= data_aug_prob!=0)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)

            test_loss,test_acc = test_model(net = net, n_classes = nclasses_,tst_loader = test_data_loader,
                                            optimizer = optimizer,device = device, lr_scheduler = lr_scheduler )

            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)
            # Registro de los valores en wandb
            wandb.log({
                'train_loss': train_loss, 'train_acc': train_acc,
                'test_loss': test_loss,'test_acc': test_acc
            }, step=epoch)

            save_max = False
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                wandb.run.summary['max_test_acc'] = max_test_acc
                save_max = True
                
            checkpoint = {
                'net': net.state_dict(),'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),'epoch': epoch,
                'max_test_acc': max_test_acc
            }

            if save_max:
                torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

            print(out_dir)
            print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')    


"""
#Código antiguo
def execute_experiment(T = 16,splitby = 'number',batch_size = 8, epochs = 30,
                        device = 'mps',lr = 0.1, inp_data= data_dir, 
                        net_name = 'DVSG_net',run_id = None ):
    
    relative_root = os.path.basename(inp_data)
    test_set = None
    #Carga de datos en función del dataset que se vaya a usar
    if relative_root == 'DVS_Gesture_dataset':
        train_set = DVS128Gesture(root = inp_data, train = True, data_type = 'frame', frames_number = T, split_by = splitby)
        test_set = DVS128Gesture(root = inp_data, train = False, data_type = 'frame', frames_number = T, split_by = splitby)
    elif relative_root == 'DVS_Animals_Dataset':
        data_set = DVSAnimals(root = inp_data, train = True, data_type = 'frame', frames_number = T, split_by = splitby) 
    elif relative_root == 'DVS_DailyAction_dataset':
        data_set = DVSDailyActions(root = inp_data,train = True, data_type = 'frame', frames_number = T, split_by = splitby) 
    elif relative_root == 'DVS_ActionRecog_dataset':
        train_set = DVSActionRecog(root = inp_data,train = True, data_type = 'frame', frames_number = T, split_by = splitby) 
        test_set = DVSActionRecog(root = inp_data,train = False, data_type = 'frame', frames_number = T, split_by = splitby) 
    else:
        raise ValueError('Unknown dataset. Could check name of the folder.')
  

    #In case the dataset has not been split yet: (Also obtain number of classes)
    if test_set is None:
        nclasses_ = len(data_set.classes)
        sizexy = data_set.get_H_W()
        labels = [sample[1] for sample in data_set]
        train_set, test_set = train_test_split(data_set, test_size = 0.2,stratify = np.array(labels), random_state = seed)
    else:
        nclasses_ = len(train_set.classes) 
        sizexy = train_set.get_H_W()

    train_size_ = len(train_set)
    test_size_ = len(test_set)

    #Arquitectura de red que se va a usar
    if net_name == 'DVSG_net':
        net = myDVSGestureNet(channels=128, output_size = nclasses_,input_sizexy= sizexy, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    elif net_name == 'resnet18':
        net = mysew_resnet18(spiking_neuron=neuron.LIFNode,num_classes = nclasses_, surrogate_function=surrogate.ATan(), detach_reset=True,cnf='ADD',zero_init_residual=True)
    else:
        raise ValueError('Unknown arquitecture. Could check posible names. Names gift: ',net_name)
    
    #Establecemos las neuronas en modo multipaso
    functional.set_step_mode(net, 'm') 

    #Registro en wandb para la monitorización
    wandb.login()
    if run_id is not None:
        hyperparameters = None
        name_experim = None
        checkpoint_file = input('Checkpoint file(.kpath) with desired trained model:')
        resume_ = 'must'
    else:
        hyperparameters = dict(epochs=epochs,
            time_step = T,
            nclasses = nclasses_,
            train_size = train_size_,
            test_size = test_size_,
            batch_size=batch_size,
            learning_rate=lr,
            dataset=relative_root,
            architecture=net_name)
        resume_ = None

    project_ref = input('Introducir el proyecto(fijarse en los que ya hay en wandb):')
    name_experim = input('Introducir el nombre concreto de esta ejecución:')

    with wandb.init(project = project_ref, name = name_experim,
                    config = hyperparameters, id = run_id,resume = resume_): 
        if device == 'cuda':
            #Limpió la cache de la GPU
            torch.cuda.empty_cache()

        net.to(device)
        
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True
        )

        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
            pin_memory=True
        )
        print('Tamaño de imágenes',sizexy)
        print('Número de clases: ',nclasses_)
        print('Nº instancias train/test:', train_size_,'/', test_size_)
        #Optimizamos con SGD
        optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
        
        #El learning rate irá disminuyendo siguiendo un coseno según pasen las épocas. Luego vuelve a aumentar hasta llegar al valor inicial siguiendo este mismo coseno
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        max_test_acc = -1
        start_epoch = 0

        if run_id is not None:
            dicts = torch.load(checkpoint_file)

            lr_scheduler.load_state_dict(dicts['lr_scheduler'])
            optimizer.load_state_dict(dicts['optimizer'])
            net.load_state_dict(dicts['net'])
            max_test_acc = dicts['max_test_acc']
            start_epoch = dicts['epoch'] + 1
            print('Trained model succesfully loaded')

        root_file = os.path.dirname(__file__)
        out_dir = os.path.join(root_file,'result_logs',relative_root, f'T{T}_b{batch_size}_lr{lr}_{name_experim}')

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f'Mkdir {out_dir}.')

        writer = SummaryWriter(out_dir, purge_step=start_epoch)
        with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
            args_txt.write('# epochs %d, batch_size %d'%(epochs,batch_size))
            args_txt.write('\n')
            args_txt.write(' '.join(sys.argv))

        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            net.train()
            train_loss = 0
            train_acc = 0
            train_samples = 0
            for frame, label in train_data_loader: #DUDA ¿Todas las instancias que hay en el batch corresponden a la misma etiqueta?
                optimizer.zero_grad()
                frame = frame.to(device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]  IMPORTANT, COULDN'T BE NECESSARY FOR SOME DATASETS
                label = label.to(device)
                label_onehot = F.one_hot(label, nclasses_).float()

                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

                train_samples += label.numel()
                train_loss += loss.item() * label.numel()
                train_acc += (out_fr.argmax(1) == label).float().sum().item()

                functional.reset_net(net)           #DUDA ¿Por qué no se resetean los pesos después de cada instancia en lugar después de haber pasado todo el batch?

            train_time = time.time()
            train_speed = train_samples / (train_time - start_time)
            train_loss /= train_samples
            train_acc /= train_samples


            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            lr_scheduler.step()

            net.eval()
            test_loss = 0
            test_acc = 0
            test_samples = 0
            with torch.no_grad():
                for frame, label in test_data_loader:
                    frame = frame.to(device)
                    frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                    label = label.to(device)
                    label_onehot = F.one_hot(label, nclasses_).float()
                    out_fr = net(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                    test_samples += label.numel()
                    test_loss += loss.item() * label.numel()
                    test_acc += (out_fr.argmax(1) == label).float().sum().item()
                    functional.reset_net(net)
            test_time = time.time()
            test_speed = test_samples / (test_time - train_time)
            test_loss /= test_samples
            test_acc /= test_samples
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)
            # Registro de los valores en wandb
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc
            }, step=epoch)

            save_max = False
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                save_max = True
                
            checkpoint = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'max_test_acc': max_test_acc
            }

            if save_max:
                torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

            print(out_dir)
            print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
            print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
            print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
"""