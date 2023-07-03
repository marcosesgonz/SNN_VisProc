import torch
import sys
import torch.nn.functional as F
import wandb
import random
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from Datasets import DVSAnimals, DVSDailyActions, DVSActionRecog
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from models import myDVSGestureNet
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


data_dir = '/Users/marcosesquivelgonzalez/Desktop/MasterCDatos/TFM/data/DVS_Gesture_dataset'


def execute_experiment(T = 16,splitby = 'number',batch_size = 8, epochs = 30, device = 'mps',lr = 0.1, inp_data= data_dir, net_name = 'DVSG_net'):
    
    start_epoch = 0
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
    else:
        raise ValueError('Unknown arquitecture. Could check posible names.')
    
    #Establecemos las neuronas en modo multipaso
    functional.set_step_mode(net, 'm') 


    #Registro en wandb para la monitorización
    wandb.login()
    hyperparameters = dict(epochs=epochs,
        time_step = T,
        nclasses = nclasses_,
        train_size = train_size_,
        test_size = test_size_,
        labels = train_set.class_to_idx,
        batch_size=batch_size,
        learning_rate=lr,
        dataset=relative_root,
        architecture=net_name,
        )
    
    project_ref = input('Introducir el proyecto(fijarse en los que ya hay en wandb):')
    name_experim = input('Introducir el nombre concreto de esta ejecución(CUIDADO con sobreescribir¿?):')

    with wandb.init(project = project_ref, name=name_experim,config=hyperparameters): 
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
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum = 0.9)
        
        #El learning rate irá disminuyendo siguiendo un coseno según pasen las épocas. Luego vuelve a aumentar hasta llegar al valor inicial siguiendo este mismo coseno
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        max_test_acc = -1
        root_file = os.path.dirname(__file__)
        out_dir = os.path.join(root_file,'result_logs',relative_root, f'T{T}_b{batch_size}_lr{lr}')


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
        print('Max test accuracy(not 100percent sure) = ',max_test_acc)


