import torch
import wandb
import random
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from sklearn.model_selection import StratifiedKFold,LeaveOneGroupOut
import numpy as np
import os
from utils import load_net,loading_data,test_model,train_model,reset_weights, num_trainable_params, convert_to_3d_eframes
from sklearn.metrics import confusion_matrix
#set the seed for reproducibility
seed = 310
try:
    import cupy as cp
except:
    cp = None

def set_seed():
    if cp != None:
        cp.random.seed(seed)  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.mps.deterministic = True
    torch.backends.cuda.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

data_dir = '/Users/marcosesquivelgonzalez/Desktop/MasterCDatos/TFM/data/DVS_Gesture_dataset'

#Data:80% train and 20% test
def execute_experiment_TrTstSplit(project_ref, name_experim, T = 16, splitby = 'number', batch_size = 8, data_type = 'frame', convert_to_3dframes = False, avg_before_fc_resnets = True,
                        epochs = 30,gpu = True,lr = 0.1, ler_rate_type = 'CosAnn',inp_data= data_dir, neuron_type = 'LIF',noutp_per_class = 10, nneurons_linear_layer = 512,
                        net_name = 'DVSG_net',run_id = None, split_tr_tst = True, softm = False, channels = 128, drop_out2d = None, resnet_pretrained = False, fine_tuning = False,
                        factor_tau = 0.8 , scale_factor = 50, data_aug_prob = 0, mix_strategy = 'num_events'
                        ):
    set_seed()
    device = ("cuda" if (torch.cuda.is_available() and gpu) else 'mps' if gpu else 'cpu')
    
    transform_ = convert_to_3d_eframes if convert_to_3dframes else None
    inchannels = 3 if convert_to_3dframes else 2
    print(f'In channels: {inchannels}')
    
    relative_root = os.path.basename(inp_data)
    #Carga de datos en función del dataset que se vaya a usar
    train_set,test_set, nclasses_, sizexy = loading_data(input_data = inp_data,time_step = T, datatype = data_type,
                                                         splitmeth = splitby, tr_tst_split = split_tr_tst,
                                                         tau_factor = factor_tau, scale_factor = scale_factor,
                                                         data_aug_prob = data_aug_prob,mix_strategy=mix_strategy, transform = transform_)
    train_size_,test_size_ = len(train_set),len(test_set)
    #Arquitectura de red que se va a usar, modo multipaso 'm' por defecto
    cupy = True if device == 'cuda' else False
    SNNmodel = not net_name.endswith('ANN')
    net = load_net(net_name = net_name, n_classes = nclasses_, size_xy = sizexy, noutp_per_class = noutp_per_class, nneurons_linear_layer = nneurons_linear_layer,in_channels=inchannels,
                    neuron_type = neuron_type, channels = channels, drop_out2d = drop_out2d, resnet_pretrained = resnet_pretrained, fine_tuning = fine_tuning, avg_before_fc_resnets = avg_before_fc_resnets,
                    cupy = cupy, softm = softm, num_frames = T)
    n_params = num_trainable_params(net)
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
            device = device,
            DatAug_probability = data_aug_prob,
            neuron_type = neuron_type,
            architecture=net_name,
            n_trainable_params = '%.6e'%n_params)
        if splitby == 'exp_decay':
            hyperparameters['tau factor'] = factor_tau
            hyperparameters['scale factor'] = scale_factor
        if net_name == 'DVSG_net':
            hyperparameters['nneurons_linear_layer'] = nneurons_linear_layer
            hyperparameters['noutp_per_class'] = noutp_per_class
        resume_ = None

    with wandb.init(project = project_ref, name = name_experim,
                    config = hyperparameters, id = run_id,resume = resume_): 
        if device == 'cuda':
            #Limpió la cache de la GPU
            torch.cuda.empty_cache()
        print(net)
        net.to(device)
        
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers = 2,
            pin_memory=True
        )
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers = 2,
            pin_memory=True
        )
        print('Using %s as device'% device)
        print('Applying 2d to 3d conversion to frames.') if convert_to_3dframes else None 
        print('SNN model: ',SNNmodel)
        print('Number of trainable paramenters:  %.6e'%n_params)
        print('Tamaño de imágenes',sizexy,'\nNúmero de clases: ',nclasses_,'\nNº instancias train/test:', train_size_,'/', test_size_)

        #Optimizamos con SGD
        if resnet_pretrained and fine_tuning:
            print('Doing fine tuning...') 
            #We only give to the optimizer the fc output layer to do fine tuning(the rest of parameters are frozen)
            optimizer = torch.optim.SGD(net.fc.parameters(), lr = lr, momentum = 0.9)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9) 

        #El learning rate irá disminuyendo siguiendo un coseno según pasen las épocas. Luego vuelve a aumentar hasta llegar al valor inicial siguiendo este mismo coseno
        if ler_rate_type == 'CosAnn':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        elif ler_rate_type == 'Step':
            assert epochs/3 == int(epochs/3)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size = epochs/3,gamma = 0.1)
        elif ler_rate_type == 'Const':
            lr_scheduler = None
        else: 
            raise NotImplemented("The possible learning rate types are 'Cosine Annealing' and 'Constant'")
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
            print(f'Logs saved in {out_dir}.')

        writer = SummaryWriter(out_dir, purge_step=start_epoch)

        for epoch in range(start_epoch, epochs):
            train_loss, train_acc = train_model(net=net, n_classes = nclasses_,tr_loader = train_data_loader,
                                                optimizer = optimizer,device = device, lr_scheduler = lr_scheduler, SNNmodel = SNNmodel,
                                                data_augmented= data_aug_prob!=0)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)

            test_loss,test_acc = test_model(net = net, n_classes = nclasses_,tst_loader = test_data_loader,SNNmodel = SNNmodel,
                                            device = device)

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
                'epoch': epoch, 'max_test_acc': max_test_acc
            }
            if lr_scheduler is not None:
                checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

            if save_max:
                torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

            #print(out_dir)
            print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')    

"""project_ref, name_experim, T = 16, splitby = 'number', batch_size = 8, data_type = 'frame',
                        epochs = 30,gpu = True,lr = 0.1, inp_data= data_dir, neuron_type = 'LIF',
                        net_name = 'DVSG_net',run_id = None, split_tr_tst = True,
                        factor_tau = 0.8 , scale_factor = 50, data_aug_prob = 0,
                        """
def execute_experiment_kfold(project_ref, name_experim, T = 16, splitby = 'number', batch_size = 8, data_type='frame',
                        epochs = 65, gpu = True,lr = 0.1, ler_rate_type = 'CosAnn', inp_data = data_dir, neuron_type = 'LIF', net_name = 'DVSG_net',
                         noutp_per_class = 10, nneurons_linear_layer = 512, resnet_pretrained = False, fine_tuning = False,
                        run_id = None, kfolds = 5, factor_tau = 0.8 , scale_factor = 50, 
                        data_aug_prob = 0, nworkers = 2, pinmemory = True, softm = False, mix_strategy = 'num_events'
                        ):
    set_seed()
    device = ("cuda" if (torch.cuda.is_available() and gpu) else 'mps' if gpu else 'cpu')
    if device == 'cuda':
        #Limpió la cache de la GPU
        torch.cuda.empty_cache()
    print('Using %s as device'% device)

    transform_ = convert_to_3d_eframes if net_name == 'resnet18' and data_type == 'frame' else None
    relative_root = os.path.basename(inp_data)
    #Carga de datos en función del dataset que se vaya a usar
    data_set, nclasses_, sizexy = loading_data(input_data = inp_data,time_step = T, datatype = data_type, splitmeth = splitby,tr_tst_split = False,
                                                         tau_factor = factor_tau,scale_factor= scale_factor, data_aug_prob = data_aug_prob, mix_strategy = mix_strategy,transform = transform_)
    data_size = len(data_set)
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
            nfolds = kfolds,
            data_size = data_size,
            batch_size=batch_size,
            learning_rate=lr,
            dataset=relative_root,
            device = device,
            DatAug_probability = data_aug_prob,
            neuron_type = neuron_type,
            architecture=net_name)
        if splitby == 'exp_decay':
            hyperparameters['tau factor'] = factor_tau
            hyperparameters['scale factor'] = scale_factor
        if net_name == 'DVSG_net':
            hyperparameters['nneurons_linear_layer'] = nneurons_linear_layer
            hyperparameters['noutp_per_class'] = noutp_per_class
        resume_ = None

    with wandb.init(project = project_ref, name = name_experim,
                    config = hyperparameters, id = run_id,resume = resume_): 
        
        print('Tamaño de imágenes',sizexy,'\nNúmero de clases: ',nclasses_,'\nNº instancias(promedio) train/test:', data_size*(kfolds-1)/kfolds,'/', data_size/kfolds)
        
        root_file = os.path.dirname(__file__)
        out_dir = os.path.join(root_file,'result_logs',relative_root, f'T{T}_b{batch_size}_lr{lr}_{name_experim}_kf{kfolds}')

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f'Mkdir {out_dir}.')

        # For fold results
        results = {}
        for str in ['val_loss','val_acc','train_loss','train_acc']:
            results[str] = []
        #Cupy backend if possible
        cupy = True if device == 'cuda' else False
        #Arquitectura de red que se va a usar, modo multipaso 'm' por defecto
        net = load_net(net_name = net_name, n_classes = nclasses_, size_xy = sizexy, noutp_per_class = noutp_per_class, nneurons_linear_layer = nneurons_linear_layer,
                        neuron_type = neuron_type, resnet_pretrained = resnet_pretrained, fine_tuning = fine_tuning, cupy = cupy, softm = softm, num_frames = T)
        n_params = num_trainable_params(net)
        wandb.config.update({'n_trainable_params' : '%.6e' %n_params})
        net.to(device)
        SNNmodel = not net_name.endswith('ANN')
        print('SNN model: ',SNNmodel)

        if relative_root == 'MAD_dataset':
            print('Using leave_one_out strategy')
            #LeaveOneOut strategy for MAD dataset. One fold per subject(5 subjects)
            logo = LeaveOneGroupOut()
            folds_iterator = logo.split(X = data_set, groups = data_set.subjects)
        else:
            #Cross validation of 5 folds
            folds_creator = StratifiedKFold(n_splits = kfolds,shuffle=True,random_state=seed)
            folds_iterator = folds_creator.split(data_set, y = [sample[1] for sample in data_set])

        for nkfold,(train_idx,test_idx) in enumerate(folds_iterator):
            print('----------Fold {}-------------'.format(nkfold))
            #Reseteo los pesos de la red
            print('Reseting wegihts..')
            net.apply(reset_weights)
            #Reseteo el accuracy máximo en test
            max_val_acc = 0

            #Optimizamos con SGD
            if resnet_pretrained and fine_tuning:
                print('Doing fine tuning...') 
                #We only give to the optimizer the fc output layer to do fine tuning(the rest of parameters are frozen)
                optimizer = torch.optim.SGD(net.fc.parameters(), lr = lr, momentum = 0.9)
            else:
                optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9) 

            #El learning rate irá disminuyendo siguiendo un coseno según pasen las épocas. Luego vuelve a aumentar hasta llegar al valor inicial siguiendo este mismo coseno
            if ler_rate_type == 'CosAnn':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            elif ler_rate_type == 'Const':
                lr_scheduler = None
            else: 
                raise NotImplemented("The possible learning rate types are 'Cosine Annealing' and 'Constant'")

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

            train_data_loader = torch.utils.data.DataLoader(
                            dataset = data_set, batch_size = batch_size,
                            sampler = train_subsampler, num_workers = nworkers,
                            drop_last = True, pin_memory = pinmemory)
            test_data_loader = torch.utils.data.DataLoader(
                            dataset = data_set, batch_size = batch_size,
                            sampler = test_subsampler, num_workers = nworkers,
                            drop_last = False, pin_memory = pinmemory)
            
            for epoch in range(epochs):
                train_loss, train_acc = train_model(net=net, n_classes = nclasses_,tr_loader = train_data_loader,
                                                    optimizer = optimizer,device = device, lr_scheduler = lr_scheduler, SNNmodel = SNNmodel,
                                                    data_augmented = data_aug_prob!=0)
                val_loss,val_acc = test_model(net = net, n_classes = nclasses_,tst_loader = test_data_loader,
                                                    device = device, SNNmodel = SNNmodel)

                wandb.log({f'train_loss_k{nkfold}': train_loss, f'train_acc_k{nkfold}': train_acc, f'test_loss_k{nkfold}':val_loss, f'test_acc_k{nkfold}':val_acc}, step = epoch)

                save_max = False
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    val_acc_fold = max_val_acc
                    val_loss_fold = val_loss
                    train_acc_fold = train_acc
                    train_loss_fold = train_loss
                    wandb.run.summary[f'max_val_acc_k{nkfold}'] = max_val_acc
                    save_max = True

                print(f' epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={val_loss: .4f}, test_acc ={val_acc: .4f}, max_test_acc ={max_val_acc: .4f}') 
                checkpoint = {'net': net.state_dict(),'optimizer': optimizer.state_dict(),'lr_scheduler': lr_scheduler.state_dict(),'epoch': epoch}

                if save_max:
                    torch.save(checkpoint, os.path.join(out_dir, f'checkpoint_fold{nkfold}_max.pth')) 
                
                torch.save(checkpoint, os.path.join(out_dir, f'checkpoint_fold{nkfold}_latest.pth')) 
            

            results['val_acc'].append(val_acc_fold)
            results['val_loss'].append(val_loss_fold)
            results['train_acc'].append(train_acc_fold)
            results['train_loss'].append(train_loss_fold)

        #Registro de los valores en wandb
        wandb.run.summary['mean_val_acc'] = np.mean(results['val_acc'])
        wandb.run.summary['mean_val_loss'] = np.mean(results['val_loss'])
        wandb.run.summary['mean_train_acc'] = np.mean(results['train_acc'])
        wandb.run.summary['mean_train_loss'] = np.mean(results['train_loss'])


def infer_data(net_name,data,checkpoint_file,frames_num,noutp_per_class = 10, nneurons_linear_layer = 512, channels = 128, neuron_type = 'LIF',drop_out2d = None,
                resnet_pretrained = False, fine_tuning = False, softm = False, gpu = True, nworkers = 2, use_sublabels = True, batch_size_ = 8, return_acc = False):
    """
    'data' should be a class with the atribute 'classes' and 'samples'(if it is used sublabels) and the method 'get_H_W()'. 
    """
    set_seed()
    nclasses_ = len(data.classes)
    sizexy = data.get_H_W()

    if use_sublabels:
        sublabels = [os.path.splitext(os.path.basename(sample[0]).split('_')[1])[0] for sample in data.samples]

    val_data_loader = torch.utils.data.DataLoader(
                            dataset = data, batch_size = batch_size_,
                            num_workers = nworkers,shuffle = False, #Shuffle false it is important to preserve the order of data.samples in the dataloader
                            drop_last = False, pin_memory = True)
    
    device = ("cuda" if (torch.cuda.is_available() and gpu) else 'mps' if gpu else 'cpu')
    #Cupy backend if possible
    cupy = True if device == 'cuda' else False
    net = load_net(net_name = net_name, n_classes = nclasses_, size_xy = sizexy, noutp_per_class = noutp_per_class, nneurons_linear_layer = nneurons_linear_layer,
                    neuron_type = neuron_type, channels = channels, drop_out2d = drop_out2d, resnet_pretrained = resnet_pretrained, fine_tuning = fine_tuning,
                    cupy = cupy, softm = softm, num_frames = frames_num)
    SNNmodel = not net_name.endswith('ANN')
    dicts = torch.load(checkpoint_file, map_location = torch.device(device))
    net.load_state_dict(dicts['net'])
    net.to(device)
    net.eval()
    #test_loss = 0
    #test_acc = 0
    #test_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i,(frame, label) in enumerate(val_data_loader):
            frame = frame.to(device)
            if SNNmodel:
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to(device)
            #label_onehot = F.one_hot(label, nclasses_).float()
            out_fr = net(frame)
            #loss = F.mse_loss(out_fr, label_onehot)
            #test_samples += label.numel()
            #test_loss += loss.item() * label.numel()
            #test_acc += (out_fr.argmax(1) == label).float().sum().item()
            if SNNmodel:
                functional.reset_net(net)

            all_preds.append(out_fr.argmax(1).cpu().numpy())
            all_labels.append(label.cpu().numpy())


        #test_loss /= test_samples
        #test_acc /= test_samples

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        #confusion = confusion_matrix(all_labels, all_preds)
        if return_acc:
            return np.mean(all_labels==all_preds)
        elif use_sublabels:
            return all_labels,all_preds,sublabels
        else:
            return all_labels,all_preds
        #return  test_acc, confusion
        


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