import torch
import sys
import torch.nn.functional as F
import wandb
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.tensorboard import SummaryWriter
import time
import os
import datetime

#Instanciamos la red
net = parametric_lif_net.DVSGestureNet(channels=128, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
functional.set_step_mode(net, 'm')
#print(net)
#Dispositivo a usar para entrenar
device = 'mps'
#Localización del dataset de DVSGesture
data_dir = '/Users/marcosesquivelgonzalez/Desktop/MasterCDatos/TFM/data/DVS_Gesture_dataset'
#Time-step
T = 16
#Tamaño de los lotes
batch_size = 8 # equivalente a args.b
#Número de épocas
start_epoch, epochs = 0, 30
#learning rate
lr = 0.1

net.to(device)

train_set = DVS128Gesture(root=data_dir, train=True, data_type='frame', frames_number=T, split_by='number')
test_set = DVS128Gesture(root=data_dir, train=False, data_type='frame', frames_number=T, split_by='number')



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

print('Instancias de train: ',len(train_set)) #Efectivamente son 1176 instancias
print('Instancias de test: ',len(test_set)) 

#Optimizamos con SGD

optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum = 0.9)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

max_test_acc = -1

proj_dir = '/Users/marcosesquivelgonzalez/Desktop/MasterCDatos/TFM/Projects'
out_dir = os.path.join(proj_dir, f'T{T}_b{batch_size}_lr{lr}')
print(out_dir)

"""
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print(f'Mkdir {out_dir}.')

writer = SummaryWriter(out_dir, purge_step=start_epoch)
with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
    args_txt.write('Experimento 1: DVS_gesture, epochs %d, batch_size %d,')
    args_txt.write('\n')
    args_txt.write(' '.join(sys.argv))

for epoch in range(start_epoch, epochs):
    start_time = time.time()
    net.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    for frame, label in train_data_loader:
        optimizer.zero_grad()
        frame = frame.to(device)
        frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
        label = label.to(device)
        label_onehot = F.one_hot(label, 11).float()

        out_fr = net(frame).mean(0)
        loss = F.mse_loss(out_fr, label_onehot)
        loss.backward()
        optimizer.step()

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        functional.reset_net(net)

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
            label_onehot = F.one_hot(label, 11).float()
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