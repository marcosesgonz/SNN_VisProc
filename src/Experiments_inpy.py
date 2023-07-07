import Laboratory
import os

src_root = os.path.dirname(__file__)
data_root = os.path.join(os.path.dirname(src_root),'data')
data_animals = os.path.join(data_root,'DVS_Animals_Dataset')
data_gesture = os.path.join(data_root,'DVS_Gesture_dataset')
data_dailyactions = os.path.join(data_root,'DVS_DailyAction_dataset')
data_actrec = os.path.join(data_root,'DVS_ActionRecog_dataset')

def inputdata(text,default):
    input_ = input(text)
    input_ = default if len(input_) == 0 else input_
    return input_

if __name__ == '__main__':

    runid_ = inputdata('Run_id para reanudar ejecucion(default = None): ', None)
    netname = inputdata('Nombre de la red (DVSG_net (default) / resnet18): ', 'DVSG_net')
    dataset = input('Dataset to use(Animals/Gesture/DailyAct/ActRec): ')
    if dataset == 'Animals':
        dataset = data_animals
    elif dataset == 'Gesture':
        dataset = data_gesture
    elif dataset =='DailyAct':
        dataset = data_dailyactions
    elif dataset == 'ActRec':
        dataset = data_actrec 
    timestep_ = int(inputdata('Time step (default = 16): ', 16))
    split_strat = inputdata('Split strategy (number(default) / time/ exp_decay ): ', 'number')
    tau_factor,scale_factor = None, None
    if split_strat == 'exp_decay':
        tau_factor = float(inputdata('Time step (default = 16): ', 0.8))
        scale_factor = int(inputdata('Time step (default = 16): ', 50))
    epochs_ = int(inputdata('Número de épocas (default = 50): ', 50)) 
    batch_size_ = int(inputdata('Tamaño de lote (default 8): ', 8))
    learning_rate_ = float(inputdata('Set learning rate (default = 0.1): ', 0.1))
    device_ = inputdata('Device (cuda(default) /mps): ','cuda')

    Laboratory.execute_experiment_v2(inp_data = dataset,
                                    T = timestep_,
                                    splitby= split_strat,
                                    epochs = epochs_,
                                    batch_size = batch_size_,
                                    lr = learning_rate_,
                                    device = device_,
                                    net_name=netname,
                                    run_id = runid_,
                                    factor_tau= tau_factor,
                                    scale_factor= scale_factor)
    