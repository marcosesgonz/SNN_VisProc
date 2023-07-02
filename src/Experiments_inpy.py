import Laboratory
import os

src_root = os.path.dirname(__file__)
data_root = os.path.join(os.path.dirname(src_root),'data')
data_animals = os.path.join(data_root,'SLAnimals_Dataset')
data_gesture = os.path.join(data_root,'DVS_Gesture_dataset')
data_dailyactions = os.path.join(data_root,'DVS_DailyAction_dataset')
data_actrec = os.path.join(data_root,'DVS_ActionRecog_dataset')

if __name__ == '__main__':

    dataset = input('Dataset to use(Animals/Gesture/DailyAct/ActRec): ')
    if dataset == 'Animals':
        dataset = data_animals
    if dataset == 'Gesture':
        dataset = data_gesture
    if dataset =='DailyAct':
        dataset = data_dailyactions
    if dataset == 'ActRec':
        dataset = data_actrec
    
    epochs_ = input('Número de épocas(default = 30): ')
    if len(epochs_) == 0:
        epochs_ = 8
    else:
        epochs_ = int(epochs_)
    
    batch_size_ = input('Tamaño de lote(default 8): ')
    if len(batch_size_) == 0:
        batch_size_ = 8
    else:
        batch_size_ = int(batch_size_)

    learning_rate_ = input('Set learning rate(default = 0.1): ')
    if len(learning_rate_) == 0:
        learning_rate_ = 0.1
    else:
        learning_rate_ = float(learning_rate_)
    
    device_ = input('Device(cuda/mps): ')

    Laboratory.execute_experiment(inp_data = dataset,
                                    epochs = epochs_,
                                    batch_size = batch_size_,
                                    lr = learning_rate_,
                                    device = device_)