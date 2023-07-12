import os

src_root = os.path.dirname(__file__)
data_root = os.path.join(os.path.dirname(src_root),'data')
data_animals = os.path.join(data_root,'DVS_Animals_Dataset')
data_gesture = os.path.join(data_root,'DVS_Gesture_dataset')
data_dailyactions = os.path.join(data_root,'DVS_DailyAction_dataset')
data_actrec = os.path.join(data_root,'DVS_ActionRecog_dataset')

from Laboratory import execute_experiment_v2
time_step = 16

if __name__ == '__main__':
    execute_experiment_v2(project_ref = 'Experimento1', name_experim = 'DVSDailyAct_b2_byexpdecay_T26', T = 26, splitby = 'exp_decay', batch_size = 2, epochs = 65, device = 'cuda',
                          lr = 0.1, inp_data= data_dailyactions)
    execute_experiment_v2(project_ref = 'Experimento1', name_experim = 'DVSGesture_b2_byexpdecay_T26', T = 26, splitby = 'exp_decay', batch_size = 2, epochs = 65, device = 'cuda',
                          lr = 0.1, inp_data= data_gesture)