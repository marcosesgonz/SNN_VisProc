import os

src_root = os.path.dirname(__file__)
data_root = os.path.join(os.path.dirname(src_root),'data')
data_animals = os.path.join(data_root,'DVS_Animals_Dataset')
data_gesture = os.path.join(data_root,'DVS_Gesture_dataset')
data_dailyactions = os.path.join(data_root,'DVS_DailyAction_dataset')
data_actrec = os.path.join(data_root,'DVS_ActionRecog_dataset')
data_MAD = os.path.join(data_root,'MAD_dataset')

from Laboratory import  execute_experiment_kfold,execute_experiment_TrTstSplit
nepochs = 85
if __name__ == '__main__':
    execute_experiment_kfold(project_ref = 'Experiments_MAD', name_experim = 'MAD_b2_video_T22_RANN', T = 22, batch_size = 1, epochs = nepochs,data_type='video',net_name='DVSG_RANN',softm=False,
                          lr = 0.1, inp_data = data_MAD, gpu=True)
    execute_experiment_TrTstSplit(project_ref = 'Experiments_MAD', name_experim = 'MAD_b1_byexpdecay_T22', T = 22, batch_size = 1, epochs = nepochs,data_type='frame',softm=False,splitby='exp_decay',
                          lr = 0.1, inp_data = data_MAD, gpu=True)
    #execute_experiment_TrTstSplit(project_ref = 'Experiments_MAD', name_experim = 'MAD_b2_video_T22_RANN_prueba', T = 22, batch_size = 2, epochs = nepochs,data_type='video',net_name='DVSG_RANN',softm=False,
    #                      lr = 0.1, inp_data = data_MAD, gpu=True)
    #execute_experiment_kfold(project_ref = 'ExperimentKFolds', name_experim = 'DVSActRec_b2_byexpdecay_T22', T = 22, splitby = 'exp_decay', 
    #                         batch_size = 2, epochs = nepochs, inp_data = data_actrec,nworkers = 8)
    #execute_experiment_kfold(project_ref = 'FinalResults', name_experim = 'DVSAnimals_b2_bynumber_T16',  kfolds = 5, T = 26, splitby = 'exp_decay', batch_size = 2, epochs = 65, inp_data = data_animals)
    #execute_experiment_kfold(project_ref = 'FinalResults', name_experim = 'DVSAnimals_b2_byyexpdecay_T26',  kfolds = 5, T = 26, splitby = 'exp_decay', batch_size = 2, epochs = 65, inp_data = data_animals)
    #execute_experiment_kfold(project_ref = 'FinalResults', name_experim = 'DVSDailyAct_b2_byyexpdecay_T26', kfolds = 5, T = 26, splitby = 'exp_decay', batch_size = 2, epochs = 65, inp_data = data_dailyactions)
