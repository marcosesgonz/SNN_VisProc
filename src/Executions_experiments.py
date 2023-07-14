import os

src_root = os.path.dirname(__file__)
data_root = os.path.join(os.path.dirname(src_root),'data')
data_animals = os.path.join(data_root,'DVS_Animals_Dataset')
data_gesture = os.path.join(data_root,'DVS_Gesture_dataset')
data_dailyactions = os.path.join(data_root,'DVS_DailyAction_dataset')
data_actrec = os.path.join(data_root,'DVS_ActionRecog_dataset')

from Laboratory import  execute_experiment_kfold,execute_experiment_TrTstSplit

if __name__ == '__main__':
    execute_experiment_TrTstSplit(project_ref = 'Experimento1', name_experim = 'DVSActRec_b2_bynumber_T16', T = 16, splitby = 'number', batch_size = 2, epochs = 65, 
                          lr = 0.1, inp_data= data_actrec)
    execute_experiment_TrTstSplit(project_ref = 'Experimento1', name_experim = 'DVSActRec_b2_byexpdecay_T26', T = 26, splitby = 'exp_decay', batch_size = 2, epochs = 65, 
                          lr = 0.1, inp_data= data_actrec)
    #execute_experiment_kfold(project_ref = 'FinalResults', name_experim = 'DVSActRec_b2_byyexpdecay_T26',  kfolds = 5, T = 8, splitby = 'number', batch_size = 1, epochs = 65, inp_data = data_actrec)
    #execute_experiment_kfold(project_ref = 'FinalResults', name_experim = 'DVSAnimals_b2_bynumber_T16',  kfolds = 5, T = 26, splitby = 'exp_decay', batch_size = 2, epochs = 65, inp_data = data_animals)
    #execute_experiment_kfold(project_ref = 'FinalResults', name_experim = 'DVSAnimals_b2_byyexpdecay_T26',  kfolds = 5, T = 26, splitby = 'exp_decay', batch_size = 2, epochs = 65, inp_data = data_animals)
    #execute_experiment_kfold(project_ref = 'FinalResults', name_experim = 'DVSDailyAct_b2_byyexpdecay_T26', kfolds = 5, T = 26, splitby = 'exp_decay', batch_size = 2, epochs = 65, inp_data = data_dailyactions)
