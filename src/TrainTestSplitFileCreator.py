import os
import numpy 
import random

def create_traintest_split_file(root, seed = 40,end_files = '.aedat',train_size = 0.8):
    random.seed(seed)
    """
    Create two .txt files in 'root' folder with the root of the samples for train and for test:
        trials_to_train.txt
        trials_to_test.txt
    Parameters:
        root (str): Root with the samples. This root must include inside one folder per classs.
    """
    file_train = os.path.join(root,'trials_to_train.txt')
    file_test = os.path.join(root,'trials_to_test.txt')
    if os.path.exists(file_train) or os.path.exists(file_test):
        raise ('Files already exist. If they are wrong. Delete it')
    
    labels = [it.name for it in os.scandir(root) if it.is_dir()] 

    with open(file_train,'w') as f_train, open(file_test,'w') as f_test:
        for label in labels:
            aedat_dir = os.path.join(root,label)
            aedats_list = [file for file in os.listdir(aedat_dir) if file.endswith(end_files)]
            random.shuffle(aedats_list)
            n_samples = len(aedats_list)
            n_samples_train = int(train_size * n_samples)

            for i,fname in enumerate(aedats_list):
                    file = os.path.join(label, fname)
                    if i < n_samples_train:
                        f_train.write(file +'\n')
                    else:
                        f_test.write(file +'\n')


AR_extracted_data = '/Users/marcosesquivelgonzalez/Desktop/MasterCDatos/TFM/data/DVS_ActionRecog_dataset/extract/ActionRecognition'
create_traintest_split_file(AR_extracted_data)
