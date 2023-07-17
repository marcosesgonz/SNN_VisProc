import subprocess

execut_path = '/Users/marcosesquivelgonzalez/Desktop/MasterCDatos/TFM/src/my_rpg_e2vid/run_reconstruction.py'
pretained_model_path = '/Users/marcosesquivelgonzalez/Desktop/MasterCDatos/TFM/src/my_rpg_e2vid/pretrained/E2VID_lightweight.pth.tar'
npz_event_file_path = '/Users/marcosesquivelgonzalez/Desktop/MasterCDatos/TFM/data/DVS_Animals_Dataset/events_np/train/bird/user00_indoor_0.npz'
outp_folder = '/Users/marcosesquivelgonzalez/Desktop/prueba'
args = ['-c',pretained_model_path,'-i',npz_event_file_path,'--output_folder',outp_folder,
        '--fixed_duration','--time_step','22']
subprocess.run(['python3',execut_path]+args)