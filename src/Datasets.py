from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
from torchvision.datasets import utils
from abc import abstractmethod
from spikingjelly import datasets as sjds
from torchvision.datasets import DatasetFolder
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import event_integration_to_frame
from spikingjelly import configure
from spikingjelly.datasets import np_savez
import struct
import tonic
import random
import subprocess
import shutil
import json
import cv2
from typing import cast

def load_npz_video(file_name):
    return np.load(file_name,allow_pickle=True)['video'].astype(np.float32)


#Class of spikingjelly edited by me in only 3 lines to avoid problems.
class MyNeuromorphicDatasetFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            factor_tau: float = 1.5,
            scale_factor: int = 50
    ) -> None:
        '''
        :param root: root path of the dataset
        :type root: str
        :param train: whether use the train set. Set ``True`` or ``False`` for those datasets provide train/test
            division, e.g., DVS128 Gesture dataset. If the dataset does not provide train/test division, e.g., CIFAR10-DVS,
            please set ``None`` and use :class:`~split_to_train_test_set` function to get train/test set
        :type train: bool
        :param data_type: `event` or `frame`
        :type data_type: str
        :param frames_number: the integrated frame number
        :type frames_number: int
        :param split_by: `time` or `number` or 'exp_decay'
        :type split_by: str
        :param duration: the time duration of each frame
        :type duration: int
        :param custom_integrate_function: a user-defined function that inputs are ``events, H, W``.
            ``events`` is a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
            ``H`` is the height of the data and ``W`` is the weight of the data.
            For example, H=128 and W=128 for the DVS128 Gesture dataset.
            The user should define how to integrate events to frames, and return frames.
        :type custom_integrate_function: Callable
        :param custom_integrated_frames_dir_name: The name of directory for saving frames integrating by ``custom_integrate_function``.
            If ``custom_integrated_frames_dir_name`` is ``None``, it will be set to ``custom_integrate_function.__name__``
        :type custom_integrated_frames_dir_name: str or None
        :param transform: a function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        :type transform: callable
        :param target_transform: a function/transform that takes
            in the target and transforms it.
        :type target_transform: callable
        The base class for neuromorphic dataset. Users can define a new dataset by inheriting this class and implementing
        all abstract methods. Users can refer to :class:`spikingjelly.datasets.dvs128_gesture.DVS128Gesture`.
        If ``data_type == 'event'``
            the sample in this dataset is a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``.
        If ``data_type == 'frame'`` and ``frames_number`` is not ``None``
            events will be integrated to frames with fixed frames number. ``split_by`` will define how to split events.
            See :class:`cal_fixed_frames_number_segment_index` for
            more details.
        If ``data_type == 'frame'``, ``frames_number`` is ``None``, and ``duration`` is not ``None``
            events will be integrated to frames with fixed time duration.
        If ``data_type == 'frame'``, ``frames_number`` is ``None``, ``duration`` is ``None``, and ``custom_integrate_function`` is not ``None``:
            events will be integrated by the user-defined function and saved to the ``custom_integrated_frames_dir_name`` directory in ``root`` directory.
            Here is an example from SpikingJelly's tutorials:

            .. code-block:: python

                from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
                from typing import Dict
                import numpy as np
                import spikingjelly.datasets as sjds
                def integrate_events_to_2_frames_randomly(events: Dict, H: int, W: int):
                    index_split = np.random.randint(low=0, high=events['t'].__len__())
                    frames = np.zeros([2, 2, H, W])
                    t, x, y, p = (events[key] for key in ('t', 'x', 'y', 'p'))
                    frames[0] = sjds.integrate_events_segment_to_frame(x, y, p, H, W, 0, index_split)
                    frames[1] = sjds.integrate_events_segment_to_frame(x, y, p, H, W, index_split, events['t'].__len__())
                    return frames
                root_dir = 'D:/datasets/DVS128Gesture'
                train_set = DVS128Gesture(root_dir, train=True, data_type='frame', custom_integrate_function=integrate_events_to_2_frames_randomly)
                from spikingjelly.datasets import play_frame
                frame, label = train_set[500]
                play_frame(frame)
        '''

        events_np_root = os.path.join(root, 'events_np')

        if not os.path.exists(events_np_root):

            download_root = os.path.join(root, 'download')

            if os.path.exists(download_root):
                print(f'The [{download_root}] directory for saving downloaded files already exists, check files...')
                # check files
                resource_list = self.resource_url_md5()
                for i in range(resource_list.__len__()):
                    file_name, url, md5 = resource_list[i]
                    fpath = os.path.join(download_root, file_name)
                    if not utils.check_integrity(fpath=fpath, md5=md5):
                        print(f'The file [{fpath}] does not exist or is corrupted.')

                        if os.path.exists(fpath):
                            # If file is corrupted, we will remove it.
                            #os.remove(fpath)                                                                                           #EDITADO POR MI, LO HE COMENTADO
                            print(f'Warning: [{fpath}] exist but could be corrupted. Care.')            #print(f'Remove [{fpath}]')     #EDITADO POR MI, HE CAMBIADO LO COMENTADO POR ESTA LINEA

                        if self.downloadable():
                            # If file does not exist, we will download it.
                            print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                            utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
                        else:
                            if not os.path.exists(fpath):                                                                               #EDITADO POR MI, ESTE IF NO ESTABA
                                raise NotImplementedError(
                                    f'This dataset can not be downloaded by SpikingJelly, please download [{file_name}] from [{url}] manually and put files at {download_root}.')

            else:
                os.mkdir(download_root)
                print(f'Mkdir [{download_root}] to save downloaded files.')
                resource_list = self.resource_url_md5()
                if self.downloadable():
                    # download and extract file
                    for i in range(resource_list.__len__()):
                        file_name, url, md5 = resource_list[i]
                        print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                        utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
                else:
                    raise NotImplementedError(f'This dataset can not be downloaded by SpikingJelly, '
                                              f'please download files manually and put files at [{download_root}]. '
                                              f'The resources file_name, url, and md5 are: \n{resource_list}')

            # We have downloaded files and checked files. Now, let us extract the files
            extract_root = os.path.join(root, 'extract')
            if os.path.exists(extract_root):
                print(f'The directory [{extract_root}] for saving extracted files already exists.\n'
                      f'SpikingJelly will not check the data integrity of extracted files.\n'
                      f'If extracted files are not integrated, please delete [{extract_root}] manually, '
                      f'then SpikingJelly will re-extract files from [{download_root}].')
                # shutil.rmtree(extract_root)
                # print(f'Delete [{extract_root}].')
            else:
                os.mkdir(extract_root)
                print(f'Mkdir [{extract_root}].')
                self.extract_downloaded_files(download_root, extract_root)

            # Now let us convert the origin binary files to npz files
            os.mkdir(events_np_root)
            print(f'Mkdir [{events_np_root}].')
            print(f'Start to convert the origin data from [{extract_root}] to [{events_np_root}] in np.ndarray format.')
            self.create_events_np_files(extract_root, events_np_root)

        H, W = self.get_H_W()

        if data_type == 'event':
            _root = events_np_root
            _loader = np.load
            _transform = transform
            _target_transform = target_transform

        elif data_type == 'frame':
            if frames_number is not None:
                assert frames_number > 0 and isinstance(frames_number, int)
                assert split_by in ('time', 'number' ,'exp_decay')
                if split_by == 'exp_decay':
                    fact_tau = str(factor_tau).replace('.','_')
                    frames_np_root = os.path.join(root, f'frames_num{frames_number}_splitby_{split_by}_tau{fact_tau}_scale{scale_factor}')
                else:
                    frames_np_root = os.path.join(root, f'frames_number_{frames_number}_split_by_{split_by}')
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')

                    # create the same directory structure
                    sjds.create_same_directory_structure(events_np_root, frames_np_root)

                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        for e_root, e_dirs, e_files in os.walk(events_np_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    if e_file.endswith('.npz'):  
                                        events_np_file = os.path.join(e_root, e_file)
                                        print(f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                        tpe.submit(event_integration_to_frame.integrate_events_to_frame_wfixed_frames_num,
                                                    self.load_events_np, events_np_file, output_dir, split_by, frames_number, H, W,print_save= True,
                                                    factor_tau = factor_tau, scale_factor = scale_factor)
                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = sjds.load_npz_frames
                _transform = transform
                _target_transform = target_transform

            elif duration is not None:
                assert duration > 0 and isinstance(duration, int)
                frames_np_root = os.path.join(root, f'duration_{duration}')
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')

                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')
                    # create the same directory structure
                    sjds.create_same_directory_structure(events_np_root, frames_np_root)
                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        for e_root, e_dirs, e_files in os.walk(events_np_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    if e_file.endswith('.npz'):  
                                        events_np_file = os.path.join(e_root, e_file)
                                        print(f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                        tpe.submit(sjds.integrate_events_file_to_frames_file_by_fixed_duration, self.load_events_np, events_np_file, output_dir, duration, H, W, True)

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = sjds.load_npz_frames
                _transform = transform
                _target_transform = target_transform

            elif custom_integrate_function is not None:
                if custom_integrated_frames_dir_name is None:
                    custom_integrated_frames_dir_name = custom_integrate_function.__name__

                frames_np_root = os.path.join(root, custom_integrated_frames_dir_name)
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')
                    # create the same directory structure
                    sjds.create_same_directory_structure(events_np_root, frames_np_root)
                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        for e_root, e_dirs, e_files in os.walk(events_np_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    if e_file.endswith('.npz'):  
                                        events_np_file = os.path.join(e_root, e_file)
                                        print( f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                        tpe.submit(sjds.save_frames_to_npz_and_print, os.path.join(output_dir, os.path.basename(events_np_file)), custom_integrate_function(np.load(events_np_file), H, W))

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = sjds.load_npz_frames
                _transform = transform
                _target_transform = target_transform


            else:
                raise ValueError('At least one of "frames_number", "duration" and "custom_integrate_function" should not be None.')
            
        elif data_type == 'video':
            if frames_number is not None:
                assert frames_number > 0 and isinstance(frames_number, int)
                vid_np_root = os.path.join(root, f'video_number_{frames_number}')
                if os.path.exists(vid_np_root):
                    print(f'The directory [{vid_np_root}] already exists.')
                else:
                    os.mkdir(vid_np_root)
                    print(f'Mkdir [{vid_np_root}].')

                    # create the same directory structure
                    sjds.create_same_directory_structure(events_np_root, vid_np_root)
                    
                    src_root = os.path.join( os.path.dirname(os.path.dirname(root)) , 'src')
                    executor_path = os.path.join(src_root,'my_rpg_e2vid','run_reconstruction.py')
                    pretrained_model_path = os.path.join(src_root,'my_rpg_e2vid','pretrained','E2VID_lightweight.pth.tar')

                    t_ckp = time.time()
                    for e_root, e_dirs, e_files in os.walk(events_np_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(vid_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    if e_file.endswith('.npz'):                 
                                        events_np_file = os.path.join(e_root, e_file)
                                        print(f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                        args = ['-c', pretrained_model_path,'-i', events_np_file, '--output_folder', output_dir, 
                                                        '--fixed_duration','--time_step', str(frames_number)]
                                        subprocess.run(['python3',executor_path]+args)
                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

            _root = vid_np_root
            _loader = load_npz_video
            _transform = transform
            _target_transform = target_transform
            
        if train is not None:
            if train:
                _root = os.path.join(_root, 'train')
            else:
                _root = os.path.join(_root, 'test')
        else:
            _root = self.set_root_when_train_is_none(_root)

        super().__init__(root=_root, loader=_loader, extensions=('.npz', ), transform=_transform, #Habría que añadir en extensions .npy si se quiere trabajar directamente con eventos
                         target_transform=_target_transform)

    def set_root_when_train_is_none(self, _root: str):
        return _root


    @staticmethod
    @abstractmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        pass

    @staticmethod
    @abstractmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        pass

    @staticmethod
    @abstractmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None
        This function defines how to extract download files.
        '''
        pass

    @staticmethod
    @abstractmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None
        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        pass

    @staticmethod
    @abstractmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W`` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        pass

    @staticmethod
    def load_events_np(fname: str):
        '''
        :param fname: file name
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        This function defines how to load a sample from `events_np`. In most cases, this function is `np.load`.
        But for some datasets, e.g., ES-ImageNet, it can be different.
        '''
        return np.load(fname)


class DVS128Gesture(MyNeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            factor_tau: float = 0.8,
            scale_factor: int = 50
    ) -> None:
        assert train is not None
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform,factor_tau,scale_factor)
    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        url = 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794'
        return [
            ('DvsGesture.tar.gz', url, '8a5c71fb11e24e5ca5b11866ca6c00a1'),
            ('gesture_mapping.csv', url, '109b2ae64a0e1f3ef535b18ad7367fd1'),
            ('LICENSE.txt', url, '065e10099753156f18f51941e6e44b66'),
            ('README.txt', url, 'a0663d3b1d8307c329a43d949ee32d19')
        ]

    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return False

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None

        This function defines how to extract download files.
        '''
        fpath = os.path.join(download_root, 'DvsGesture.tar.gz')
        print(f'Extract [{fpath}] to [{extract_root}].')
        extract_archive(fpath, extract_root)


    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        '''
        :param file_name: path of the events file
        :type file_name: str
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        :rtype: Dict

        This function defines how to read the origin binary data.
        '''
        return sjds.load_aedat_v3(file_name)

    @staticmethod
    def split_aedat_files_to_np(fname: str, aedat_file: str, csv_file: str, output_dir: str):
        events = DVS128Gesture.load_origin_data(aedat_file)
        print(f'Start to split [{aedat_file}] to samples.')
        # read csv file and get time stamp and label of each sample
        # then split the origin data to samples
        csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',', skiprows=1)

        # Note that there are some files that many samples have the same label, e.g., user26_fluorescent_labels.csv
        label_file_num = [0] * 11

        # There are some wrong time stamp in this dataset, e.g., in user22_led_labels.csv, ``endTime_usec`` of the class 9 is
        # larger than ``startTime_usec`` of the class 10. So, the following codes, which are used in old version of SpikingJelly,
        # are replaced by new codes.

        for i in range(csv_data.shape[0]):
            # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
            label = csv_data[i][0] - 1
            t_start = csv_data[i][1]
            t_end = csv_data[i][2]
            mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
            file_name = os.path.join(output_dir, str(label), f'{fname}_{label_file_num[label]}.npz')
            np_savez(file_name,
                     t=events['t'][mask],
                     x=events['x'][mask],
                     y=events['y'][mask],
                     p=events['p'][mask]
                     )
            print(f'[{file_name}] saved.')
            label_file_num[label] += 1


    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None

        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        aedat_dir = os.path.join(extract_root, 'DvsGesture')
        train_dir = os.path.join(events_np_root, 'train')
        test_dir = os.path.join(events_np_root, 'test')
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        print(f'Mkdir [{train_dir, test_dir}.')
        for label in range(11):
            os.mkdir(os.path.join(train_dir, str(label)))
            os.mkdir(os.path.join(test_dir, str(label)))
        print(f'Mkdir {os.listdir(train_dir)} in [{train_dir}] and {os.listdir(test_dir)} in [{test_dir}].')

        with open(os.path.join(aedat_dir, 'trials_to_train.txt')) as trials_to_train_txt, open(
                os.path.join(aedat_dir, 'trials_to_test.txt')) as trials_to_test_txt:
            # use multi-thread to accelerate
            t_ckp = time.time()
            with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), configure.max_threads_number_for_datasets_preprocess)) as tpe:
                print(f'Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].')

                for fname in trials_to_train_txt.readlines():
                    fname = fname.strip()
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedat_dir, fname)
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(DVS128Gesture.split_aedat_files_to_np, fname, aedat_file, os.path.join(aedat_dir, fname + '_labels.csv'), train_dir)

                for fname in trials_to_test_txt.readlines():
                    fname = fname.strip()
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedat_dir, fname)
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(DVS128Gesture.split_aedat_files_to_np, fname, aedat_file,
                                   os.path.join(aedat_dir, fname + '_labels.csv'), test_dir)

            print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'All aedat files have been split to samples and saved into [{train_dir, test_dir}].')

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 128, 128

class DVSAnimals(MyNeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = True,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            factor_tau: float = 0.8,
            scale_factor: int = 50
    ) -> None:
        """
        The DVS Animals dataset, which is proposed by Fully Event-Based Camera.

        The origin dataset can be split it into train and test set by ``train_test_split()`` from sklearn.
        """
        #Definitions of labels
        labeldef_root = os.path.join(root,'extract','SL-Animals-DVS_gestures_definitions.csv')
        labels_load = np.loadtxt(labeldef_root,dtype=str,delimiter=',',skiprows=1)
        labels_defs = dict()
        for label in labels_load:
            labels_defs[int(label[0]) - 1] = label[1]
        self.classes_def = labels_defs
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform,factor_tau,scale_factor)


    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        url = 'http://www2.imse-cnm.csic.es/neuromorphs/index.php/SL-ANIMALS-DVS-Database'
        return [
            ('allusers_aedat.rar',url,''),
            ('tags_updated_19_08_2020',url,'')
        ]

    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return False

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None

        This function defines how to extract download files.
        '''
        #fpath = os.path.join(download_root, 'DvsGesture.tar.gz')
        print(f'Extract it manually in animals folder. This folder contains .aedat and .csv archives')
        
    @staticmethod
    def get_aer_events_from_file(filename, data_version, data_start):
        """
        
        FUNCIÓN COPIADA DE TONIC.IO

        Get aer events from an aer file.

        Parameters:
            filename (str):         The name of the .aedat file
            data_version (float):   The version of the .aedat file
            data_start (int):       The start index of the data

        Returns:
            all_events:          Numpy structured array:
                                    ['address'] the address of a neuron which fires
                                    ['timeStamp'] the timeStamp in mus when a neuron fires
        """
        filename = os.path.expanduser(filename)
        assert os.path.isfile(filename), "The .aedat file does not exist."
        f = open(filename, "rb")
        f.seek(data_start)

        if 2 <= data_version < 3:
            event_dtype = np.dtype([("address", ">u4"), ("timeStamp", ">u4")]) 
            all_events = np.fromfile(f, event_dtype)
        elif data_version > 3:
            event_dtype = np.dtype([("address", "<u4"), ("timeStamp", "<u4")])
            event_list = []
            while True:
                header = f.read(28)
                if not header or len(header) == 0:
                    break

                # read header
                capacity = struct.unpack("I", header[16:20])[0]
                event_list.append(np.fromfile(f, event_dtype, capacity))
            all_events = np.concatenate(event_list)
        else:
            raise NotImplementedError()
        f.close()
        return all_events
    

    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        '''
        :param file_name: path of the events file
        :type file_name: str
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        :rtype: Dict

        This function defines how to read the origin binary data.
        '''

        """Get the aer events from DVS with resolution of rows and cols are (128, 128)

        Parameters:
            filename: filename

        Returns:
            a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        """
        data_version, data_start = tonic.io.read_aedat_header_from_file(file_name)
        all_events = DVSAnimals.get_aer_events_from_file(file_name, data_version, data_start)
        all_addr = all_events["address"]
        t = all_events["timeStamp"].astype(np.uint32)  

        x = (all_addr >> 8) & 0x007F
        y = (all_addr >> 1) & 0x007F
        p = all_addr & 0x1

        events_dict = dict()
        events_dict['x'] = np.array(x)
        events_dict['y'] = np.array(y)
        events_dict['t'] = np.array(t)
        events_dict['p'] = np.array(p)   

        return  events_dict               
    @staticmethod        
    def split_aedat_files_to_np(classes: dict,fname: str, aedat_file: str, csv_file: str, output_dir: str):
        """
        fname: parte inicial del nombre que le pondré al archivo .npz
        aedat_file: fichero aedat donde se leen los eventos
        csv_file: donde se leen las etiquetas y límites de tiempo de cada etiqueta para el correspondiente fichero aedat
        output_dir: archivo raiz donde se irán almacenando los .npz
        """
        events = DVSAnimals.load_origin_data(aedat_file)
        print(f'Start to split [{aedat_file}] to samples.')
        # read csv file and get time stamp and label of each sample
        # then split the origin data to samples
        csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',', skiprows=1)

        # Note that there are some files that many samples have the same label, e.g., user26_fluorescent_labels.csv
        label_file_num = [0] * 19
        for i in range(csv_data.shape[0]):
            # the label of Animals is 1, 2, ..., 19. We set 0 as the first label, rather than 1
            label = csv_data[i][0] - 1
            ind_start = csv_data[i][1]
            ind_end = csv_data[i][2]
            #print('etiqueta, ind_inicial, ind_final: ',label,ind_start,ind_end)
            file_name = os.path.join(output_dir, classes[label], f'{fname}_{label_file_num[label]}.npz')#Hago un directorio para cada etiqueta
            np_savez(file_name,
                     t=events['t'][ind_start:ind_end],
                     x=events['x'][ind_start:ind_end],
                     y=events['y'][ind_start:ind_end],
                     p=events['p'][ind_start:ind_end]
                     )
            print(f'[{file_name}] saved.')
            label_file_num[label] += 1
    
    @staticmethod
    def create_traintest_split_file(root, seed = 40,end_files = '.aedat',train_size = 0.8):
        """
        Create two .txt files in 'root' folder with the root of the samples for train and for test:
            trials_to_train.txt
            trials_to_test.txt
        Parameters:
            root (str): Root with the samples. This root must include inside one folder per classs.
        """
        random.seed(seed)
        file_train = os.path.join(root,'trials_to_train.txt')
        file_test = os.path.join(root,'trials_to_test.txt')
        if os.path.exists(file_train) or os.path.exists(file_test):
            raise ('Files to split data already exist. If they are wrong. Delete it')
        
        aedats_list = [file for file in os.listdir(root) if file.endswith(end_files)]
        #sublabel refers to the environment where the 'videos' where taken (indoor, sunlight....)
        sublabels_list = [file.split('_')[1] for file in aedats_list]
        train, test = train_test_split(aedats_list,stratify=sublabels_list,test_size=0.2,random_state=seed,train_size=train_size)

        with open(file_train,'w') as f_train:
            for file in train:
                f_train.write(file +'\n')

        with open(file_test,'w') as f_test:
            for file in test:
                f_test.write(file +'\n')

    def create_events_np_files(self,extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None

        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        aedat_dir = os.path.join(extract_root, 'Animals')
        train_dir = os.path.join(events_np_root, 'train')
        test_dir = os.path.join(events_np_root, 'test')
    
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        print(f'Mkdir [{train_dir, test_dir}.')
        for label in self.classes_def.values():
            os.mkdir(os.path.join(train_dir, label))
            os.mkdir(os.path.join(test_dir, label))
        print(f'Mkdir {os.listdir(train_dir)} in [{train_dir}] and {os.listdir(test_dir)} in [{test_dir}].')

        file_train = os.path.join(aedat_dir,'trials_to_train.txt')
        file_test = os.path.join(aedat_dir,'trials_to_test.txt')
        if  (not os.path.exists(file_train)) or (not os.path.exists(file_test)):
            print('Files to split data doesnt exist. Creating them: ',file_train,' ',file_test)
            DVSAnimals.create_traintest_split_file(aedat_dir)
        # use multi-thread to accelerate
        with open(file_train) as trials_to_train_txt, open(file_test) as trials_to_test_txt:
            # use multi-thread to accelerate
            t_ckp = time.time()
            with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), configure.max_threads_number_for_datasets_preprocess)) as tpe:
                print(f'Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                for fname in trials_to_train_txt.readlines():
                    fname = fname.strip()
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedat_dir, fname)
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(DVSAnimals.split_aedat_files_to_np,self.classes_def, fname, aedat_file, os.path.join(aedat_dir, fname + '.csv'), train_dir)
                for fname in trials_to_test_txt.readlines():
                    fname = fname.strip()
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedat_dir, fname)
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(DVSAnimals.split_aedat_files_to_np,self.classes_def, fname, aedat_file, os.path.join(aedat_dir, fname + '.csv'), test_dir)
            print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'All aedat files have been split to samples and saved into [{train_dir, test_dir}].')

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 128, 128
    
class DVSDailyActions(MyNeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = True,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            factor_tau: float = 1.5,
            scale_factor: int = 50
    ) -> None:
        """
        The DVS Animals dataset, which is proposed by Fully Event-Based Camera.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.


        .. admonition:: Note
            :class: note

            There are 1121 samples. See that they are all stored in the train folder.

            .. code-block:: python

                from spikingjelly.datasets import dvs128_gesture

                src_root = os.path.dirname(__file__)
                data_root = os.path.join(os.path.dirname(src_root),'data')
                data_animals = os.path.join(data_root,'SLAnimals_Dataset')
                data_set = DVSAnimals(data_animals,train=True, data_type='frame', frames_number=T, split_by=splitby)
                labels = [sample[1] for sample in data_set]
                print('Posible labels:',np.unique(labels))
                train_set, test_set = train_test_split(data_set, test_size = 0.2,stratify = np.array(labels), random_state = seed) 

            
            The origin dataset can be split it into train and test set by ``train_test_split()`` from sklearn.

        """
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform,factor_tau,scale_factor)
    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        url = 'https://drive.google.com/drive/folders/1JrYJnikaJdiNgq5Zz5pwbN-nwns-NNpz'
        #No files in download folder. The files are put it directly in extract folder
        return [] 

    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return False

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None

        This function defines how to extract download files.
        '''
        #fpath = os.path.join(download_root, 'DvsGesture.tar.gz')
        print(f'Extract it manually in .../extract/DailyAction folder. This folder contains one .aedat file per sample')
        
    @staticmethod
    def get_aer_events_from_file(filename, data_version, data_start):
        """
        
        FUNCIÓN COPIADA DE TONIC.IO

        Get aer events from an aer file.

        Parameters:
            filename (str):         The name of the .aedat file
            data_version (float):   The version of the .aedat file
            data_start (int):       The start index of the data

        Returns:
            all_events:          Numpy structured array:
                                    ['address'] the address of a neuron which fires
                                    ['timeStamp'] the timeStamp in mus when a neuron fires
        """
        filename = os.path.expanduser(filename)
        assert os.path.isfile(filename), "The .aedat file does not exist."
        f = open(filename, "rb")
        f.seek(data_start)

        if 2 <= data_version < 3:
            event_dtype = np.dtype([("address", ">u4"), ("timeStamp", ">u4")]) 
            all_events = np.fromfile(f, event_dtype)
        elif data_version > 3:
            event_dtype = np.dtype([("address", "<u4"), ("timeStamp", "<u4")])
            event_list = []
            while True:
                header = f.read(28)
                if not header or len(header) == 0:
                    break
                # read header
                capacity = struct.unpack("I", header[16:20])[0]
                event_list.append(np.fromfile(f, event_dtype, capacity))
            all_events = np.concatenate(event_list)
        else:
            raise NotImplementedError()
        f.close()
        return all_events
    

    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        '''
        :param file_name: path of the events file
        :type file_name: str
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        :rtype: Dict

        This function defines how to read the origin binary data.
        '''

        """Get the aer events from DVS with resolution of rows and cols are (128, 128)

        Parameters:
            filename: filename

        Returns:
            a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        """
        data_version, data_start = tonic.io.read_aedat_header_from_file(file_name)
        all_events = DVSDailyActions.get_aer_events_from_file(file_name, data_version, data_start)
        all_addr = all_events["address"]
        t = all_events["timeStamp"].astype(np.uint32)  

        x = (all_addr >> 8) & 0x007F
        y = (all_addr >> 1) & 0x007F
        p = all_addr & 0x1

        events_dict = dict()
        events_dict['x'] = np.array(x)
        events_dict['y'] = np.array(y)
        events_dict['t'] = np.array(t)
        events_dict['p'] = np.array(p)   

        return  events_dict                
        
                                           

    @staticmethod
    def split_aedat_files_to_np(fname: str, aedat_file: str, output_dir: str):
        """
        fname: parte inicial del nombre que le pondré al archivo .npz
        aedat_file: fichero aedat donde se leen los eventos
        #csv_file: donde se leen las etiquetas y límites de tiempo de cada etiqueta para el correspondiente fichero aedat
        output_dir: archivo raiz donde se irán almacenando los .npz
        """
        events = DVSDailyActions.load_origin_data(aedat_file)
        #print(f'Start to split [{aedat_file}] to samples.')
        #print('Tiempo de eventos: ',events['t'])

        file_name = os.path.join(output_dir, f'{fname}.npz')
        np_savez(file_name,
                    t=events['t'],
                    x=events['x'],
                    y=events['y'],
                    p=events['p']
                    )
        print(f'[{file_name}] saved.')
            
    @staticmethod
    def create_traintest_split_file(root, seed = 40,end_files = '.aedat',train_size = 0.8):
        """
        Create two .txt files in 'root' folder with the root of the samples for train and for test:
            trials_to_train.txt
            trials_to_test.txt
        Parameters:
            root (str): Root with the samples. This root must include inside one folder per classs.
        """
        random.seed(seed)
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
                        file = label + ',' + fname
                        if i < n_samples_train:
                            f_train.write(file +'\n')
                        else:
                            f_test.write(file +'\n')

    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None

        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        aedats_directories = os.path.join(extract_root, 'DailyAction')
        train_dir = os.path.join(events_np_root, 'train')
        test_dir = os.path.join(events_np_root, 'test')
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        print(f'Mkdir [{train_dir, test_dir}.')
        #subfolders name corresponds to the labels
        labels = [it.name for it in os.scandir(aedats_directories) if it.is_dir()] 
        for label in labels:
            os.mkdir(os.path.join(train_dir, label))
            os.mkdir(os.path.join(test_dir, label))
        print(f'Mkdir {os.listdir(train_dir)} in [{train_dir}] and {os.listdir(test_dir)} in [{test_dir}].')


        file_train = os.path.join(aedats_directories,'trials_to_train.txt')
        file_test = os.path.join(aedats_directories,'trials_to_test.txt')
        if  (not os.path.exists(file_train)) or (not os.path.exists(file_test)):
            print('Files to split data doesnt exist. Creating them: ',file_train,' ',file_test)
            DVSDailyActions.create_traintest_split_file(aedats_directories)

        with open(file_train) as trials_to_train_txt, open(file_test) as trials_to_test_txt:
            # use multi-thread to accelerate
            t_ckp = time.time()
            with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), configure.max_threads_number_for_datasets_preprocess)) as tpe:
                random.seed(42)
                print(f'Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                for line in trials_to_train_txt.readlines():
                    label, fname = line.strip().split(',')
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedats_directories,label, fname)
                        fname = os.path.splitext(fname)[0]
                        output_dir = os.path.join(train_dir,label)
                        tpe.submit(DVSDailyActions.split_aedat_files_to_np, fname, aedat_file,  output_dir)

                for line in trials_to_test_txt.readlines():
                    label, fname = line.strip().split(',')
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedats_directories,label, fname)
                        fname = os.path.splitext(fname)[0]
                        output_dir = os.path.join(test_dir,label)
                        tpe.submit(DVSDailyActions.split_aedat_files_to_np, fname, aedat_file,  output_dir)
            print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'All aedat files have been split to samples and saved into [{train_dir, test_dir}].')

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 128, 128  

class DVSActionRecog(MyNeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = True,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            factor_tau: float = 1.5,
            scale_factor: int = 50
    ) -> None:
        """
        Images were reduced by half size for convenience.
        """
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform,factor_tau,scale_factor)
    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        url = 'https://tongjieducn-my.sharepoint.com/personal/ziyang_tongji_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fziyang%5Ftongji%5Fedu%5Fcn%2FDocuments%2FPAFBenchmark%2FAction%20Recognition&ga=1'
        #No files in download folder. The files are put it directly in extract folder
        return [] 

    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return False

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None

        This function defines how to extract download files.
        '''
        #fpath = os.path.join(download_root, 'DvsGesture.tar.gz')
        print(f'Extract it manually in .../extract/ActionRecognition folder. This folder contains subfolders for each class with one .aedat file per sample.')
        
    @staticmethod
    def load_origin_data(file_name: str, startTime = 0, numEvents = 1e10) -> Dict:
        """ DESCRIPTION: This function reads a given aedat file and converts it into four lists indicating 
                     timestamps, x-coordinates, y-coordinates and polarities of the event stream. 
    
        Args:
            file: the path of the file to be read, including extension (str).
            numEvents: the maximum number of events allowed to be read (int, default value=1e10).
            startTime: the start event timestamp (in microseconds) where the conversion process begins (int, default value=0).

        Return:
            A dictionary with:
                t: list of timestamps in microseconds.
                x: list of x-coordinates in pixels.
                y: list of y-coordinates in pixels.`
                p: list of polarities (0: on -> off, 1: off -> on).       
        """
        sizeX = 346
        sizeY = 260
        x0 = 0
        y0 = 0
        x1 = sizeX
        y1 = sizeY

        triggerevent = int('400', 16)
        polmask = int('800', 16)
        xmask = int('003FF000', 16)
        ymask = int('7FC00000', 16)
        typemask = int('80000000', 16)
        typedvs = int('00', 16)
        xshift = 12
        yshift = 22
        polshift = 11
        x = []
        y = []
        ts = []
        pol = []
        
        length = 0
        aerdatafh = open(file_name, 'rb')
        k = 0
        p = 0
        statinfo = os.stat(file_name)
        if length == 0:
            length = statinfo.st_size
            
        lt = aerdatafh.readline()
        while lt and str(lt)[2] == "#":
            p += len(lt)
            k += 1
            lt = aerdatafh.readline()
            continue

        aerdatafh.seek(p)
        tmp = aerdatafh.read(8)
        p += 8
        while p < length:
            ad, tm = struct.unpack_from('>II', tmp)
            ad = abs(ad)
            if tm >= startTime:
                if (ad & typemask) == typedvs:
                    xo = sizeX - 1 - float((ad & xmask) >> xshift)
                    yo = float((ad & ymask) >> yshift)
                    polo = 1 - float((ad & polmask) >> polshift)
                    if xo >= x0 and xo < x1 and yo >= y0 and yo < y1:
                        x.append(xo)
                        y.append(yo)
                        pol.append(polo)
                        ts.append(tm)
            aerdatafh.seek(p)
            tmp = aerdatafh.read(8)
            p += 8

        events_dict = dict()
        events_dict['x'] = np.array(x).astype('uint32')//2
        events_dict['y'] = np.array(y).astype('uint32')//2
        events_dict['t'] = np.array(ts).astype('uint32')
        events_dict['p'] = np.array(pol).astype('uint32')

        return  events_dict                
        
                                           

    @staticmethod
    def split_aedat_files_to_np(fname: str, aedat_file: str, output_dir: str):
        """
        fname: parte inicial del nombre que le pondré al archivo .npz
        aedat_file: fichero aedat donde se leen los eventos
        output_dir: archivo raiz donde se irán almacenando los .npz
        """
        events = DVSActionRecog.load_origin_data(aedat_file)
        #print(f'Start to split [{aedat_file}] to samples.')
        #print('Tiempo de eventos: ',events['t'])

        file_name = os.path.join(output_dir, f'{fname}.npz')
        np_savez(file_name,
                    t=events['t'],
                    x=events['x'],
                    y=events['y'],
                    p=events['p']
                    )
        print(f'[{file_name}] saved.')
            
    @staticmethod
    def create_traintest_split_file(root, seed = 40,end_files = '.aedat',train_size = 0.8):
        """
        Create two .txt files in 'root' folder with the root of the samples for train and for test:
            trials_to_train.txt
            trials_to_test.txt
        Parameters:
            root (str): Root with the samples. This root must include inside one folder per classs.
        """
        random.seed(seed)
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
                        file = label + ',' + fname
                        if i < n_samples_train:
                            f_train.write(file +'\n')
                        else:
                            f_test.write(file +'\n')

    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None

        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        aedats_directories = os.path.join(extract_root, 'ActionRecognition')
        train_dir = os.path.join(events_np_root, 'train')
        test_dir = os.path.join(events_np_root, 'test')
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        print(f'Mkdir [{train_dir, test_dir}.')
        #subfolders name corresponds to the labels
        labels = [it.name for it in os.scandir(aedats_directories) if it.is_dir()] 
        for label in labels:
            os.mkdir(os.path.join(train_dir, label))
            os.mkdir(os.path.join(test_dir, label))
        print(f'Mkdir {os.listdir(train_dir)} in [{train_dir}] and {os.listdir(test_dir)} in [{test_dir}].')


        file_train = os.path.join(aedats_directories,'trials_to_train.txt')
        file_test = os.path.join(aedats_directories,'trials_to_test.txt')
        if  (not os.path.exists(file_train)) or (not os.path.exists(file_test)):
            print('Files to split data doesnt exist. Creating them: ',file_train,' ',file_test)
            DVSActionRecog.create_traintest_split_file(aedats_directories)

        with open(file_train) as trials_to_train_txt, open(file_test) as trials_to_test_txt:
            # use multi-thread to accelerate
            t_ckp = time.time()
            with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), configure.max_threads_number_for_datasets_preprocess)) as tpe:
                random.seed(42)
                print(f'Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                for line in trials_to_train_txt.readlines():
                    label, fname = line.strip().split(',')
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedats_directories,label, fname)
                        fname = os.path.splitext(fname)[0]
                        output_dir = os.path.join(train_dir,label)
                        tpe.submit(DVSActionRecog.split_aedat_files_to_np, fname, aedat_file,  output_dir)

                for line in trials_to_test_txt.readlines():
                    label, fname = line.strip().split(',')
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedats_directories,label, fname)
                        fname = os.path.splitext(fname)[0]
                        output_dir = os.path.join(test_dir,label)
                        tpe.submit(DVSActionRecog.split_aedat_files_to_np, fname, aedat_file,  output_dir)


            print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'All aedat files have been split to samples and saved into [{train_dir, test_dir}].')

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(346/2, 260/2)`` for the DAVIS346redcolor(Frame sizes where reduced by half).
            https://www.frontiersin.org/articles/10.3389/fnbot.2019.00038/full
        :rtype: tuple
        '''
        return 260//2, 346//2
    




mad_classes = {
  0: "cup_drink",
  1: "cup_pound",
  2: "cup_shake",
  3: "cup_move",
  4: "cup_pour",
  5: "stone_pound",
  6: "stone_move",
  7: "stone_play",
  8: "stone_grind",
  9: "stone_carve",
  10: "sponge_squeeze",
  11: "sponge_flip",
  12: "sponge_wash",
  13: "sponge_wipe",
  14: "sponge_scratch",
  15: "spoon_scoop",
  16: "spoon_stir",
  17: "spoon_hit",
  18: "spoon_eat",
  19: "spoon_sprinkle",
  20: "knife_cut",
  21: "knife_chop",
  22: "knife_poke a hole",
  23: "knife_peel",
  24: "knife_spread",
}

objects_mad = ['cup', 'stone', 'sponge', 'spoon', 'knife']   

class MAD():
    def __init__(
            self,
            root: str,
            train: bool = True,
            test_subj_id: int = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            factor_tau: float = 0.8,
            scale_factor: int = 50
    ) -> None:
        """
        If test_sub_id is not specified, full dataset will be loaded. If else, it depends in train bool value (train/test split)
        """
        self.n_steps = frames_number
        events_np_root = os.path.join(root, 'events_np')
        self.data_type = data_type

        if not os.path.exists(events_np_root):
            raise ('Please download events files from https://drive.ugr.es/index.php/s/rLq5wVayaoj8yff/download and unzip it in events_np folder')
        else:
            assert 'spatula' not in os.listdir(events_np_root), 'Please remove spatula folder'
            items = ['cup','knife','sponge','spoon','stone']
            items_file_list = MAD.list_only_dirs(events_np_root)
            if items == items_file_list:
                print('Rearranging files from events_np')
                for item in items_file_list:
                    for action in MAD.list_only_dirs(os.path.join(events_np_root,item)):
                        new_dir = os.path.join(events_np_root,item + '_' + action)
                        os.mkdir(new_dir)
                        for sample in sorted(os.listdir(os.path.join(events_np_root,item,action))):
                            origin = os.path.join(events_np_root,item,action,sample)
                            destination = os.path.join(new_dir,sample)
                            shutil.move(origin,destination)
                    shutil.rmtree(os.path.join(events_np_root,item))

            
        H, W = self.get_H_W()

        if data_type == 'event':
            _root = events_np_root
            _loader = np.load

        elif data_type == 'frame':
            if frames_number is not None:
                assert frames_number > 0 and isinstance(frames_number, int)
                assert split_by in ('time', 'number' ,'exp_decay')
                if split_by == 'exp_decay':
                    fact_tau = str(factor_tau).replace('.','_')
                    frames_np_root = os.path.join(root, f'frames_num{frames_number}_splitby_{split_by}_tau{fact_tau}_scale{scale_factor}')
                else:
                    frames_np_root = os.path.join(root, f'frames_number_{frames_number}_split_by_{split_by}')
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')

                    # create the same directory structure
                    sjds.create_same_directory_structure(events_np_root, frames_np_root)

                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        for e_root, e_dirs, e_files in os.walk(events_np_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    if e_file.endswith('.npy'):  
                                        events_np_file = os.path.join(e_root, e_file)
                                        print(f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                        tpe.submit(event_integration_to_frame.integrate_events_to_frame_wfixed_frames_num,
                                                    self.load_events_np, events_np_file, output_dir, split_by, frames_number, H, W,print_save= True,
                                                    factor_tau = factor_tau, scale_factor = scale_factor)
                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = sjds.load_npz_frames

            elif duration is not None:
                assert split_by in ['time','number']
                assert duration > 0 and isinstance(duration, int)
                frames_np_root = os.path.join(root, f'duration_{duration}')
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')

                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')
                    # create the same directory structure
                    sjds.create_same_directory_structure(events_np_root, frames_np_root)
                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        for e_root, e_dirs, e_files in os.walk(events_np_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    if e_file.endswith('.npz'):  
                                        events_np_file = os.path.join(e_root, e_file)
                                        print(f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                        tpe.submit(sjds.integrate_events_file_to_frames_file_by_fixed_duration, self.load_events_np, events_np_file, output_dir, duration, H, W, True)

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = sjds.load_npz_frames
                if custom_integrated_frames_dir_name is None:
                    custom_integrated_frames_dir_name = custom_integrate_function.__name__

                frames_np_root = os.path.join(root, custom_integrated_frames_dir_name)
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')
                    # create the same directory structure
                    sjds.create_same_directory_structure(events_np_root, frames_np_root)
                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        for e_root, e_dirs, e_files in os.walk(events_np_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    if e_file.endswith('.npz'):  
                                        events_np_file = os.path.join(e_root, e_file)
                                        print( f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                        tpe.submit(sjds.save_frames_to_npz_and_print, os.path.join(output_dir, os.path.basename(events_np_file)), custom_integrate_function(np.load(events_np_file), H, W))

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = sjds.load_npz_frames

            else:
                raise ValueError('At least one of "frames_number","duration" should not be None.')
            
        elif data_type == 'video':
            raw_vid_root = os.path.join(root, 'raw_video')
            items_jsons_root = os.path.join(root,'mad-meta-data')
            assert os.path.isdir(raw_vid_root) and os.path.isdir(items_jsons_root)
            vid_root = os.path.join(root,'video')
            if os.path.exists(vid_root):
                print('Video folder already exists.')
            else:
                os.mkdir(vid_root)
                sjds.create_same_directory_structure(events_np_root, vid_root)

                jsons_items_files = [it.name for it in os.scandir(items_jsons_root) if it.name.endswith('_db.json')]
                assert len(jsons_items_files) == 5
                dataset_info = dict()
                for json_item in jsons_items_files:
                    with open(os.path.join(items_jsons_root,json_item)) as f:
                        dataset_info[json_item] = json.load(f) 
                pll = 0
                for subject in MAD.list_only_dirs(raw_vid_root):
                    for item in MAD.list_only_dirs(os.path.join(raw_vid_root,subject)):
                        subj,itm,rgb = item.split('_')      #Folders looks like adam_cup_rgb
                        data_item = dataset_info[itm + '_db.json']
                        frames_dir = os.path.join(raw_vid_root,subject,item) 

                        for d in data_item:
                            if d['subject'] == subj and d['object'] == itm:
                                label = d['attention_type'] - 1 + objects_mad.index(d['object'])*5
                                assert itm in mad_classes[label]

                                action_class_folder = os.path.join(vid_root, mad_classes[label])
                                if not os.path.exists(action_class_folder):
                                    raise ValueError('La estructura de ficheros fue copia de events_np. No debe de haber error aquí')
                                    #print(f'Creating {action_class_folder}')
                                    #os.mkdir(action_class_folder)

                                startf,endf = d['start_frame'],d['end_frame']
                                 
                                MAD.jpg_files_to_npz( start = startf, end = endf, input_dir = frames_dir, output_dir = action_class_folder,segment_id = d['segment_id'], subj_id = d['sub_id'])                         

            _root = vid_root
            _loader = self.load_npz_video_with_T_frames

        self.loader = _loader
        self.classes, self.class_to_idx = self.find_classes()
        self.data,self.subjects = self.make_dataset(directory = _root,class_to_idx = self.class_to_idx, extensions = ('.npz','.npy'))

        if test_subj_id is not None:
            assert test_subj_id in [1,2,3,4,5]
            if train:
                self.data = self.data[self.subjects != test_subj_id]
                self.subjects[self.subjects != test_subj_id]
            else:
                self.data = self.data[self.subjects == test_subj_id]
                self.subjects[self.subjects == test_subj_id]

    def set_root_when_train_is_none(self, _root: str):
        return _root
    
    def load_npz_video_with_T_frames(self,file_name):
        all_frames_img = np.load(file_name, allow_pickle = True)['video']
        n_frames = len(all_frames_img)
        rate = n_frames/self.n_steps
        idxs = (np.arange(1,self.n_steps + 1) * rate - 1).astype(np.uint8)
        final_image = all_frames_img[idxs].astype(np.float32)
        return final_image

    @staticmethod
    def jpg_files_to_npz(start: int, end: int, input_dir: str, output_dir: str,segment_id: int, subj_id: int):
        frames = []
        #print(f'Start {start}, end {end}, segment_id {segment_id}, subject_id {subj_id}')
        for i in range(start,end+1):
            jpg_name = str(i).zfill(8) + '.jpg'
            img = MAD.preprocess_image(os.path.join(input_dir,jpg_name))
            frames.append(img)
        frames = np.array(frames)
        outp_name_file = os.path.join(output_dir,f'S{subj_id}_segment_' + str(segment_id).zfill(3))
        np.savez(outp_name_file, video = frames)

    @staticmethod
    def preprocess_image(img_path):
        img = cv2.imread(img_path)
        if img.shape[2] != 3:
            raise ImportError(f'File {img_path} doesnt have three channels.')
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_gray_resized = cv2.resize(img_gray,(196,147))
        return img_gray_resized

    @staticmethod
    def list_only_dirs(root):
        return sorted([it.name for it in os.scandir(root) if it.is_dir()] )
    
    def get_H_W(self) -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
        :rtype: tuple
        '''
        if self.data_type in ['event','frame']:
            return 180, 160   #Estas son las dimensiones para la cámara DVS. Para la cámara convencional las dimensiones son 147,196.
        elif self.data_type == 'video':
            return 147, 196
        else:
            raise ValueError

    @staticmethod
    def load_events_np(fname: str):
        '''
        :param fname: file name
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        This function defines how to load a sample from `events_np`. In most cases, this function is `np.load`.
        But for some datasets, e.g., ES-ImageNet, it can be different.
        '''
        events = np.load(fname, allow_pickle=True).item()
        #Rename times key: ts -> t
        events['t'] = events['ts']
        del events['ts']
        #Rename polarity: [-1,1] -> [0,1]
        events['p'][ events['p'] == -1 ] = 0
        return events
    
    @staticmethod
    def find_classes():
        classes = list(mad_classes.values())
        class_to_idx = {cls_name: i  for i,cls_name in mad_classes.items()}

        return classes, class_to_idx
    
    def make_dataset(self,directory: str, class_to_idx: dict, extensions):

        directory = os.path.expanduser(directory)
        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        #extensions = ('.npz','.npy')
        def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions)) 
        
        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        subjects = []
        max_len_str = 0
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                print(target_dir)
                raise ValueError('Debe de haber algun error entre los nombres de las clases y el de las carpetas')
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)

                    if len(path) > max_len_str:
                        max_len_str = len(path)

                    if is_valid_file(path):
                        subject_id = int(fname[1])          #fname[1] contiene el subject id... S1_segment_001.npy por ejemplo
                        assert subject_id in [1,2,3,4,5]
                        item = path, class_index
                        subject = subject_id 
                        subjects.append(subject)
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return np.array(instances, dtype=[('string', f'U{max_len_str}'), ('integer', int)]), np.array(subjects)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.data[index]
        x = self.loader(path)
        return x, target

    def __len__(self) -> int:
        return len(self.data)