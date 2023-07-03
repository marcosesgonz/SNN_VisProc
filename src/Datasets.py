from typing import Callable, Dict, Optional, Tuple
import numpy as np
from spikingjelly import datasets as sjds
from torchvision.datasets import DatasetFolder
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
from spikingjelly import configure
from spikingjelly.datasets import np_savez
import struct
import tonic
import random

class DVSAnimals(sjds.NeuromorphicDatasetFolder):
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
        assert train is True
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
        
        #Definitions of labels
        labeldef_root = os.path.join(root,'extract','SL-Animals-DVS_gestures_definitions.csv')
        labels_load = np.loadtxt(labeldef_root,dtype=str,delimiter=',',skiprows=1)
        labels_defs = dict()
        for label in labels_load:
            labels_defs[label[1]] = int(label[0]) - 1
        self.class_to_idx = labels_defs

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
        t = all_events["timeStamp"].astype(np.uint32)  #Añadí el '*1e-3'Tonic daba el tiempo en nanosegunds(mus), por lo que lo paso a microsegundos(us)(QUITADO)

        x = (all_addr >> 8) & 0x007F
        y = (all_addr >> 1) & 0x007F
        p = all_addr & 0x1

        events_dict = dict()
        events_dict['x'] = np.array(x)
        events_dict['y'] = np.array(y)
        events_dict['t'] = np.array(t)
        events_dict['p'] = np.array(p)   

        return  events_dict                 #sjds.load_aedat_v3(file_name)
        
                                           

    @staticmethod
    def split_aedat_files_to_np(fname: str, aedat_file: str, csv_file: str, output_dir: str):
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

        # There are some wrong time stamp in this dataset, e.g., in user22_led_labels.csv, ``endTime_usec`` of the class 9 is
        # larger than ``startTime_usec`` of the class 10. So, the following codes, which are used in old version of SpikingJelly,
        # are replaced by new codes.
        #print('Tiempo de eventos: ',events['t'])

        for i in range(csv_data.shape[0]):
            # the label of Animals is 1, 2, ..., 19. We set 0 as the first label, rather than 1
            label = csv_data[i][0] - 1
            ind_start = csv_data[i][1]
            ind_end = csv_data[i][2]
            #print('etiqueta, ind_inicial, ind_final: ',label,ind_start,ind_end)
            file_name = os.path.join(output_dir, str(label), f'{fname}_{label_file_num[label]}.npz')#Hago un directorio para cada etiqueta
            np_savez(file_name,
                     t=events['t'][ind_start:ind_end],
                     x=events['x'][ind_start:ind_end],
                     y=events['y'][ind_start:ind_end],
                     p=events['p'][ind_start:ind_end]
                     )
            print(f'[{file_name}] saved.')
            label_file_num[label] += 1
            

        # old codes:

        # index = 0
        # index_l = 0
        # index_r = 0
        # for i in range(csv_data.shape[0]):
        #     # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
        #     label = csv_data[i][0] - 1
        #     t_start = csv_data[i][1]
        #     t_end = csv_data[i][2]
        #
        #     while True:
        #         t = events['t'][index]
        #         if t < t_start:
        #             index += 1
        #         else:
        #             index_l = index
        #             break
        #     while True:
        #         t = events['t'][index]
        #         if t < t_end:
        #             index += 1
        #         else:
        #             index_r = index
        #             break
        #
        #     file_name = os.path.join(output_dir, str(label), f'{fname}_{label_file_num[label]}.npz')
        #     np.savez(file_name,
        #         t=events['t'][index_l:index_r],
        #         x=events['x'][index_l:index_r],
        #         y=events['y'][index_l:index_r],
        #         p=events['p'][index_l:index_r]
        #     )
        #     print(f'[{file_name}] saved.')
        #     label_file_num[label] += 1

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
        aedat_dir = os.path.join(extract_root, 'Animals')
        data_dir = os.path.join(events_np_root, 'train')
    
        os.mkdir(data_dir)
        print(f'Mkdir {data_dir}.')
        for label in range(19):
            os.mkdir(os.path.join(data_dir, str(label)))
        print(f'Mkdir {os.listdir(data_dir)} in [{data_dir}].')

        #with open(os.path.join(aedat_dir, 'trials_to_train.txt')) as trials_to_train_txt, open(
        #        os.path.join(aedat_dir, 'trials_to_test.txt')) as trials_to_test_txt:
        # use multi-thread to accelerate
        t_ckp = time.time()
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), configure.max_threads_number_for_datasets_preprocess)) as tpe:
            print(f'Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].')

            for fname in os.listdir(aedat_dir):
                if fname.endswith('.aedat'):
                    aedat_file = os.path.join(aedat_dir, fname)
                    fname = os.path.splitext(fname)[0]
                    tpe.submit(DVSAnimals.split_aedat_files_to_np, fname, aedat_file, os.path.join(aedat_dir, fname + '.csv'), data_dir)


        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'All aedat files have been split to samples and saved into [{data_dir}].')

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 128, 128
    

class DVSDailyActions(sjds.NeuromorphicDatasetFolder):
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
        print('Starting new version')
        assert train is True
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
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
        data_dir = os.path.join(events_np_root, 'train')
        os.mkdir(data_dir)
        print(f'Mkdir {data_dir}.')
        #subfolders name corresponds to the labels
        labels = [it.name for it in os.scandir(aedats_directories) if it.is_dir()] 
        for label in labels:
            os.mkdir(os.path.join(data_dir, label))
        print(f'Mkdir {os.listdir(data_dir)} in [{data_dir}].')

        # use multi-thread to accelerate
        t_ckp = time.time()
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), configure.max_threads_number_for_datasets_preprocess)) as tpe:
            print(f'Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].')

            for label in labels:
                aedat_dir = os.path.join(aedats_directories,label)
                output_dir = os.path.join(data_dir, label)
                for fname in os.listdir(aedat_dir):
                    if fname.endswith('.aedat'):
                        aedat_file = os.path.join(aedat_dir, fname)
                        #File name without .aedat
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(DVSDailyActions.split_aedat_files_to_np, fname, aedat_file, output_dir)


        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'All aedat files have been split to samples and saved into [{data_dir}].')

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 128, 128
    

class DVSActionRecog(sjds.NeuromorphicDatasetFolder):
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
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
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
        events_dict['x'] = np.array(x)
        events_dict['y'] = np.array(y)
        events_dict['t'] = np.array(ts)
        events_dict['p'] = np.array(pol) 

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
            For example, this function returns ``(346, 260)`` for the DAVIS346redcolor.
            https://www.frontiersin.org/articles/10.3389/fnbot.2019.00038/full
        :rtype: tuple
        '''
        return 260, 346