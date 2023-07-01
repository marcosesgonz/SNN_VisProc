from typing import Callable, Dict, Optional, Tuple
import numpy as np
from spikingjelly import datasets as sjds
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
from spikingjelly import configure
from spikingjelly.datasets import np_savez
import struct
import tonic

class DVSAnimals(sjds.NeuromorphicDatasetFolder):
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
    ) -> None:
        """
        The DVS128 Gesture dataset, which is proposed by `A Low Power, Fully Event-Based Gesture Recognition System <https://openaccess.thecvf.com/content_cvpr_2017/html/Amir_A_Low_Power_CVPR_2017_paper.html>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.


        .. admonition:: Note
            :class: note

            In SpikingJelly, there are 1176 train samples and 288 test samples. The total samples number is 1464.

            .. code-block:: python

                from spikingjelly.datasets import dvs128_gesture

                data_dir = 'D:/datasets/DVS128Gesture'
                train_set = dvs128_gesture.DVS128Gesture(data_dir, train=True)
                test_set = dvs128_gesture.DVS128Gesture(data_dir, train=False)
                print(f'train samples = {train_set.__len__()}, test samples = {test_set.__len__()}')
                print(f'total samples = {train_set.__len__() + test_set.__len__()}')

                # train samples = 1176, test samples = 288
                # total samples = 1464


            While from the origin paper, `the DvsGesture dataset comprises 1342 instances of a set of 11 hand and arm \
            gestures`. The difference may be caused by different pre-processing methods.

            `snnTorch <https://snntorch.readthedocs.io/>`_ have the same numbers with SpikingJelly:

            .. code-block:: python

                from snntorch.spikevision import spikedata

                train_set = spikedata.DVSGesture("D:/datasets/DVS128Gesture/temp2", train=True, num_steps=500, dt=1000)
                test_set = spikedata.DVSGesture("D:/datasets/DVS128Gesture/temp2", train=False, num_steps=1800, dt=1000)
                print(f'train samples = {train_set.__len__()}, test samples = {test_set.__len__()}')
                print(f'total samples = {train_set.__len__() + test_set.__len__()}')

                # train samples = 1176, test samples = 288
                # total samples = 1464


            But `tonic <https://tonic.readthedocs.io/>`_ has different numbers, which are close to `1342`:

            .. code-block:: python

                import tonic

                train_set = tonic.datasets.DVSGesture(save_to='D:/datasets/DVS128Gesture/temp', train=True)
                test_set = tonic.datasets.DVSGesture(save_to='D:/datasets/DVS128Gesture/temp', train=False)
                print(f'train samples = {train_set.__len__()}, test samples = {test_set.__len__()}')
                print(f'total samples = {train_set.__len__() + test_set.__len__()}')

                # train samples = 1077, test samples = 264
                # total samples = 1341


            Here we show how 1176 train samples and 288 test samples are got in SpikingJelly.

            The origin dataset is split to train and test set by ``trials_to_train.txt`` and ``trials_to_test.txt``.


            .. code-block:: shell

                trials_to_train.txt:

                    user01_fluorescent.aedat
                    user01_fluorescent_led.aedat
                    ...
                    user23_lab.aedat
                    user23_led.aedat

                trials_to_test.txt:

                    user24_fluorescent.aedat
                    user24_fluorescent_led.aedat
                    ...
                    user29_led.aedat
                    user29_natural.aedat

            SpikingJelly will read the txt file and get the aedat file name like ``user01_fluorescent.aedat``. The corresponding \
            label file name will be regarded as ``user01_fluorescent_labels.csv``.

            .. code-block:: shell

                user01_fluorescent_labels.csv:

                    class	startTime_usec	endTime_usec
                    1	80048239	85092709
                    2	89431170	95231007
                    3	95938861	103200075
                    4	114845417	123499505
                    5	124344363	131742581
                    6	133660637	141880879
                    7	142360393	149138239
                    8	150717639	157362334
                    8	157773346	164029864
                    9	165057394	171518239
                    10	172843790	179442817
                    11	180675853	187389051




            Then SpikingJelly will split the aedat to samples by the time range and class in the csv file. In this sample, \
            the first sample ``user01_fluorescent_0.npz`` is sliced from the origin events ``user01_fluorescent.aedat`` with \
            ``80048239 <= t < 85092709`` and ``label=0``. ``user01_fluorescent_0.npz`` will be saved in ``root/events_np/train/0``.





        """
        print('Starting new version')
        assert train is not None
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
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