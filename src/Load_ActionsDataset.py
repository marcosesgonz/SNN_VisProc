#import cPickle
import gzip
import os
import json
import numpy
import scipy.io

__datasets = {}

actual_path = os.path.dirname(__file__)
father_path = os.path.dirname(os.path.abspath(actual_path))

mad_data_dir = os.path.join(father_path,'data/Actions_Dataset/MAD')
haf_data_dir = os.path.join(father_path,'data/Actions_Dataset/HAF')
mad_force_result = 'result/estimated_force/regression_mad.mat'
default_valid_portion = 0.1

def prepare_minibatch(data, maxlen=None, unfold_y=False):
    """ Prepare one minibatch
    :data: tuple of each data elements
           data[0] should always be the feature sequence
    :maxlen: remove sequences that longer than this maximum length
    :unfold_y: if true, unfold label y to a sequence
    """
    if maxlen is not None:
        selected = [ i for i in len(data[0]) if data[0][i].shape[0] < maxlen ]
        if len(selected) < len(data[0]):
            data = [ [x[i] for i in selected] for x in data ]

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(data[0])
    lengths = [s.shape[0] for s in data[0]]
    maxlen = max(lengths)
    mask = numpy.zeros((maxlen, n_samples)).astype(numpy.float)
    for i in range(n_samples):
        mask[:lengths[i], i] = 1.

    data_new = []
    for seq in data:
        if type(seq[0]) is int:
            if unfold_y:
                seq_new = numpy.zeros((maxlen, n_samples, 1)).astype('int32')
                for i in range(n_samples):
                    seq_new[:lengths[i], i, :] = seq[i]
            else:
                seq_new = seq

        else:
            seq_new = numpy.zeros((maxlen, n_samples, seq[0].shape[1])).astype(numpy.float)
            for i in range(n_samples):
                seq_new[:lengths[i], i, :] = seq[i]

        data_new.append(seq_new)

    return [data_new[0], mask,]+data_new[1:]


def load_dataset_haf(valid_portion=default_valid_portion, test_subject=None, object_list=None, spec='xf'):
    '''Loads the dataset
    :valid_portion: The proportion of the full train set used for the validation set.
    :object_list: select objects
    :spec: dataset specification
              'xf'  video + force
              'xy'  video + class_label
              'xfy' video + force + class_label
    '''
    data_dir = haf_data_dir

    # subject_list = ['fbr', 'fbr2', 'fm', 'leo', 'yz']
    subject_list = __subjects_haf
    if object_list is None:
        # object_list = ['cup', 'fork', 'knife', 'sponge']
        object_list = __objects_haf

    set_x = []
    set_f = []
    set_y = []
    set_meta = []

    print('reading datasets:')
    if test_subject is not None:
        print('    test subject: %s' % (test_subject))
    if object_list is not None:
        print('    using objects:', object_list)

    for sbj in subject_list:
        dataset_path = os.path.join(data_dir, '%s_db.json' % (sbj))
        print('loading data: %s' % (dataset_path, ))
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            dataset = [d for d in dataset if d['object'] in object_list]

        # load the image features into memory
        features_path = os.path.join(data_dir, '%s_feats.mat' % (sbj))
        # print 'loading features: %s' % (features_path, )
        features_struct = scipy.io.loadmat(features_path)
        features = features_struct['feats']

        for i, d in enumerate(dataset):
            feats = numpy.asarray(features[d['start_fid']:d['end_fid']+1, :])
            d['attention_type'] = d['attention_type'] + object_list.index(d['object'])*5
            d['test_flag'] = False
            if test_subject is None and (i+1)%5 == 0:
                d['test_flag'] = True
            if test_subject is not None and d['subject']==test_subject:
                d['test_flag'] = True

            set_x.append(feats)

        set_f += [numpy.asarray(d['force_data'], 'float32') for d in dataset]
        set_y += [d['attention_type']-1 for d in dataset]
        set_meta += dataset

    if spec == 'xfy':
        dataset = [set_x, set_f, set_y, set_meta]
    elif spec == 'xf':
        dataset = [set_x, set_f, set_meta]
    elif spec == 'xy':
        dataset = [set_x, set_y, set_meta]
    else:
        print('Error: dataset specification can only be \'xfy\', \'xf\', or \'xy\', \'%s\' was given.' % (spec))

    # split training and testing set
    train_idx = [ i for i in range(len(set_meta)) if not set_meta[i]['test_flag'] ]
    test_idx = [ i for i in range(len(set_meta)) if set_meta[i]['test_flag'] ]
    test = [ [x[i] for i in test_idx] for x in dataset ]
    train = [ [x[i] for i in train_idx] for x in dataset ]

    # shuffle training set, then split validation set from available training data
    n_samples = len(train_idx)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid = [ [x[i] for i in sidx[n_train:]] for x in train ]
    train = [ [x[i] for i in sidx[:n_train]] for x in train ]

    return dataset#train, valid, test


def load_dataset_mad(valid_portion=default_valid_portion, test_subject=None, object_list=None, spec='xy', shuffle=True):
    '''Loads the dataset
    :valid_portion: The proportion of the full train set used for the validation set.
    :object_list: select objects
    :spec: dataset specification
              'xf'  video + force
              'xy'  video + class_label
              'xfy' video + force + class_label
    '''
    data_dir = mad_data_dir

    # subject_list = ['fbr', 'fbr2', 'fm', 'leo', 'yz']
    subject_list = __subjects_mad
    if object_list is None:
        # object_list = ['cup', 'fork', 'knife', 'sponge']
        object_list = __objects_mad

    set_x = []
    set_f = []
    set_y = []
    set_meta = []

    print('reading datasets:')
    if test_subject is not None:
        print('    test subject: %s' % (test_subject))
    if object_list is not None:
        print('    using objects:', object_list)

    fake_force = True
    if os.path.exists(mad_force_result):
        fake_force = False
        force_data_all = scipy.io.loadmat(mad_force_result)

    for obj in object_list:
        dataset_path = os.path.join(data_dir, '%s_db.json' % (obj))
        print('loading data: %s' % (dataset_path, ))
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            dataset = [d for d in dataset if d['object'] in object_list]

        # load the image features into memory
        features_path = os.path.join(data_dir, '%s_feat.mat' % (obj))
        # print 'loading features: %s' % (features_path, )
        features_struct = scipy.io.loadmat(features_path)
        features = features_struct['feats'].T

        if not fake_force:
            force_data = force_data_all['force_all'][0]
            force_data = force_data_all['force_all'][0][numpy.where(force_data_all['object_id'] == obj.ljust(6))]
            ct = 0

        for i, d in enumerate(dataset):
            feats = numpy.asarray(features[d['start_fid']:d['end_fid']+1, :])
            d['attention_type'] = d['attention_type'] + object_list.index(d['object'])*5
            d['test_flag'] = False
            if test_subject is None and (i+1)%5 == 0:
                d['test_flag'] = True
            if test_subject is not None and d['subject']==test_subject:
                d['test_flag'] = True

            set_x.append(feats)
            if fake_force:
                set_f.append(numpy.zeros((feats.shape[0], 4), 'float32'))  # fake force
            else:
                set_f.append(force_data[ct])
                ct += 1

        set_y += [d['attention_type']-1 for d in dataset]
        set_meta += dataset

    if spec == 'xfy':
        dataset = [set_x, set_f, set_y, set_meta]
    elif spec == 'xf':
        dataset = [set_x, set_f, set_meta]
    elif spec == 'xy':
        dataset = [set_x, set_y, set_meta]
    else:
        print('Error: dataset specification can only be \'xfy\', \'xf\', or \'xy\', \'%s\' was given.' % (spec))

    # split training and testing set
    train_idx = [ i for i in range(len(set_meta)) if not set_meta[i]['test_flag'] ]
    test_idx = [ i for i in range(len(set_meta)) if set_meta[i]['test_flag'] ]
    test = [ [x[i] for i in test_idx] for x in dataset ]
    train = [ [x[i] for i in train_idx] for x in dataset ]

    # shuffle training set, then split validation set from available training data
    n_samples = len(train_idx)
    if shuffle:
        sidx = numpy.random.permutation(n_samples)
    else:
        sidx = numpy.asarray(range(n_samples))
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid = [ [x[i] for i in sidx[n_train:]] for x in train ]
    train = [ [x[i] for i in sidx[:n_train]] for x in train ]

    print('Done')
    return dataset#train, valid, test


## Register datasets
__objects_haf = ['cup', 'fork', 'knife', 'sponge']
__subjects_haf = ['fbr', 'fbr2', 'fm', 'leo', 'yz']
__objects_mad = ['cup', 'stone', 'sponge', 'spoon', 'knife']
__subjects_mad = ['and', 'fer', 'gui', 'mic', 'kos']

# MAD
__datasets['action_mad'] = (
    lambda: load_dataset_mad(spec='xy'),
    prepare_minibatch
    )

# The following mode is only for applying force regression on MAD dataset.
# So make sure we turn off the shuffle tag and do not split validation set.
__datasets['force_mad'] = (
    lambda: load_dataset_mad(valid_portion=0.0, test_subject='',spec='xf',shuffle=False),
    prepare_minibatch
    )
for sbj in __subjects_mad:
    for obj in __objects_mad:
        __datasets['action_'+obj+'_'+sbj+'_mad'] = (lambda sbj=sbj, obj=obj: load_dataset_mad(spec='xy',
                                                        test_subject=sbj,
                                                        object_list=[obj,]),
                                             prepare_minibatch)
        __datasets['action_n_force_'+obj+'_'+sbj+'_mad'] = (lambda sbj=sbj, obj=obj: load_dataset_mad(spec='xfy',
                                                        test_subject=sbj,
                                                        object_list=[obj,]),
                                             prepare_minibatch)

# HAF
__datasets['force'] = (
    lambda: load_dataset_haf(spec='xf'),
    prepare_minibatch
    )
__datasets['action'] = (
    lambda: load_dataset_haf(spec='xy'),
    prepare_minibatch
    )
__datasets['action_n_force'] = (
    lambda: load_dataset_haf(spec='xfy'),
    prepare_minibatch
    )


for obj in __objects_haf:
    __datasets['action_'+obj] = (lambda obj=obj: load_dataset_haf(spec='xy', object_list=[obj,]),
                                 prepare_minibatch)
    __datasets['action_n_force_'+obj] = (lambda obj=obj: load_dataset_haf(spec='xfy', object_list=[obj,]),
                                 prepare_minibatch)

for sbj in __subjects_haf:
    for obj in __objects_haf:
        __datasets['action_'+obj+'_'+sbj] = (lambda sbj=sbj, obj=obj: load_dataset_haf(spec='xy',
                                                        test_subject=sbj,
                                                        object_list=[obj,]),
                                             prepare_minibatch)
        __datasets['action_n_force_'+obj+'_'+sbj] = (lambda sbj=sbj, obj=obj: load_dataset_haf(spec='xfy',
                                                        test_subject=sbj,
                                                        object_list=[obj,]),
                                             prepare_minibatch)

print(__datasets.keys())
def get_dataset(name):
    """ Get dataset loading and preparing function handles by name. """
    if __datasets['name'] == None:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __datasets[name]

