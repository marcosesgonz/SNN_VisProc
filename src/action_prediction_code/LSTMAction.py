import cPickle as pkl
import os, sys
import time
import numpy as np
import scipy.io as sio

import DataFactory
from lstm import *

import pylab


# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    randn = np.random.rand(options['feat_dim'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * np.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = np.zeros((options['ydim'],)).astype(config.floatX)

    return params


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.tensor3('x', dtype=config.floatX)
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = theano.dot(x, tparams['Wemb'])
    output, one_step_func = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)

    output = output * mask[:, :, None]

    if options['use_dropout']:
        output = dropout_layer(output, use_noise, trng)

    # mean pooling
    wg = tensor.arange(n_timesteps).astype(config.floatX)
    wg = wg[:, None] / mask.sum(axis=0)
    proj_all = output * wg[:, :, None]
    proj = proj_all.sum(axis=0)
    proj = proj / mask.sum(axis=0)[:, None]
    proj = tensor.dot(proj, tparams['U']) + tparams['b']
    pred = tensor.nnet.softmax(proj)

    off = 1e-8
    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    return use_noise, x, mask, y, f_pred, cost


def pred_probs(f_pred, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = np.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([ [data[0][t] for t in valid_index],
                                    [data[1][t] for t in valid_index] ])
        pred_probs = f_pred(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples processed' % (n_done, n_samples)

    return probs

def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([ [data[0][t] for t in valid_index],
                                    [data[1][t] for t in valid_index] ])
        preds = f_pred(x, mask)
        targets = [data[1][t] for t in valid_index]
        valid_err += (preds == targets).sum()

    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])
    return valid_err


def train_model(model_file, use_object, test_subject, run_id, options):

    figure_dir = options['figure_dir']
    batch_size = options['batch_size']

    load_data, prepare_data = DataFactory.get_dataset('action_'+use_object+'_'+test_subject+'_mad')

    print 'Loading data'
    train, valid, test = load_data()
    nCls = len(np.unique(train[1]))
    ydim = nCls
    options['ydim'] = ydim
    print "model options", options

    print 'Building model'
    params = init_params(options)

    if options['reload_model']:
        load_params(model_file, params)

    tparams = init_tparams(params)
    (use_noise, x, mask, y, f_pred, cost) = build_model(tparams, options)

    if options['decay_c'] > 0.:
        decay_c = options['decay_c']
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    grads = tensor.grad(cost, wrt=tparams.values())

    lr = tensor.scalar(name='lr')
    symlist = [x, mask, y]
    f_grad_shared, f_update = options['optimizer'](lr, tparams, grads, symlist, cost)

    kf_valid = get_minibatches_idx(len(valid[0]), options['valid_batch_size'])

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    best_p = None
    bad_count = 0

    if options['validFreq'] == -1:
        options['validFreq'] = len(train[0]) / batch_size
    if options['saveFreq'] == -1:
        options['saveFreq'] = len(train[0]) / batch_size

    print 'Start optimization'
    uidx = 0  # the number of iterations
    estop = False
    best_eidx = 0
    start_time = time.time()
    try:
        for eidx in xrange(options['max_epochs']):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                x = [train[0][t] for t in train_index]
                y = [train[1][t] for t in train_index]

                # Get the data in np.ndarray format
                x, mask, y = prepare_data([x, y])
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(options['lrate'])

                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if np.mod(uidx, options['dispFreq']) == 0:
                    print 'Epoch', eidx+1, ', Iteration', uidx, ', Cost', cost

                if np.mod(uidx, options['validFreq']) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)

                    history_errs.append([train_err, valid_err])

                    pylab.figure(1); pylab.clf()
                    lines = pylab.plot(np.array(history_errs))
                    pylab.legend(lines, ['train', 'valid'])
                    pylab.savefig("%s/err_action_%s_%s_run%d.png"%(figure_dir, use_object, test_subject, run_id))
                    time.sleep(0.1)

                    # update best params for 1st iteration, lower validation error, or no validation
                    if (uidx == 0 or
                        valid_err <= np.array(history_errs)[:,1].min() or
                        np.isnan(valid_err)):

                        best_p = unzip(tparams)
                        best_eidx = eidx
                        bad_counter = 0
                        if valid_err <= np.array(history_errs)[:,1].min():
                            print '  New best validation results.'

                    print 'TrainErr=%.06f  ValidErr=%.06f' % (train_err, valid_err)

                    if (len(history_errs) > options['patience'] and
                        valid_err >= np.array(history_errs)[:-options['patience'], 1].min()):
                        bad_counter += 1
                        if bad_counter > options['patience']:
                            print 'Early stop at epoch %d, saved best epoch %d' % (eidx+1, best_eidx+1)
                            estop = True
                            break

            if model_file and (eidx+1) % options['saveFreq'] == 0:
                print 'Saving...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                np.savez(model_file, history_errs=history_errs, **params)
                pkl.dump(options, open('%s.pkl' % model_file, 'wb'), -1)
                print 'Done'

            if estop:
                break


    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()

    print 'Saved best epoch %d' % (best_eidx+1)
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)

    print 'Train ', train_err, 'Valid ', valid_err
    if model_file:
        np.savez(model_file, train_err=train_err,
                    valid_err=valid_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, valid_err


def confusion_matrix(gt, pred, nCls):
    cm = np.zeros((nCls, nCls))
    for i in xrange(nCls):
        idxCls = np.where(gt==i)[0]
        if idxCls.size == 0:
            continue
        predCls = pred[idxCls]
        for j in xrange(nCls):
            cm[j, i] = np.where(predCls==j)[0].shape[0]

    return cm

def test_confusion_matrix():
    from random import randint
    nCls = 10
    gt = np.tile(np.arange(nCls), (7, 1))
    gt = np.reshape(gt.T, (gt.size, ))
    pred = np.asarray([randint(0, nCls-1) for x in xrange(gt.size)])
    cm = confusion_matrix(gt, pred, nCls)

    # print cm
    print np.sum(cm, axis=0)
    print np.sum(cm, axis=1)
    return cm


def test_model(model_file, use_object, test_subject, run_id, options):
    load_data, prepare_data = DataFactory.get_dataset('action_'+use_object+'_'+test_subject+'_mad')

    print 'Loading data'
    train, valid, test = load_data()

    nCls = len(np.unique(train[1]))
    ydim = nCls
    options['ydim'] = ydim
    print "model options", options

    subject_id = [ s['subject'] for s in test[-1] ]
    object_id = [ s['object'] for s in test[-1] ]
    action_id = [ meta['attention_type'] for meta in test[-1] ]

    result_dir = options['result_dir']
    print 'result_dir: ', result_dir

    params = init_params(options)
    load_params(model_file, params)
    tparams = init_tparams(params)
    (use_noise, x, mask, y, f_pred, cost) = build_model(tparams, options)

    print "%d test examples" % len(test[0])

    preds_list = []
    projs_list = []
    gts_list = []

    for t in xrange(len(test[0])):
        x, mask, y = prepare_data([ [test[0][t]], [test[1][t]] ], maxlen=None)
        preds = f_pred(x, mask)
        preds_list.append(preds[0])
        gts_list.append(test[1][t])

    mat_data = {'prediction': preds_list,
                'action_id': action_id,
                'object_id': object_id,
                'subject_id': subject_id,
                'gts': gts_list}
    sio.savemat('%s/action_cls_%s_%s.mat'%(result_dir, use_object, test_subject), mat_data)

    accu = [ int(p == gt) for p, gt in zip(preds_list, gts_list) ]
    print '\nAccuracy: %.2f %%\n' % (sum(accu)/float(len(accu))*100.0)


def get_default_options():
    options = {
        'dim_proj': 128,       # word embeding dimension and LSTM number of hidden units.
        'feat_dim': 4096,      # CNN feature dimension
        'batch_size': 10,      # The batch size during training.
        'valid_batch_size': 5, # The batch size used for validation/test set.
        'maxlen': None,        # Max length of sequences

        'encoder': 'lstm',     # network name
        'optimizer': adadelta, # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).

        'max_epochs': 100,     # The maximum number of epoch to run
        'decay_c': 0.,         # Weight decay for the classifier applied to the U weights.
        'lrate': 0.0001,       # Learning rate for sgd (not used for adadelta and rmsprop)
        'use_dropout': True,   # if False slightly faster, but worst test error

        'patience': 25,        # Number of epoch to wait before early stop if no progress
        'validFreq': 20,       # Compute the validation error after this number of update.
        'saveFreq': 50,        # Save the parameters after every saveFreq updates
        'dispFreq': 50,        # Display to stdout the training progress every N updates
        'reload_model': None,  # Path to a saved model we want to start from.
    }
    return options

def action_exp(obj, run_id):
    options = get_default_options()
    options['dim_proj'] = 64
    options['batch_size'] = 10
    options['object'] = obj
    options['max_epochs'] = 100
    options['method_name'] = 'default'

    model_dir = 'model/action_model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    figure_dir = 'figure'
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    result_dir = 'result/test_results_action_run%d' % (run_id)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    options['model_dir'] = model_dir
    options['figure_dir'] = figure_dir
    options['result_dir'] = result_dir

    subjects = DataFactory.__subjects_mad

    # training
    for test_subject in subjects:
        model_file = '%s/lstm_action_%s_%s_run%d_model.npz' % (model_dir, obj, test_subject, run_id)
        print 'Start training model: %s' % (model_file)
        train_model(model_file, obj, test_subject, run_id, options)

    # testing
    for test_subject in subjects:
        model_file = '%s/lstm_action_%s_%s_run%d_model.npz' % (model_dir, obj, test_subject, run_id)
        if os.path.isfile(model_file):
            print 'Start testing model: %s' % (model_file)
            test_model(model_file, obj, test_subject, run_id, options)


if __name__ == '__main__':

    run_id = 0
    if len(sys.argv) > 1:
        run_id = int(sys.argv[1])

    print "Demo action prediction, run_id = %d" % (run_id)
    objects = DataFactory.__objects_mad
    for obj in objects:
        action_exp(obj, run_id=run_id)


# run with log
# stdbuf -oL python LSTMAction.py  2>&1 | tee train_action_ts`date +%s`.log

