import os
import json
import datetime
import pathlib
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
set_session(sess)
import keras
from sklearn.metrics import roc_auc_score

from nets.vgg8_multilabel import VGG8
from data_gen import ModifiedDataGenerator
from load_data import load_data_lung_binary, load_data_ct_report, load_data_svr, get_all_labels
from aux_resave_plots import save_logs_as_png


def train_model(job_file_path):
    print('Reading job info from ' + job_file_path)
    with open(job_file_path, 'r') as f:
        job = json.load(f)

    timestamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
    job_dir = 'jobs/%s-%s' % (timestamp, job['name'])
    pathlib.Path(job_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(job_file_path, os.path.join(job_dir, 'job.json'))

    print('\n### Running training job ' + job_dir + '\n')

    multilabel, x_train, x_val, y_train, y_val = load_job_data(job)

    model_type = parse_model(job['model'])
    activation = 'sigmoid' if multilabel else 'softmax'

    num_classes = y_val.shape[1]
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    model = model_type(weights=None, include_top=True, input_shape=input_shape, classes=num_classes,
                       activation=activation, fc_num=job['fc_num'], fc_size=job['fc_size'], pooling=job['pooling'])

    if job['pretrain'] and job['pretrain'] != 'imagenet':
        model = import_weights(model, job)

    optimizer = create_optimizer(job['optimizer'], job['learning_rate'])
    loss = 'binary_crossentropy' if multilabel else 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    train_gen = ModifiedDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, rescale=1.,
                                      zoom_range=0.2, fill_mode='nearest', cval=0, crop_to=job['crop_to'])

    val_gen = ModifiedDataGenerator(rescale=1., crop_to=job['crop_to'])

    epochs = job['epochs']
    callbacks = organize_callbacks(job, job_dir, optimizer, epochs)

    batch_size = job['batch_size']
    model.fit_generator(train_gen.flow(x_train, y_train, batch_size),
                        steps_per_epoch=(x_train.shape[0] + batch_size - 1) // batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=val_gen.flow(x_val, y_val),
                        validation_steps=(x_val.shape[0] + batch_size - 1) // batch_size)

    weights_path = os.path.join(job_dir, 'weights.hdf5')
    print('Saving weights to ' + weights_path)
    model.save_weights(weights_path)

    if job['crop_to'] > 0:
        x_val = val_gen.crop_data_bacth(x_val)

    predictions = model.predict(x_val, batch_size=batch_size)

    predictions_path = os.path.join(job_dir, 'val-pred.txt')
    print('Saving predictions to ' + predictions_path)
    pd.DataFrame(data=predictions).to_csv(predictions_path, index=None, header=None)

    evaluate_aucs(num_classes, predictions, y_val, job_dir, multilabel, job)

    save_logs_as_png(job_dir)

    print('\n### Finished\n\n')


def import_weights(model, job):
    job_dir = job['pretrain']

    job_file_path = os.path.join(job_dir, 'job.json')

    print('Reading job info from ' + job_file_path)
    with open(job_file_path, 'r') as f:
        job0 = json.load(f)

    multilabel = False
    model_type = parse_model(job0['model'])
    activation = 'sigmoid' if multilabel else 'softmax'

    sz = job0['image_size']
    num_classes = 2
    model0 = model_type(weights=None, include_top=True, input_shape=(sz, sz, 3), classes=num_classes,
                        activation=activation, fc_num=job['fc_num'], fc_size=job['fc_size'], pooling=job['pooling'])

    weights_path = os.path.join(job_dir, 'weights.hdf5')
    print('Loading weights from ' + weights_path)
    model0.load_weights(weights_path)

    for i, layer0 in enumerate(model0.layers):
        go = 'conv' in layer0.name
        if job['pretrain_layers'] == 'all':
            go = go or ('fc' in layer0.name)

        if go:
            print('Copying weights from "' + layer0.name + '" to "' + model.layers[i].name + '"')
            model.layers[i].set_weights(layer0.get_weights())

    return model


def load_job_data(job):
    if job['mode'] == 'lung_binary':
        (x_train, y_train), (x_val, y_val) = load_data_lung_binary(job['data_dir'], job['image_size'],
                                                                   job['projections'])
        multilabel = False
        y_train = keras.utils.to_categorical(y_train, 2)
        y_val = keras.utils.to_categorical(y_val, 2)

        return multilabel, x_train, x_val, y_train, y_val

    if job['mode'] == 'ct_report':
        (x_train, y_train), (x_val, y_val) = load_data_ct_report(job['data_dir'], job['image_size'], job['projections'])

        all_labels = get_all_labels()
        multilabel = len(job['labels']) > 1

        y0s = [y_train, y_val]
        ys = []
        for y0 in y0s:
            y = np.zeros((y0.shape[0], 0), dtype=y0.dtype)
            for label in job['labels']:
                j = all_labels.index(label)
                column = y0[:, j:j + 1]
                y = np.concatenate((y, column), axis=1)
            ys.append(y)

        y_train, y_val = ys[0], ys[1]

        if not multilabel:
            y_train = keras.utils.to_categorical(y_train, 2)
            y_val = keras.utils.to_categorical(y_val, 2)

        return multilabel, x_train, x_val, y_train, y_val

    if job['mode'] == 'svr':
        (x_train, y_train), (x_val, y_val) = load_data_svr(job['data_dir'], job['image_size'], job['projections'])

        multilabel = False
        y_train = keras.utils.to_categorical(y_train, 2)
        y_val = keras.utils.to_categorical(y_val, 2)

        return multilabel, x_train, x_val, y_train, y_val

    print('Unknown mode "' + job['mode'] + '"')
    exit(1)


def parse_model(model_type):
    if model_type == 'VGG8':
        return VGG8
    else:
        print('Unknown net model: ' + model_type)
        return None


def create_optimizer(optimizer, learning_rate):
    if optimizer == 'RMSprop':
        return keras.optimizers.rmsprop(lr=learning_rate, decay=1.e-6)
    elif optimizer == 'SGD':
        return keras.optimizers.sgd(lr=learning_rate, decay=1.e-6, nesterov=True, momentum=0.9)
    elif optimizer == 'Adam':
        return keras.optimizers.adam(lr=learning_rate, decay=1.e-6)
    else:
        print('Unknown optimizer: ' + optimizer)
        return None


def organize_callbacks(job, job_dir, optimizer, epochs):
    tensor_board = keras.callbacks.TensorBoard(log_dir=os.path.join(job_dir, 'logdir'),
                                               histogram_freq=0, write_graph=True, write_images=True)
    callbacks = [tensor_board]
    learning_rate = job['learning_rate']
    if optimizer.__class__.__name__ == 'SGD':
        def schedule(epoch):
            if epoch < epochs // 3:
                return learning_rate
            if epoch < 2 * epochs // 3:
                return learning_rate * 0.1
            return learning_rate * 0.01

        callbacks.append(keras.callbacks.LearningRateScheduler(schedule=schedule))
    return callbacks


def evaluate_aucs(num_classes, predictions, y_val, job_dir, multilabel, job):
    if multilabel:
        aucs = []
        lines = ''
        for j in range(num_classes):
            roc_auc = roc_auc_score(y_val[:, j].ravel(), predictions[:, j].ravel())
            line = 'Class "%s" AUC: %.03f' % (job['labels'][j], roc_auc)
            print(line)
            lines += line + '\n'
            aucs.append(roc_auc)
        mean_auc = sum(aucs) / num_classes

        out_file_name = 'mean_auc_%.3f.txt' % mean_auc
        print('Saving AUCs to ' + out_file_name)
        with open(os.path.join(job_dir, out_file_name), 'w') as f:
            f.write(lines)
    else:
        j = 1
        roc_auc = roc_auc_score(y_val[:, j].ravel(), predictions[:, j].ravel())
        print('AUC: %.03f' % roc_auc)
        out_file_name = 'auc_%.3f.txt' % roc_auc
        print('Creating empty file ' + out_file_name)
        with open(os.path.join(job_dir, out_file_name), 'w') as f:
            f.writelines([''])


if __name__ == '__main__':
    train_model('job.json')
