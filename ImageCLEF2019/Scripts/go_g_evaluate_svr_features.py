import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
set_session(sess)
from keras.models import Model

from nets.vgg8_multilabel import VGG8
from load_data import load_data_svr, load_test_data_svr


def evaluate_training_data():
    job, layer_model, model = load_models()

    data = load_data_svr(job['data_dir'], job['image_size'], job['projections'])

    dfiles = ['../data/clef_projections_v1.0/train.txt', '../data/clef_projections_v1.0/val.txt']
    out_files = ['results/SVR_train_ID_xyz_scores_descs.txt', 'results/SVR_val_ID_xyz_scores_descs.txt']

    for i, (x, _) in enumerate(data):
        scores = model.predict(x)[:, 1]
        descs = layer_model.predict(x)

        scores = scores.reshape(x.shape[0] // 3, 3)
        descs = descs.reshape(x.shape[0] // 3, descs.shape[1] * 3)

        filenames = list(pd.read_csv(dfiles[i], header=None).get_values().flatten())

        data = np.concatenate((scores, descs), axis=1)
        df = pd.DataFrame(data=data)
        df['Filename'] = pd.Series(filenames, index=df.index)

        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

        print('Saving table to ' + out_files[i])
        df.to_csv(out_files[i], index=0)


def evaluate_test_data():
    job, layer_model, model = load_models()

    x = load_test_data_svr(job['data_dir'], job['image_size'], job['projections'])

    out_file = 'results/SVR_test_ID_xyz_scores_descs.txt'

    scores = model.predict(x)[:, 1]
    descs = layer_model.predict(x)

    scores = scores.reshape(x.shape[0] // 3, 3)
    descs = descs.reshape(x.shape[0] // 3, descs.shape[1] * 3)

    list_file = 'metadata/TestSet_metaData.csv'
    filenames = pd.read_csv(list_file)['Filename'].get_values().flatten()

    data = np.concatenate((scores, descs), axis=1)
    df = pd.DataFrame(data=data)
    df['Filename'] = pd.Series(filenames, index=df.index)

    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    print('Saving table to ' + out_file)
    df.to_csv(out_file, index=0)


def load_models():
    job_dir = 'jobs/20190425-111553-clef1.0_VGG8max_fc2-128_256_xyz_svr_Ep60_Tune_all'
    desc_layer = 'fc2'

    job_file_path = os.path.join(job_dir, 'job.json')
    print('Reading job info from ' + job_file_path)
    with open(job_file_path, 'r') as f:
        job = json.load(f)

    input_shape = (job['image_size'], 2 * job['image_size'], 3)
    model = VGG8(weights=None, include_top=True, input_shape=input_shape, classes=2, activation='softmax',
                 fc_num=job['fc_num'], fc_size=job['fc_size'], pooling=job['pooling'])

    layer_model = Model(inputs=model.input, outputs=model.get_layer(desc_layer).output)

    weigths_path = os.path.join(job_dir, 'weights.hdf5')
    print('Loading weights from ' + weigths_path)
    model.load_weights(weigths_path)

    return job, layer_model, model


def main():
    evaluate_training_data()
    evaluate_test_data()


if __name__ == '__main__':
    main()
