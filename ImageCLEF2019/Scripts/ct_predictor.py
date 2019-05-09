import os
import json
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
set_session(sess)
from skimage import io

from nets.vgg8_multilabel import VGG8


class CtPredictor:
    def __init__(self):
        self.lung_job_dir = 'jobs/20190419-211912-clef1.0_VGG8max_fc2-128_256_xyz_lung_binary_Ep120'
        self.ct_job_dir = 'jobs/20190420-142553-clef1.1_VGG8max_fc2-128_256_xyz_ct_report_3cl_Ep50_Tune_conv'
        self.lung_projection_dir = '../data/clef_projections_v1.0'
        self.ct_projection_dir = '../data/clef_projections_v1.1'
        self.calcif_projection_dir = '../data/clef_projections_v1.2'

        self.lung_model = self._load_lung_model()
        self.ct_model = self._load_ct_model()

    def predict_case(self, case_id):
        left_lung_score, right_lung_score = self._predict_lungs(case_id)

        capacity_score, pleurisy_score, caverns_score = self._predict_ct(case_id)

        calcification_score = self._predict_calcification(case_id)

        return left_lung_score, right_lung_score, calcification_score, caverns_score, pleurisy_score, capacity_score

    def _predict_lungs(self, case_id):
        print('Reading projections from ' + os.path.join(self.lung_projection_dir, case_id))

        lung_scores = {}
        for side in ['left', 'right']:
            projection_scores = []
            for dim in 'xyz':
                projection_filename = '%s_%s_%s_proj_mean_max_std.png' % (case_id, side, dim)
                projection_path = os.path.join(self.lung_projection_dir, case_id, projection_filename)

                im = io.imread(projection_path)
                im = im.astype(np.float32) / 255.
                x = (im - 0.5)[np.newaxis, ...]

                predictions = self.lung_model.predict(x)
                projection_scores.append(predictions[0][1])

            lung_scores[side] = max(projection_scores)

        return lung_scores['left'], lung_scores['right']

    def _predict_ct(self, case_id):
        print('Reading projections from ' + os.path.join(self.ct_projection_dir, case_id))

        projection_scores = {}
        for dim in 'xyz':
            ims = []
            for side in ['left', 'right']:
                projection_filename = '%s_%s_%s_proj_mean_max_std.png' % (case_id, side, dim)
                projection_path = os.path.join(self.ct_projection_dir, case_id, projection_filename)

                ims.append(io.imread(projection_path))

            im = np.concatenate(ims, axis=1)
            im = im.astype(np.float32) / 255.
            x = (im - 0.5)[np.newaxis, ...]

            predictions = self.ct_model.predict(x)
            projection_scores[dim] = predictions[0]

        capacity_decrease_score = projection_scores['x'][0]
        pleurisy_score = projection_scores['x'][1]
        caverns_score = projection_scores['y'][2]

        return capacity_decrease_score, pleurisy_score, caverns_score

    def _predict_calcification(self, case_id):
        dim = 'z'
        ims = []
        for side in ['left', 'right']:
            projection_filename = '%s_%s_%s_proj_mean_max_std.png' % (case_id, side, dim)
            projection_path = os.path.join(self.calcif_projection_dir, case_id, projection_filename)

            ims.append(io.imread(projection_path))

        im = np.concatenate(ims, axis=1)
        im = im.astype(np.float32) / 255.

        calcification_score = np.mean(im)
        return calcification_score

    def _load_lung_model(self):
        job_file_path = os.path.join(self.lung_job_dir, 'job.json')
        print('Reading job info from ' + job_file_path)
        with open(job_file_path, 'r') as f:
            job = json.load(f)

        input_shape = (job['image_size'], job['image_size'], 3)
        model = VGG8(weights=None, include_top=True, input_shape=input_shape, classes=2, activation='softmax',
                     fc_num=job['fc_num'], fc_size=job['fc_size'], pooling=job['pooling'])

        weigths_path = os.path.join(self.lung_job_dir, 'weights.hdf5')
        print('Loading weights from ' + weigths_path)
        model.load_weights(weigths_path)

        return model

    def _load_ct_model(self):
        job_file_path = os.path.join(self.ct_job_dir, 'job.json')
        print('Reading job info from ' + job_file_path)
        with open(job_file_path, 'r') as f:
            job = json.load(f)

        input_shape = (job['image_size'], 2 * job['image_size'], 3)
        model = VGG8(weights=None, include_top=True, input_shape=input_shape, classes=3, activation='sigmoid',
                     fc_num=job['fc_num'], fc_size=job['fc_size'], pooling=job['pooling'])

        weigths_path = os.path.join(self.ct_job_dir, 'weights.hdf5')
        print('Loading weights from ' + weigths_path)
        model.load_weights(weigths_path)

        return model


if __name__ == '__main__':
    predictor = CtPredictor()
    result = predictor.predict_case('CTR_TRN_004')
    print(result)
