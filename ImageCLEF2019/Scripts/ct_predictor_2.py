import os
import json
import numpy as np
from skimage import io

from ct_predictor import CtPredictor
from nets.vgg8_multilabel import VGG8


class CtPredictor2(CtPredictor):
    def __init__(self):
        CtPredictor.__init__(self)

        self.capacity_job_dir = 'jobs/20190424-120814-clef1.1_VGG8max_fc2-128_256_xyz_ct_report_cap_Ep50_Tune_conv'
        self.caverns_job_dir = 'jobs/20190424-113032-clef1.1_VGG8max_fc2-128_256_xyz_ct_report_cav_Ep120_Tune_conv'

        self.capacity_model = self._load_binary_model(self.capacity_job_dir)
        self.caverns_model = self._load_binary_model(self.caverns_job_dir)

    def predict_case(self, case_id):
        left_lung_score, right_lung_score = self._predict_lungs(case_id)

        _, pleurisy_score, _ = self._predict_ct(case_id)

        calcification_score = self._predict_calcification(case_id)

        caverns_score = self._predict_binary_label(case_id, self.caverns_model)
        capacity_score = self._predict_binary_label(case_id, self.capacity_model)

        return left_lung_score, right_lung_score, calcification_score, caverns_score, pleurisy_score, capacity_score

    def _predict_binary_label(self, case_id, model):
        print('Reading projections from ' + os.path.join(self.ct_projection_dir, case_id))

        projection_scores = []
        for dim in 'xyz':
            ims = []
            for side in ['left', 'right']:
                projection_filename = '%s_%s_%s_proj_mean_max_std.png' % (case_id, side, dim)
                projection_path = os.path.join(self.ct_projection_dir, case_id, projection_filename)

                ims.append(io.imread(projection_path))

            im = np.concatenate(ims, axis=1)
            im = im.astype(np.float32) / 255.
            x = (im - 0.5)[np.newaxis, ...]

            predictions = model.predict(x)
            projection_scores.append(predictions[0][1])

        score = sum(projection_scores) / len(projection_scores)

        return score

    @staticmethod
    def _load_binary_model(job_dir):
        job_file_path = os.path.join(job_dir, 'job.json')
        print('Reading job info from ' + job_file_path)
        with open(job_file_path, 'r') as f:
            job = json.load(f)

        input_shape = (job['image_size'], 2 * job['image_size'], 3)
        model = VGG8(weights=None, include_top=True, input_shape=input_shape, classes=2, activation='softmax',
                     fc_num=job['fc_num'], fc_size=job['fc_size'], pooling=job['pooling'])

        weigths_path = os.path.join(job_dir, 'weights.hdf5')
        print('Loading weights from ' + weigths_path)
        model.load_weights(weigths_path)

        return model


if __name__ == '__main__':
    predictor = CtPredictor2()
    result = predictor.predict_case('CTR_TRN_004')
    print(result)
