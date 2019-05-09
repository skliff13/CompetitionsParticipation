import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from load_data import load_data_lung_binary


def main():
    job_dir = 'jobs/20190419-211912-clef1.0_VGG8max_fc2-128_256_xyz_lung_binary_Ep120'
    job_file_path = os.path.join(job_dir, 'job.json')

    print('Reading job info from ' + job_file_path)
    with open(job_file_path, 'r') as f:
        job = json.load(f)

    (_, _), (_, y_val) = load_data_lung_binary(job['data_dir'], job['image_size'],
                                                               job['projections'])

    predictions_path = os.path.join(job_dir, 'val-pred.txt')
    print('Loading predictions from ' + predictions_path)
    predictions = pd.read_csv(predictions_path, header=None).get_values()

    projs = job['projections']

    lung_y = None
    preds = {}
    pred_all = None
    for p, axis in enumerate(projs):
        if lung_y is None:
            lung_y = y_val[p::len(projs)]

        preds[axis] = predictions[p::len(projs), 1]

        if pred_all is None:
            pred_all = preds[axis][..., np.newaxis]
        else:
            pred_all = np.concatenate((pred_all, preds[axis][..., np.newaxis]), axis=1)

    preds['min'] = np.min(pred_all, axis=1)
    preds['max'] = np.max(pred_all, axis=1)
    preds['mean'] = np.mean(pred_all, axis=1)

    for pred_type in preds:
        scores = preds[pred_type]
        auc = roc_auc_score(lung_y, scores)
        print('%s: AUC %.3f' % (pred_type, auc))


if __name__ == '__main__':
    main()
