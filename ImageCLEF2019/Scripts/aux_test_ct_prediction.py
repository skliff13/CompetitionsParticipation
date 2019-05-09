import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from load_data import load_data_ct_report, get_all_labels


def main():
    job_dir = 'jobs/20190420-142553-clef1.1_VGG8max_fc2-128_256_xyz_ct_report_3cl_Ep50_Tune_conv'
    job_file_path = os.path.join(job_dir, 'job.json')

    print('Reading job info from ' + job_file_path)
    with open(job_file_path, 'r') as f:
        job = json.load(f)

    (_, _), (_, y_val) = load_data_ct_report(job['data_dir'], job['image_size'], job['projections'])

    all_labels = get_all_labels()

    y = np.zeros((y_val.shape[0], 0), dtype=y_val.dtype)
    for label in job['labels']:
        j = all_labels.index(label)
        column = y_val[:, j:j + 1]
        y = np.concatenate((y, column), axis=1)

    y_val = y

    predictions_path = os.path.join(job_dir, 'val-pred.txt')
    print('Loading predictions from ' + predictions_path)
    predictions = pd.read_csv(predictions_path, header=None).get_values()

    projs = job['projections']

    lung_y = None
    preds = {}
    pred_all = None
    for p, axis in enumerate(projs):
        if lung_y is None:
            lung_y = y_val[p::len(projs), :]

        preds[axis] = predictions[p::len(projs), :]

        if pred_all is None:
            pred_all = preds[axis][..., np.newaxis]
        else:
            pred_all = np.concatenate((pred_all, preds[axis][..., np.newaxis]), axis=2)

    preds['min'] = np.min(pred_all, axis=2)
    preds['max'] = np.max(pred_all, axis=2)
    preds['mean'] = np.mean(pred_all, axis=2)

    for j, label in enumerate(job['labels']):
        print('\nLabel:', label)
        for pred_type in preds:
            scores = preds[pred_type]
            auc = roc_auc_score(lung_y[:, j], scores[:, j])
            print('%s: AUC %.3f' % (pred_type, auc))


if __name__ == '__main__':
    main()
