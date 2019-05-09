import numpy as np
import pandas as pd

from ct_predictor import CtPredictor
from ct_predictor_2 import CtPredictor2
from ct_predictor_3 import CtPredictor3


def predict_test_set():
    df = pd.read_csv('metadata/TestSet_metaData.csv')

    # predictor = CtPredictor()
    # predictor = CtPredictor2()
    predictor = CtPredictor3()

    case_ids, xs = [], []
    for _, row in df.iterrows():
        case_id = row['Filename'][:-7]
        case_ids.append(case_id)

        x = predictor.predict_case(case_id)
        xs.append(x)

    xs = np.array(xs)

    d = {'a_CaseID': case_ids}
    for j in range(xs.shape[1]):
        mn = np.min(xs[:, j])
        mx = np.max(xs[:, j])
        xs[:, j] = (xs[:, j] - mn) / (mx - mn)

        d['b_feature%i' % (j + 1)] = list(xs[:, j])

    # pd.DataFrame.from_dict(d).to_csv('results/CTR_run1_multilabel.txt', header=None, index=None)
    # pd.DataFrame.from_dict(d).to_csv('results/CTR_run2_2binary.txt', header=None, index=None)
    pd.DataFrame.from_dict(d).to_csv('results/CTR_run3_pleurisy_as_SegmDiff.txt', header=None, index=None)


if __name__ == '__main__':
    predict_test_set()
