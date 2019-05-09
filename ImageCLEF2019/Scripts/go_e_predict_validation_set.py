import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ct_predictor import CtPredictor
from ct_predictor_2 import CtPredictor2
from ct_predictor_3 import CtPredictor3


def predict_val_set():
    df = pd.read_csv('metadata/TrainingSet_metaData.csv')
    val_cases = pd.read_csv('../data/clef_projections_v1.0/val.txt', header=None).get_values().flatten()
    val_cases = set(val_cases)

    # predictor = CtPredictor()
    # predictor = CtPredictor2()
    predictor = CtPredictor3()

    case_ids, xs = [], []
    for _, row in df.iterrows():
        if row['Filename'] in val_cases:
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

    pd.DataFrame.from_dict(d).to_csv('results/CTR_run3_val_predictions.txt', header=None, index=None)


def evaluate_val_set():
    df = pd.read_csv('metadata/TrainingSet_metaData.csv')
    val_cases = pd.read_csv('../data/clef_projections_v1.0/val.txt', header=None).get_values().flatten()
    val_cases = set(val_cases)

    df_pred = pd.read_csv('results/CTR_run3_val_predictions.txt', header=None)

    columns = ['CTR_LeftLungAffected', 'CTR_RightLungAffected', 'CTR_Calcification', 'CTR_Caverns', 'CTR_Pleurisy',
               'CTR_LungCapacityDecrease']

    case_ids, xs, ys = [], [], []
    for _, row in df.iterrows():
        if row['Filename'] in val_cases:
            y = list()
            for column in columns:
                y.append(row[column])
            ys.append(y)

            case_id = row['Filename'][:-7]
            case_ids.append(case_id)

            row_pred = df_pred.loc[df_pred[0] == case_id]
            x = list(row_pred.get_values()[0][1:])
            xs.append(x)

    xs = np.array(xs)
    ys = np.array(ys)

    print('\nEvaluation on the validation set:')
    lines = []
    mean_auc = 0
    for j in range(ys.shape[1]):
        auc = roc_auc_score(ys[:, j], xs[:, j])
        mean_auc += auc / ys.shape[1]
        lines.append('%s: %f' % (columns[j], auc))

    lines.append('Mean AUC: %f' % mean_auc)

    string = '\n'.join(lines)
    print(string)

    with open('results/CTR_run3_val_evaluation.txt', 'w') as f:
        f.write(string)


def main():
    predict_val_set()
    evaluate_val_set()


if __name__ == '__main__':
    main()
