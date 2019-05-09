import numpy as np
from sklearn.metrics import roc_auc_score
from load_data import load_data_ct_report, get_all_labels

data_dir = '../data/clef_projections_v1.2'

(x_train, y_train), (x_val, y_val) = load_data_ct_report(data_dir, 256, 'z')
labels = get_all_labels()

x = np.mean(x_train + 0.5, axis=(1, 2, 3))
y = y_train[:, labels.index('CTR_Calcification')]

print(roc_auc_score(y, x))
