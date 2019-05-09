import os
import numpy as np
import nibabel as nb
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    calc_diffs()
    calc_diffs_for_test()

    df = pd.read_csv('../data/ID_SegmDiff_Pleurisy.csv')

    x = np.array(df['SegmDiff'])
    y = np.array(df['Pleurisy'])
    print('Development data AUC', roc_auc_score(y, x))

    val_filenames = pd.read_csv('../data/clef_projections_v1.0/val.txt', header=None).get_values().flatten()
    val_filenames = set(val_filenames)

    x = []
    y = []
    for _, row in df.iterrows():
        if row['Filename'] in val_filenames:
            x.append(row['SegmDiff'])
            y.append(row['Pleurisy'])

    x = np.array(x)
    y = np.array(y)
    print('Val set AUC', roc_auc_score(y, x))


def calc_diffs():
    segm_dir1 = '/path/to/clef2019_masks/TrainingSet_Masks'
    segm_dir2 = '/path/to/clef2019_reg/TrainingSet_Masks'
    labels_file = 'metadata/TrainingSet_metaData.csv'

    df = pd.read_csv(labels_file)

    filenames = []
    segm_diff = []
    pleurisy = []
    for index, row in df.iterrows():
        print('%i / %i' % (index + 1, df.shape[0]))

        filename = row['Filename']
        filenames.append(filename)

        segm_path = os.path.join(segm_dir1, filename)
        print('Reading segmentation from "%s"' % segm_path)
        sgm1 = nb.load(segm_path).get_data()
        sgm1 = sgm1[:, ::-1, :] > sgm1[0, 0, 0]

        segm_path = os.path.join(segm_dir2, filename)
        print('Reading segmentation from "%s"' % segm_path)
        sgm2 = nb.load(segm_path).get_data()
        sgm2 = sgm2[:, ::-1, :] > sgm2[0, 0, 0]

        diff = (np.sum(sgm2) - np.sum(sgm1)) / sgm2.size

        segm_diff.append(diff)
        pleurisy.append(row['CTR_Pleurisy'])

    df1 = pd.DataFrame({'Filename': filenames, 'SegmDiff': segm_diff, 'Pleurisy': pleurisy},
                       columns=['Filename', 'SegmDiff', 'Pleurisy'])
    df1.to_csv('../data/ID_SegmDiff_Pleurisy.csv', index=None)


def calc_diffs_for_test():
    segm_dir1trn = '/path/to/clef2019_masks/TrainingSet_Masks'
    segm_dir1tst = '/path/to/clef2019_masks/TestSet_Masks'
    segm_dir2trn = '/path/to/clef2019_reg/TrainingSet_Masks'
    segm_dir2tst = '/path/to/clef2019_reg/TestSet_Masks'

    labels_file1 = 'metadata/TrainingSet_metaData.csv'
    labels_file2 = 'metadata/TestSet_metaData.csv'

    filenames = list(pd.read_csv(labels_file1)['Filename'].get_values())
    filenames += list(pd.read_csv(labels_file2)['Filename'].get_values())

    segm_diff = []
    for index, filename in enumerate(filenames):
        print('%i / %i' % (index + 1, len(filenames)))

        if 'TRN' in filename:
            segm_dir1 = segm_dir1trn
            segm_dir2 = segm_dir2trn
        else:
            segm_dir1 = segm_dir1tst
            segm_dir2 = segm_dir2tst

        segm_path = os.path.join(segm_dir1, filename)
        print('Reading segmentation from "%s"' % segm_path)
        sgm1 = nb.load(segm_path).get_data()
        sgm1 = sgm1[:, ::-1, :] > sgm1[0, 0, 0]

        segm_path = os.path.join(segm_dir2, filename)
        print('Reading segmentation from "%s"' % segm_path)
        sgm2 = nb.load(segm_path).get_data()
        sgm2 = sgm2[:, ::-1, :] > sgm2[0, 0, 0]

        diff = (np.sum(sgm2) - np.sum(sgm1)) / sgm2.size

        segm_diff.append(diff)

    df1 = pd.DataFrame({'Filename': filenames, 'SegmDiff': segm_diff}, columns=['Filename', 'SegmDiff'])
    df1.to_csv('../data/ID_SegmDiff.csv', index=None)


if __name__ == '__main__':
    main()
