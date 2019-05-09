import os
import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import resize


def get_all_labels():
    return ['CTR_LeftLungAffected', 'CTR_RightLungAffected', 'CTR_LungCapacityDecrease', 'CTR_Calcification',
            'CTR_Pleurisy', 'CTR_Caverns']


def load_data_lung_binary(data_dir, image_size, use_projections):
    x_train_val = []
    y_train_val = []

    df = pd.read_csv('metadata/TrainingSet_metaData.csv')

    for id_list_file in ['train.txt', 'val.txt']:
        x = _load_subset_lung_x(data_dir, id_list_file, image_size, use_projections)
        y = _load_subset_lung_y_binary(data_dir, id_list_file, df, use_projections)

        x_train_val.append(x)
        y_train_val.append(y)

    print('train_data:', x_train_val[0].shape, y_train_val[0].shape)
    print('val_data:', x_train_val[1].shape, y_train_val[1].shape)
    print('num labels in train:', np.sum(y_train_val[0], axis=0))
    print('num labels in val:', np.sum(y_train_val[1], axis=0))

    return (x_train_val[0], y_train_val[0]), (x_train_val[1], y_train_val[1])


def load_data_ct_report(data_dir, image_size, use_projections):
    x_train_val = []
    y_train_val = []

    df = pd.read_csv('metadata/TrainingSet_metaData.csv')

    for id_list_file in ['train.txt', 'val.txt']:
        x = _load_subset_ct_x(data_dir, id_list_file, image_size, use_projections)
        y = _load_subset_ct_y_report(data_dir, id_list_file, df, use_projections)

        x_train_val.append(x)
        y_train_val.append(y)

    print('train_data:', x_train_val[0].shape, y_train_val[0].shape)
    print('val_data:', x_train_val[1].shape, y_train_val[1].shape)
    print('num labels in train:', np.sum(y_train_val[0], axis=0))
    print('num labels in val:', np.sum(y_train_val[1], axis=0))

    return (x_train_val[0], y_train_val[0]), (x_train_val[1], y_train_val[1])


def load_data_svr(data_dir, image_size, use_projections):
    x_train_val = []
    y_train_val = []

    df = pd.read_csv('metadata/TrainingSet_metaData.csv')

    for id_list_file in ['train.txt', 'val.txt']:
        x = _load_subset_ct_x(data_dir, id_list_file, image_size, use_projections)
        y = _load_subset_ct_y_svr(data_dir, id_list_file, df, use_projections)

        x_train_val.append(x)
        y_train_val.append(y)

    print('train_data:', x_train_val[0].shape, y_train_val[0].shape)
    print('val_data:', x_train_val[1].shape, y_train_val[1].shape)
    print('num labels in train:', np.sum(y_train_val[0], axis=0))
    print('num labels in val:', np.sum(y_train_val[1], axis=0))

    return (x_train_val[0], y_train_val[0]), (x_train_val[1], y_train_val[1])


def load_test_data_svr(data_dir, image_size, use_projections):
    list_file = 'metadata/TestSet_metaData.csv'

    x = _load_test_set_ct_x(data_dir, list_file, image_size, use_projections)

    print('test_data:', x.shape)

    return x


def _load_subset_lung_x(data_dir, id_list_file, image_size, use_projections):
    print('Loading data using ' + id_list_file)
    case_ids = pd.read_csv(os.path.join(data_dir, id_list_file), header=None).get_values().flatten()
    x = np.zeros((len(case_ids) * len(use_projections) * 2, image_size, image_size, 3), dtype=np.float32)
    row_counter = [0]

    for i, case_id in enumerate(case_ids):
        case_id = case_id[:-7]
        if i % 50 == 0:
            print('Cases: %i / %i' % (i, len(case_ids)))

        _read_case_lung(case_id, data_dir, image_size, row_counter, use_projections, x)

    return x


def _load_subset_ct_x(data_dir, id_list_file, image_size, use_projections):
    print('Loading data using ' + id_list_file)
    case_ids = pd.read_csv(os.path.join(data_dir, id_list_file), header=None).get_values().flatten()
    x = np.zeros((len(case_ids) * len(use_projections), image_size, image_size * 2, 3), dtype=np.float32)
    row_counter = [0]

    for i, case_id in enumerate(case_ids):
        case_id = case_id[:-7]
        if i % 50 == 0:
            print('Cases: %i / %i' % (i, len(case_ids)))

        _read_case_ct(case_id, data_dir, image_size, row_counter, use_projections, x)

    return x


def _load_test_set_ct_x(data_dir, list_file, image_size, use_projections):
    print('Loading data using ' + list_file)
    file_names = pd.read_csv(list_file)['Filename'].get_values().flatten()

    x = np.zeros((len(file_names) * len(use_projections), image_size, image_size * 2, 3), dtype=np.float32)
    row_counter = [0]

    for i, case_id in enumerate(file_names):
        case_id = case_id[:-7]
        if i % 50 == 0:
            print('Cases: %i / %i' % (i, len(file_names)))

        _read_case_ct(case_id, data_dir, image_size, row_counter, use_projections, x)

    return x


def _load_subset_lung_y_binary(data_dir, id_list_file, df, use_projections):
    print('Loading data using ' + id_list_file)
    case_ids = pd.read_csv(os.path.join(data_dir, id_list_file), header=None).get_values().flatten()

    y = []
    for case_id in case_ids:
        affected = dict()
        affected['left'] = df.loc[df['Filename'] == case_id]['CTR_LeftLungAffected'].get_values()[0]
        affected['right'] = df.loc[df['Filename'] == case_id]['CTR_RightLungAffected'].get_values()[0]

        for side in ['left', 'right']:
            for _ in use_projections:
                y.append(affected[side])

    y = np.array(y)
    return y


def _load_subset_ct_y_report(data_dir, id_list_file, df, use_projections):
    print('Loading data using ' + id_list_file)
    case_ids = pd.read_csv(os.path.join(data_dir, id_list_file), header=None).get_values().flatten()

    labels = get_all_labels()

    y = []
    for case_id in case_ids:
        row = df.loc[df['Filename'] == case_id]

        values = []
        for label in labels:
            values.append(row[label].get_values()[0])

        for _ in use_projections:
            y.append(np.array(values))

    y = np.array(y)
    return y


def _load_subset_ct_y_svr(data_dir, id_list_file, df, use_projections):
    print('Loading data using ' + id_list_file)
    case_ids = pd.read_csv(os.path.join(data_dir, id_list_file), header=None).get_values().flatten()

    mapping = {'LOW': 0, 'HIGH': 1}

    y = []
    for case_id in case_ids:
        row = df.loc[df['Filename'] == case_id]

        severity_str = row['SVR_Severity'].get_values()[0]
        values = [mapping[severity_str]]

        for _ in use_projections:
            y.append(np.array(values))

    y = np.array(y)
    return y


def _read_case_lung(case_id, data_dir, image_size, row_counter, use_projections, x):
    subdir = os.path.join(data_dir, case_id)

    for side in ['left', 'right']:
        for projection in use_projections:
            im = _read_projection_image(case_id, image_size, projection, side, subdir)

            x[row_counter[0], :, :, :] = im - 0.5
            row_counter[0] += 1


def _read_case_ct(case_id, data_dir, image_size, row_counter, use_projections, x):
    subdir = os.path.join(data_dir, case_id)

    for projection in use_projections:
        im_left = _read_projection_image(case_id, image_size, projection, 'left', subdir)
        im_right = _read_projection_image(case_id, image_size, projection, 'right', subdir)

        x[row_counter[0], :, :, :] = np.concatenate((im_left - 0.5, im_right - 0.5), axis=1)

        row_counter[0] += 1


def _read_projection_image(case_id, image_size, projection, side, subdir):
    img_file_name = '%s_%s_%s_proj_mean_max_std.png' % (case_id, side, projection)

    im = io.imread(os.path.join(subdir, img_file_name))
    im = im.astype(np.float32) / 255.

    img_shape = (image_size, image_size)
    if im.shape[0] != img_shape[0] or im.shape[1] != img_shape[1]:
        im = resize(im, img_shape)

    return im


def main():
    data_dir = '../data/clef_projections_v1.0'
    use_projections = 'xyz'
    image_size = 256

    load_data_svr(data_dir, image_size, use_projections)


if __name__ == '__main__':
    main()
