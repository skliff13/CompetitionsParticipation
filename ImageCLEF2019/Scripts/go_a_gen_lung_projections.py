import os
import pathlib
import json
import numpy as np
import nibabel as nb
import pandas as pd
from skimage import io
from skimage.morphology import binary_erosion
from skimage.transform import resize, rotate
from glob import glob


def make_structure_element(r, z2xy):
    rz = int(round(r / z2xy))
    x = np.arange(-r, r + 1, 1) / r
    y = np.arange(-r, r + 1, 1) / r
    z = np.arange(-rz, rz + 1, 1) / rz

    xx, yy, zz = np.meshgrid(x, y, z)
    el = (xx**2 + yy**2 + zz**2) <= 1

    return el


def map_lesion_labels(lbl):
    print('Performing label mapping')
    mapping = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6}

    lbl2 = lbl.copy()
    for label in mapping:
        lbl2[lbl == label] = mapping[label]

    return lbl2


def read_images(initial_path, segm_path):
    print('Reading original image from "%s"' % initial_path)
    nii = nb.load(initial_path)
    z2xy = abs(nii.affine[2, 2] / nii.affine[0, 0])
    im3 = nii.get_data() + 1024.
    im3 = im3[:, ::-1, :]
    im3 = np.swapaxes(im3, 0, 1)

    print('Reading segmentation from "%s"' % segm_path)
    sgm = nb.load(segm_path).get_data()
    sgm = sgm[:, ::-1, :] > sgm[0, 0, 0]
    sgm = np.swapaxes(sgm, 0, 1)

    return im3, sgm, z2xy


def get_cropping(bw):
    i_proj = np.max(bw, axis=1)
    non_zero = np.argwhere(i_proj).flatten()
    i0 = non_zero[0]
    i1 = non_zero[-1] + 1

    j_proj = np.max(bw, axis=0)
    non_zero = np.argwhere(j_proj).flatten()
    j0 = non_zero[0]
    j1 = non_zero[-1] + 1

    return i0, i1, j0, j1


def calc_and_save_projections(axes, im3_part, sgm_part, out_subdir, id, side, config):
    for dim, axis in enumerate(axes):
        proj_mean = np.mean(im3_part, axis=dim)
        proj_mean /= proj_mean.max()
        proj_max = np.max(im3_part, axis=dim)
        proj_max /= 1500
        proj_std = np.std(im3_part, axis=dim)
        proj_std /= proj_std.max()

        rgb = np.zeros((proj_max.shape[0], proj_max.shape[1], 3))
        rgb[:, :, 0] = proj_mean
        rgb[:, :, 1] = proj_max
        rgb[:, :, 2] = proj_std

        proj_segm = np.max(sgm_part, axis=dim)
        i0, i1, j0, j1 = get_cropping(proj_segm)

        rgb = rgb[i0:i1, j0:j1, :]
        rgb = resize(rgb, (config['out_size'], config['out_size']))
        rgb[rgb > 1] = 1

        if axis != 'z':
            rgb = rotate(rgb, 90)

        out_path = os.path.join(out_subdir, '%s_%s_%s_proj_mean_max_std.png' % (id, side, axis))
        io.imsave(out_path, rgb)


def process_ct(initial_path, segm_path, row, out_dir, config):
    case_id = row['Filename'][:-7]

    out_subdir = os.path.join(out_dir, case_id)
    pathlib.Path(out_subdir).mkdir(parents=True, exist_ok=True)

    im3, sgm, z2xy = read_images(initial_path, segm_path)
    im3[im3 < (1024 + config['HU_threshold'])] = 0

    if config['erosion_radius'] > 0:
        element = make_structure_element(config['erosion_radius'], z2xy)
        sgm = binary_erosion(sgm, element)

    im3[sgm == 0] = 0

    print('Saving results to ' + out_subdir)
    lung_ranges = {'left': (0, 256), 'right': (256, 512)}
    axes = ['y', 'x', 'z']
    for side in lung_ranges:
        x_range = lung_ranges[side]

        im3_part = im3[:, x_range[0]:x_range[1], :]
        sgm_part = sgm[:, x_range[0]:x_range[1], :]

        calc_and_save_projections(axes, im3_part, sgm_part, out_subdir, case_id, side, config)


def main():
    initial_dir = '/hdd_purple/clef2019/TrainingSet'
    segm_dir = '/hdd_purple/clef2019_reg/TrainingSet_Masks'
    labels_file = 'metadata/TrainingSet_metaData.csv'
    out_dir = '../data/clef_projections_v1.1'

    config = {'erosion_radius': 10, 'out_size': 256, 'HU_threshold': -1500, 'segm_dir': segm_dir}
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    df = pd.read_csv(labels_file)

    for index, row in df.iterrows():
        print('%i / %i' % (index + 1, df.shape[0]))

        case_id = row['Filename']

        initial_path = os.path.join(initial_dir, case_id)
        segm_path = os.path.join(segm_dir, case_id)

        if os.path.isfile(initial_path) and os.path.isfile(segm_path):
            process_ct(initial_path, segm_path, row, out_dir, config)


if __name__ == '__main__':
    main()
