import os
import json


def main():
    base_job = dict()
    base_job['name'] = 'test'
    base_job['data_dir'] = '/home/skliff13/work/crdf/CT_Descriptor/data/clef_projections_v1.0'
    base_job['mode'] = 'svr'
    base_job['projections'] = 'xyz'
    base_job['model'] = 'VGG8'
    base_job['pooling'] = 'max'
    base_job['fc_num'] = 2
    base_job['fc_size'] = 128
    base_job['pretrain'] = 'jobs/20190419-211912-clef1.0_VGG8max_fc2-128_256_xyz_lung_binary_Ep120'
    base_job['pretrain_layers'] = 'all'
    base_job['image_size'] = 256
    base_job['crop_to'] = -1
    base_job['epochs'] = 60
    base_job['optimizer'] = 'Adam'
    base_job['learning_rate'] = 1e-5
    base_job['batch_size'] = 16

    base_job['multilabel'] = False

    job = base_job.copy()
    job['name'] = make_job_name(job, base_job['data_dir'][-3:])

    print('Saving info to job.json')
    with open('job.json', 'w') as f:
        json.dump(job, f, indent=2)

    os.system('python3 train_model.py')


def make_job_name(job, dataset):
    parts = list()
    parts.append('clef%s' % dataset)
    parts.append(job['model'] + job['pooling'])
    parts.append('fc%i-%i' % (job['fc_num'], job['fc_size']))
    parts.append('%i' % job['image_size'])
    parts.append(job['projections'])
    parts.append(job['mode'])
    parts.append('Ep%i' % job['epochs'])
    if job['pretrain']:
        parts.append('Tune')
        parts.append(job['pretrain_layers'])

    return '_'.join(parts)


if __name__ == '__main__':
    main()
