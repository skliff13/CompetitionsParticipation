import os
import pandas as pd


def main():
    data_dir = '../data/clef_projections_v1.0'

    df = pd.read_csv('metadata/TrainingSet_metaData.csv')
    case_ids = list(df['Filename'].get_values())

    each_val = 4
    train_ids = []
    for i in range(each_val - 1):
        train_ids += case_ids[i::each_val]
    train_ids.sort()

    val_ids = case_ids[each_val - 1::each_val]
    val_ids.sort()

    print('Train cases: %i, val cases: %i' % (len(train_ids), len(val_ids)))
    n_first = 10
    print('First %i train and val IDs:' % n_first)
    for i in range(n_first):
        print(train_ids[i], '\t', val_ids[i])

    df = pd.DataFrame.from_dict({'ids': train_ids})
    df.to_csv(os.path.join(data_dir, 'train.txt'), header=None, index=None)

    df = pd.DataFrame.from_dict({'ids': val_ids})
    df.to_csv(os.path.join(data_dir, 'val.txt'), header=None, index=None)


if __name__ == '__main__':
    main()
