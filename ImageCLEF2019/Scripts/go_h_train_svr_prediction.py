import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.feature_selection import RFE


def evaluate_val_set():
    # labels = ['md_Disability', 'md_Relapse', 'md_SymptomsOfTB', 'md_Comorbidity', 'md_Bacillary', 'md_DrugResistance',
    #           'md_HigherEducation', 'md_ExPrisoner', 'md_Alcoholic', 'md_Smoking']
    labels = ['md_DrugResistance', 'md_HigherEducation', 'md_ExPrisoner', 'md_Alcoholic']

    xs, ys = {}, {}
    for tv in ['train', 'val']:
        df1 = pd.read_csv('results/SVR_%s_ID_xyz_scores_descs.txt' % tv)
        df2 = pd.read_csv('metadata/TrainingSet_metaData.csv')

        x, y = [], []
        for _, row in df1.iterrows():
            row2 = df2.loc[df2['Filename'] == row['Filename']]
            y.append([int(row2['SVR_Severity'] == 'HIGH')])

            x_row = []
            x_row += list(row[1:3])   # XY prediction scores
            # x_row += list(row[4:])    # XYZ CNN descriptors, not used

            for label in labels:      # metadata
                x_row.append(row2[label].get_values()[0])

            x.append(x_row)

        xs[tv] = np.array(x)
        ys[tv] = np.array(y)

    print('Total %i features' % xs['train'].shape[1])

    classifiers = [LinearRegression(), RandomForestClassifier(), LogisticRegression(), LinearSVC(), LinearSVR()]

    for classifier in classifiers:
        print('')
        print(classifier.__class__.__name__)

        best_n = -1
        best_auc = -1
        for n_features in range(1, xs['train'].shape[1] + 1):
            rfe = RFE(classifier, n_features)
            rfe = rfe.fit(xs['train'], ys['train'].ravel())
            jj = rfe.support_

            classifier.fit(xs['train'][:, jj], ys['train'].ravel())
            auc = roc_auc_score(ys['val'], classifier.predict(xs['val'][:, jj]))
            # print('%i features val AUC =' % n_features, auc)

            if auc > best_auc:
                best_auc = auc
                best_n = n_features

        print('Best AUC = %f with %i features' % (best_auc, best_n))

        classifier.fit(xs['train'], ys['train'])
        auc = roc_auc_score(ys['val'], classifier.predict(xs['val']))
        print('All features val AUC =', auc)


def predict_test_set_run1():
    labels = ['md_Disability', 'md_Relapse', 'md_SymptomsOfTB', 'md_Comorbidity', 'md_Bacillary', 'md_DrugResistance',
              'md_HigherEducation', 'md_ExPrisoner', 'md_Alcoholic', 'md_Smoking']

    df1 = pd.read_csv('results/SVR_train_ID_xyz_scores_descs.txt')
    df2 = pd.read_csv('metadata/TrainingSet_metaData.csv')

    x_train, y_train = [], []
    for _, row in df1.iterrows():
        row2 = df2.loc[df2['Filename'] == row['Filename']]
        y_train.append([int(row2['SVR_Severity'] == 'HIGH')])

        x_row = []
        x_row += list(row[1:2])   # X prediction scores
        # x_row += list(row[4:])    # XYZ CNN descriptors, not used

        for label in labels:      # metadata
            x_row.append(row2[label].get_values()[0])

        x_train.append(x_row)

    df3 = pd.read_csv('results/SVR_test_ID_xyz_scores_descs.txt')
    df4 = pd.read_csv('metadata/TestSet_metaData.csv')

    x_test, case_ids = [], []
    for _, row in df3.iterrows():
        row2 = df4.loc[df4['Filename'] == row['Filename']]

        case_ids.append(row['Filename'][:-7])

        x_row = []
        x_row += list(row[1:2])  # X prediction scores
        # x_row += list(row[4:])    # XYZ CNN descriptors, not used

        for label in labels:  # metadata
            x_row.append(row2[label].get_values()[0])

        x_test.append(x_row)

    classifier = LinearRegression()

    classifier.fit(x_train, y_train)
    pred = list(classifier.predict((x_test)))

    mn = np.min(pred)
    mx = np.max(pred)
    pred = (pred - mn) / (mx - mn)

    df = pd.DataFrame.from_dict({'a_CaseId': case_ids, 'b_Prob': pred.flatten()})
    df.to_csv('results/SRV_run1_linear.txt', header=None, index=None)


def predict_test_set_run2():
    labels = ['md_DrugResistance', 'md_HigherEducation', 'md_ExPrisoner', 'md_Alcoholic']

    df1 = pd.read_csv('results/SVR_train_ID_xyz_scores_descs.txt')
    df2 = pd.read_csv('metadata/TrainingSet_metaData.csv')

    x_train, y_train = [], []
    for _, row in df1.iterrows():
        row2 = df2.loc[df2['Filename'] == row['Filename']]
        y_train.append([int(row2['SVR_Severity'] == 'HIGH')])

        x_row = []
        x_row += list(row[1:3])   # XY prediction scores
        # x_row += list(row[4:])    # XYZ CNN descriptors, not used

        for label in labels:      # metadata
            x_row.append(row2[label].get_values()[0])

        x_train.append(x_row)

    df3 = pd.read_csv('results/SVR_test_ID_xyz_scores_descs.txt')
    df4 = pd.read_csv('metadata/TestSet_metaData.csv')

    x_test, case_ids = [], []
    for _, row in df3.iterrows():
        row2 = df4.loc[df4['Filename'] == row['Filename']]

        case_ids.append(row['Filename'][:-7])

        x_row = []
        x_row += list(row[1:3])  # X prediction scores
        # x_row += list(row[4:])    # XYZ CNN descriptors, not used

        for label in labels:  # metadata
            x_row.append(row2[label].get_values()[0])

        x_test.append(x_row)

    classifier = LinearRegression()

    classifier.fit(x_train, y_train)
    pred = list(classifier.predict((x_test)))

    mn = np.min(pred)
    mx = np.max(pred)
    pred = (pred - mn) / (mx - mn)

    df = pd.DataFrame.from_dict({'a_CaseId': case_ids, 'b_Prob': pred.flatten()})
    df.to_csv('results/SRV_run2_less_features.txt', header=None, index=None)


if __name__ == '__main__':
    evaluate_val_set()
    # predict_test_set_run1()
    predict_test_set_run2()
