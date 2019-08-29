import pandas as pd


def svr_table():
    df1 = pd.read_csv('TrainingSet_metaData.csv')
    df2 = pd.read_csv('TestSet_metaData.csv')

    fields = ['md_Disability', 'md_Relapse', 'md_SymptomsOfTB', 'md_Comorbidity', 'md_Bacillary',
              'md_DrugResistance', 'md_HigherEducation', 'md_ExPrisoner', 'md_Alcoholic', 'md_Smoking']

    for field in fields:
        s = field[3:] + ' & '
        values = df1[field]
        values = list(map(int, map(bool, values)))
        s += str(sum(values)) + ' & '
        values = df2[field]
        values = list(map(int, map(bool, values)))
        s += str(sum(values)) + ' \\\\'

        print(s)


def ctr_table():
    df1 = pd.read_csv('TrainingSet_metaData.csv')

    fields = ['CTR_LeftLungAffected', 'CTR_RightLungAffected', 'CTR_LungCapacityDecrease', 'CTR_Calcification',
              'CTR_Pleurisy', 'CTR_Caverns']

    for field in fields:
        s = field[4:] + ' & '
        s += str(sum(df1[field])) + ' \\\\'

        print(s)


svr_table()
print('\n')
ctr_table()
