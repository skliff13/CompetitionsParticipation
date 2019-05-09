import pandas as pd

from ct_predictor_2 import CtPredictor2


class CtPredictor3(CtPredictor2):
    def __init__(self):
        CtPredictor2.__init__(self)

        self.segm_diff_file = '../data/ID_SegmDiff.csv'

        self.segm_diff = {}
        df = pd.read_csv(self.segm_diff_file)
        for _, row in df.iterrows():
            self.segm_diff[row['Filename']] = row['SegmDiff']

    def predict_case(self, case_id):
        left_lung_score, right_lung_score = self._predict_lungs(case_id)

        pleurisy_score = self._predict_pleurisy(case_id)

        calcification_score = self._predict_calcification(case_id)

        caverns_score = self._predict_binary_label(case_id, self.caverns_model)
        capacity_score = self._predict_binary_label(case_id, self.capacity_model)

        return left_lung_score, right_lung_score, calcification_score, caverns_score, pleurisy_score, capacity_score

    def _predict_pleurisy(self, case_id):
        filename = case_id + '.nii.gz'
        return self.segm_diff[filename]


if __name__ == '__main__':
    predictor = CtPredictor3()
    result = predictor.predict_case('CTR_TRN_004')
    print(result)
