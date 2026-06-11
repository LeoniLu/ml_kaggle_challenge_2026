from lib2to3.fixes.fix_metaclass import find_metas

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

class FeatureEngineered:
    def __init__(self):
        self.train_data = None
        self.x_train = None
        self.y_train = None
        self.train_id = None
        self.test_data = None
        self.test_id = None
        self.x_train_augmented = None
        self.x_test_augmented = None

    def read_train_csv(self, file_path: str) -> pd.DataFrame:
        self.train_data = pd.read_csv(file_path)
        self.train_id = self.train_data['Id']
        self.x_train = self.train_data.drop(columns=['target', 'Id'])
        self.y_train = self.train_data['target']

    def read_test_csv(self, file_path: str) -> pd.DataFrame:
        self.test_data = pd.read_csv(file_path)
        self.test_id = self.test_data['Id']
        self.test_data = self.test_data.drop(columns=['Id'])

    def feature_engineering_cube_square(self):
        x_train_augmented = self.x_train.copy()
        x_train_augmented['x9_squared'] = x_train_augmented['x9'] * np.abs(x_train_augmented['x9'])
        x_train_augmented['x9_cubed'] = x_train_augmented['x9']**3
        x_train_augmented['target'] = self.y_train
        x_train_augmented['Id'] = self.train_id

        original_columns = self.x_train.columns.tolist()
        self.x_train_augmented = x_train_augmented[original_columns + ['x9_squared', 'x9_cubed', 'target', 'Id']]
        print(x_train_augmented)

        x_test_augmented = self.test_data.copy()
        x_test_augmented['x9_squared'] = x_test_augmented['x9'] * np.abs(x_test_augmented['x9'])
        x_test_augmented['x9_cubed'] = x_test_augmented['x9']**3
        x_test_augmented['Id'] = self.test_id

        original_columns = self.test_data.columns.tolist()
        self.x_test_augmented = x_test_augmented[original_columns + ['x9_squared', 'x9_cubed', 'Id']]
        print(x_test_augmented)

        x_train_augmented.to_csv("x9_square_cube_regression_train.csv", index=False)
        x_test_augmented.to_csv("x9_square_cube_regression_test.csv", index=False)

    def feature_engineering_augmentation(self):
        x9 = self.x_train['x9']
        poly_mask = np.abs(self.y_train) > 200
        poly_y_train = self.y_train[poly_mask]
        poly_x9 = x9[poly_mask]
        poly_pos_mask = poly_x9 >= 0
        poly_neg_mask = poly_x9 < 0
        poly_x9_pos = poly_x9[poly_x9 >= 0]
        poly_x9_neg = poly_x9[poly_x9 < 0]
        poly_x9_pos_fit = np.polyfit(poly_x9_pos, poly_y_train[poly_pos_mask], 3)
        poly_x9_neg_fit = np.polyfit(poly_x9_neg, poly_y_train[poly_neg_mask], 3)

        x9_pos_pred = np.polyval(poly_x9_pos_fit, poly_x9_pos)
        x9_neg_pred = np.polyval(poly_x9_neg_fit, poly_x9_neg)
        print(poly_x9_pos_fit, poly_x9_neg_fit)

        ## generating new constant features from x9
        self.x_train['x9_constant'] = np.nan
        const_mask = ~poly_mask
        nan_mask = self.x_train.loc[:,'x9'].isnull()
        self.x_train.loc[const_mask, 'x9_constant'] = self.x_train.loc[const_mask, 'x9']
        x9_for_sampling = self.x_train.loc[const_mask & (~nan_mask), 'x9']
        fill_mask = nan_mask | poly_mask

        random_sample_to_fill = np.random.choice(x9_for_sampling, size = fill_mask.sum())
        self.x_train.loc[fill_mask, 'x9_constant'] = random_sample_to_fill

        ## generating new column with polynomial x9
        self.x_train['x9_poly'] = np.nan
        x9_na_mask = self.x_train['x9'].isna()
        original_poly_pos_na_mask = (self.y_train > 200) & x9_na_mask
        original_poly_neg_na_mask = (self.y_train < -200) & x9_na_mask
        original_poly_pos = (self.y_train > 200) & (~x9_na_mask)
        original_poly_neg = (self.y_train < -200) & (~x9_na_mask)

        self.x_train.loc[original_poly_pos, 'x9_poly'] = np.polyval(poly_x9_pos_fit, self.x_train.loc[original_poly_pos, 'x9'])
        self.x_train.loc[original_poly_neg, 'x9_poly'] = np.polyval(poly_x9_neg_fit, self.x_train.loc[original_poly_neg, 'x9'])

        y = self.y_train.loc[original_poly_pos_na_mask | original_poly_neg_na_mask].to_numpy()
        # back_predict = []
        # if y:
        #     for i in y:
        #         if i >= 0:
        #             back_predict.append(self.find_x_from_y(i, poly_x9_pos_fit))
        #         else:
        #             back_predict.append(self.find_x_from_y(i, poly_x9_neg_fit))
        self.x_train.loc[original_poly_pos_na_mask | original_poly_neg_na_mask, 'x9_poly'] = y
        print("after filling all polynomials: ",self.x_train.loc[:,'x9_poly'].isnull().sum())

        self.x_train['x9_poly'] = self.x_train['x9_poly'].fillna(0)
        self.x_train['target'] = self.y_train
        self.x_train['Id'] = self.train_id
        self.x_train = self.x_train.drop(columns=['x9'])
        original_columns = self.x_train.columns.tolist()
        print(self.x_train)

        ### augmented test data
        self.test_data['x9_constant'] = self.test_data['x9']
        test_x9_na_mask = self.test_data['x9'].isna()
        x9_const_for_fill = np.random.choice(self.test_data.loc[~test_x9_na_mask, 'x9'], size = test_x9_na_mask.sum())
        self.test_data.loc[test_x9_na_mask, 'x9_constant'] = x9_const_for_fill

        self.test_data['x9_poly'] = np.nan
        pos_test_x9_mask = self.test_data['x9'] >= 0
        neg_test_x9_mask = self.test_data['x9'] < 0
        self.test_data.loc[pos_test_x9_mask, 'x9_poly'] = np.polyval(poly_x9_pos_fit, self.test_data.loc[pos_test_x9_mask, 'x9'])
        self.test_data.loc[neg_test_x9_mask, 'x9_poly'] = np.polyval(poly_x9_neg_fit, self.test_data.loc[neg_test_x9_mask, 'x9'])
        self.test_data['Id'] = self.test_id
        self.test_data = self.test_data.drop(columns=['x9'])
        print(self.test_data)

        self.x_train.to_csv("x9_augmented_random_0_filled_train.csv", index=False)
        self.test_data.to_csv("x9_augmented_random_0_filled_test.csv", index=False)




    def find_x_from_y(self, y, poly_coef):
        coeffs = np.copy(poly_coef)
        coeffs[-1] -= y
        all_roots = np.roots(coeffs)
        real_roots = all_roots[np.isreal(all_roots)].real
        print(real_roots)
        return real_roots[0]




def main():
    train_data_path = "./spring2026_kaggle_linear_regression_challenge_train.csv"
    test_data_path = "./spring2026_kaggle_linear_regression_challenge_test.csv"

    fe = FeatureEngineered()
    fe.read_train_csv(train_data_path)
    fe.read_test_csv(test_data_path)
    fe.feature_engineering_augmentation()


if __name__ == "__main__":
    main()