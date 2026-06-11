from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import RobustScaler, FunctionTransformer, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def sign_ln1p(x):
    return np.sign(x) * np.log(np.abs(x) + 1)

def logistic(x):
    return np.tanh(x/5.65)

def inverse_hyperbolic_sin(x):
    return np.arcsinh(x)*200/np.arcsinh(15)


class log_SVR:
    def __init__(self):
        self.train_data = None
        self.x_train = None
        self.y_train = None
        self.test_data = None
        self.test_id = None
        self.y_pred = None
        self.best_svr_regressor = None
        self.best_huber_regressor = None

    def read_train_csv(self, file_path: str) -> pd.DataFrame:
        self.train_data = pd.read_csv(file_path)
        self.train_id = self.train_data['Id']
        self.x_train = self.train_data.drop(columns=['target', 'Id'])
        self.y_train = self.train_data['target']

    def read_test_csv(self, file_path: str) -> pd.DataFrame:
        self.test_data = pd.read_csv(file_path)
        self.test_id = self.test_data['Id']
        self.test_data = self.test_data.drop(columns=['Id'])

    def svr_model(self):
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            #("log_transform", FunctionTransformer(func=sign_ln1p, validate=False)),
            ("scaler", RobustScaler(quantile_range=(10, 90))),
            ("svr", SVR())
        ])
        # x_features = self.x_train.columns.tolist()
        # x9_pipeline = Pipeline([
        #     ("imputer", SimpleImputer(strategy="median")),
        #     ("log_transform", FunctionTransformer(func=sign_ln1p, validate=False)),
        #     ("scaler", RobustScaler())
        # ])
        # other_x_features = [col for col in x_features if col != 'x9']
        # other_x_pipeline = Pipeline([
        #     ("imputer", SimpleImputer(strategy="median")),
        #     ("scaler", RobustScaler()),
        # ])
        # preprocessor = ColumnTransformer(
        #     transformers=[
        #         ('x9', x9_pipeline, ['x9']),
        #         ('other_x', other_x_pipeline, other_x_features)
        #     ],
        #     remainder = 'passthrough'
        # )
        # pipeline = Pipeline(steps=[('preprocessor', preprocessor),
        #                            ('svr', SVR())])
        svr_hyperparams = {
            # "svr__C": [300, 500, 700, 1000, 1200, 1500],
            # "svr__gamma": ["scale"],
            # "svr__epsilon": [15, 50, 80, 100, 120]
            "svr__C": [1500],
            "svr__gamma": ["scale"],
            "svr__epsilon": [120],
        }

        kf = KFold(n_splits=3, shuffle=True)
        svr_grid = GridSearchCV(pipeline, svr_hyperparams,
                                scoring={
                                    'r2': 'r2'
                                },
                                refit='r2',
                                cv=kf, n_jobs=-1, verbose=1)
        svr_grid.fit(self.x_train, self.y_train)
        print(f"Best parameters found: {svr_grid.best_params_}")
        print(f"Best cross-validation R2: {svr_grid.best_score_}")
        self.best_svr_regressor = svr_grid.best_estimator_

    def save_svr_predictions(self, output_path: str = "submission.csv") -> pd.DataFrame:
        output_file = Path(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_file.with_name(f"{output_file.stem}_{timestamp}{output_file.suffix}")

        predictions = self.best_svr_regressor.predict(self.test_data)
        submission = pd.DataFrame({
            "Id": self.test_id,
            "target": predictions
        })
        submission.to_csv(output_file, index=False)
        print(f"Saved submission to {output_file}")

    def huber_model(self):
        huber_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log_transform", FunctionTransformer(func=sign_ln1p, validate=False)),
            ("scaler", RobustScaler()),
            ("huber", HuberRegressor(max_iter=7000))
        ])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        huber_hyperparams = {
            "huber__epsilon": [1, 3, 5, 7, 9, 11, 13, 15],
            "huber__alpha": [0.01, 0.05, 0.1, 0.5, 0.7]
        }
        grid_search = GridSearchCV(huber_pipeline, huber_hyperparams,
                                   scoring={'r2': 'r2'},
                                   refit='r2',
                                   cv=kf, n_jobs=-1, verbose=1)
        grid_search.fit(self.x_train, self.y_train)
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation negative MSE: {grid_search.best_score_}")
        self.best_huber_regressor = grid_search.best_estimator_

    def save_huber_predictions(self, output_path: str = "submission.csv") -> pd.DataFrame:
        output_file = Path(output_path)

        output_file = output_file.with_name(f"{output_file.stem}_{output_file.suffix}")

        predictions = self.best_huber_regressor.predict(self.test_data)
        submission = pd.DataFrame({
            "Id": self.test_id,
            "target": predictions
        })
        submission.to_csv(output_file, index=False)
        print(f"Saved submission to {output_file}")

def main():
    train_data_path = "./spring2026_kaggle_linear_regression_challenge_train.csv"
    test_data_path = "./spring2026_kaggle_linear_regression_challenge_test.csv"

    svr = log_SVR()
    svr.read_train_csv(train_data_path)
    svr.read_test_csv(test_data_path)

    svr.svr_model()
    y_pred = svr.best_svr_regressor.predict(svr.x_train)
    R2 = r2_score(svr.y_train, y_pred)
    print("R2 score: ", R2)
    #svr.save_svr_predictions("SVR_R2_3fold_submission.csv")

    for i in svr.x_train.columns:
        plt.figure(figsize=(10, 6))
        plt.ylim(-500,500)
        plt.scatter(svr.x_train[i].values, y_pred)
        plt.title(f'{i} vs predicted Target')
        plt.xlabel(i)
        plt.ylabel('Target')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

if __name__ == "__main__":
    main()