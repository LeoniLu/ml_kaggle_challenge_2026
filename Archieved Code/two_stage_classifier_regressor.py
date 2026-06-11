import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime
from pathlib import Path
from sklearn.svm import SVR

class TwoStageClassifierRegressor:
    def __init__(self):
        self.train_data = None
        self.x_train = None
        self.y_train = None
        self.test_data = None
        self.test_id = None
        self.is_extreme_flag = None
        self.x_core_train = None
        self.y_core_train = None
        self.x_ext_train = None
        self.y_ext_train = None

        self.best_ext_rf_classifier = None
        self.best_core_svr_model = None
        self.best_ext_rf_regressor = None
        self.final_preds = None

    def read_train_csv(self, file_path: str) -> pd.DataFrame:
        self.train_data = pd.read_csv(file_path)
        self.x_train = self.train_data.drop(columns=['target','Id'])
        self.y_train = self.train_data['target']
        lower_bound = self.y_train.quantile(0.05)
        upper_bound = self.y_train.quantile(0.95)
        self.is_extreme_flag = (self.y_train <= lower_bound) | (self.y_train >= upper_bound)
        #self.is_extreme_flag = np.abs(self.y_train) > 500
        print(f"Number of extreme points:{self.is_extreme_flag.sum()}")
        core_mask = ~self.is_extreme_flag
        self.x_core_train = self.x_train.loc[core_mask]
        self.y_core_train = self.y_train.loc[core_mask]
        self.x_ext_train = self.x_train.loc[self.is_extreme_flag]
        self.y_ext_train = self.y_train.loc[self.is_extreme_flag]

    def read_test_csv(self, file_path: str) -> pd.DataFrame:
        self.test_data = pd.read_csv(file_path)
        self.test_id = self.test_data['Id']
        self.test_data = self.test_data.drop(columns=['Id'])

    def extreme_points_classifier(self):
        ext_clf_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(random_state=42))
        ])
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        rf_hyperparams = {
            'rf__n_estimators': [50],
            'rf__max_depth': [5],
            'rf__max_features': [0.3]
        }
        rf_grid = GridSearchCV(ext_clf_pipeline, rf_hyperparams,
                               scoring="accuracy",
                               cv=kf,
                               n_jobs=-1,verbose=1)
        rf_grid.fit(self.x_train, self.is_extreme_flag)
        print(f"RF Classifier best params: {rf_grid.best_params_}")
        print(f"RF Classifier best score: {rf_grid.best_score_}")
        self.best_ext_rf_classifier = rf_grid.best_estimator_
        print(f"RF Classifier best model: {self.best_ext_rf_classifier}")

    def core_svr_model(self):
        svr_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svr", SVR())
        ])
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        svr_hyperparams = {
            "svr__kernel": ["rbf"],
            "svr__C": [100],
            "svr__gamma": ["scale"],
            "svr__epsilon": [0.01, 0.1]
        }
        svr_grid = GridSearchCV(svr_pipeline, svr_hyperparams,
                                scoring={
                                    'neg_mean_squared_error': 'neg_mean_squared_error',
                                    'r2': 'r2'
                                },
                                refit='neg_mean_squared_error',
                                cv=kf, n_jobs=-1,verbose=1)
        svr_grid.fit(self.x_core_train, self.y_core_train)
        print(f"Core SVR best params: {svr_grid.best_params_}")
        print(f"Core SVR best score: {svr_grid.best_score_}")
        print(f"Core SVR best r2 score: {svr_grid.cv_results_['mean_test_r2'][svr_grid.best_index_]}")
        self.best_core_svr_model = svr_grid.best_estimator_
        print(f"Core SVR best model: {self.best_core_svr_model}")

    def ext_rf_model(self):
        ext_rf_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(random_state=42))
        ])
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        rf_hyperparams = {
            'rf__n_estimators': [500, 1000],
            'rf__max_depth': [20],
            'rf__max_features': [0.3]
        }
        rf_grid = GridSearchCV(ext_rf_pipeline, rf_hyperparams,
                               scoring={
                                   'neg_mean_squared_error': 'neg_mean_squared_error',
                                   'r2': 'r2'
                               },
                               refit='neg_mean_squared_error',
                               cv=kf, n_jobs=-1,verbose=1)
        rf_grid.fit(self.x_ext_train, self.y_ext_train)
        print(f"RF Regressor best params: {rf_grid.best_params_}")
        print(f"RF Regressor best score: {rf_grid.best_score_}")
        print(f"RF Regressor best r2 score: {rf_grid.cv_results_['mean_test_r2'][rf_grid.best_index_]}")
        self.best_ext_rf_regressor = rf_grid.best_estimator_
        print(f"RF Regressor best model: {self.best_ext_rf_regressor}")

    def classifier_prediction(self):
        if self.best_ext_rf_classifier is None:
            raise ValueError("best_ext_rf_classifier is empty. Train the extreme points classifier first.")
        orig_pred = self.best_ext_rf_classifier.predict(self.x_train)
        print(orig_pred.sum())
        ext_pred = self.best_ext_rf_classifier.predict(self.test_data)
        print(f"number of test set extreme points: {ext_pred.sum()}")

    def two_stage_prediction(self):
        if self.best_ext_rf_classifier is None:
            raise ValueError("best_ext_rf_classifier is empty. Train the extreme points classifier first.")
        if self.best_core_svr_model is None:
            raise ValueError("best_core_svr_model is empty. Train the core SVR model first.")
        if self.best_ext_rf_regressor is None:
            raise ValueError("best_ext_rf_regressor is empty. Train the extreme points regressor first.")


        ext_pred = self.best_ext_rf_classifier.predict(self.test_data)
        print(f"number of test set extreme points: {ext_pred.sum()}")

        preds = np.zeros(self.test_data.shape[0])
        if np.any(ext_pred):
            preds[ext_pred] = self.best_ext_rf_regressor.predict(self.test_data[ext_pred])
        if np.any(~ext_pred):
            preds[~ext_pred] = self.best_core_svr_model.predict(self.test_data[~ext_pred])
        self.final_preds = preds
        print(self.final_preds.mean(), self.final_preds.std())

    def save_predictions(self, output_path: str = "submission.csv") -> pd.DataFrame:
        output_file = Path(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_file.with_name(f"{output_file.stem}_{timestamp}{output_file.suffix}")

        submission = pd.DataFrame({
            "Id": self.test_id,
            "target": self.final_preds
        })
        #submission.to_csv(output_file, index=False)
        #self.last_submission_path = output_file
        #print(f"Saved submission to {output_file}")



