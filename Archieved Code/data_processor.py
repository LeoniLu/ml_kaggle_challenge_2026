from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer



class OutlierFilteredKFold:
    def __init__(self, n_splits: int = 10, shuffle: bool = True, random_state: int | None = 42,
                 std_multiplier: float = 1.0, max_target_deviation: float | None = None) -> None:
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.std_multiplier = std_multiplier
        self.max_target_deviation = max_target_deviation

    def split(self, X, y=None, groups=None):
        if y is None:
            raise ValueError("OutlierFilteredKFold requires y to filter training rows.")

        y_series = pd.Series(y)
        for train_index, validation_index in self.kfold.split(X, y, groups):
            y_train = y_series.iloc[train_index]
            target_mean = y_train.mean()
            if self.max_target_deviation is None:
                target_std = y_train.std()
                lower_bound = target_mean - self.std_multiplier * target_std
                upper_bound = target_mean + self.std_multiplier * target_std
            else:
                lower_bound = target_mean - self.max_target_deviation
                upper_bound = target_mean + self.max_target_deviation

            target_mask = y_train.between(lower_bound, upper_bound)
            filtered_train_index = train_index[target_mask.to_numpy()]
            yield filtered_train_index, validation_index

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.kfold.get_n_splits(X, y, groups)


class DataProcessor:
    def __init__(self) -> None:
        self.train_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None
        self.test_id: pd.Series | None = None
        self.x_train: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.x_train_exclude_outlyer: pd.DataFrame | None = None
        self.y_train_exclude_outlyer: pd.Series | None = None
        self.best_rf_model = None
        self.best_xgb_model = None
        self.best_svr_model = None
        self.best_rbf_svr_model = None
        self.best_rbf_elasticnet_svr_model = None
        self.best_linear_svr_model = None
        self.best_poly_svr_model = None
        self.best_svr_kernel = None
        self.svr_scores = {}
        self.last_submission_path: Path | None = None
        self.best_lr_model = None
        self.best_ridge_model = None

    def read_train_csv(self, file_path: str) -> pd.DataFrame:
        self.train_data = pd.read_csv(file_path)
        self.x_train = self.train_data.drop(columns=['target','Id'])
        self.y_train = self.train_data['target']
        target_mean = self.y_train.mean()
        target_std = self.y_train.std()
        target_mask = self.y_train.between(
            target_mean - 500,
            target_mean + 500
        )
        self.x_train_exclude_outlyer = self.x_train.loc[target_mask]
        self.y_train_exclude_outlyer = self.y_train.loc[target_mask]
        #print(self.x_train)
        #print(self.y_train)

    def read_test_csv(self, file_path: str) -> pd.DataFrame:
        self.test_data = pd.read_csv(file_path)
        self.test_id = self.test_data['Id']
        self.test_data = self.test_data.drop(columns=['Id'])
        #print(self.test_data)

    def linear_regression_model(self):
        lr_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("lr", LinearRegression())
        ])
        log_transformed_model = TransformedTargetRegressor(
            regressor=lr_pipeline,
            transformer=PowerTransformer(method='yeo-johnson')
        )
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        lr_hyperparams = {
            'regressor__lr__fit_intercept': [True],
            'regressor__lr__copy_X': [True],
        }
        lr_grid = GridSearchCV(log_transformed_model, lr_hyperparams,
                               scoring={
                                   'neg_mean_squared_error': 'neg_mean_squared_error',
                                   'r2': 'r2'
                               },
                               refit='neg_mean_squared_error',
                               cv=kf, n_jobs=-1, verbose=1)
        lr_grid.fit(self.x_train, self.y_train)
        print(lr_grid.best_params_)
        print(f"best score: {lr_grid.best_score_}")
        print(f"best r2 score: {lr_grid.cv_results_['mean_test_r2'][lr_grid.best_index_]}")
        self.best_lr_model = lr_grid.best_estimator_
        print(self.best_lr_model)
        return self.best_lr_model

    def save_linear_regression_predictions(self, output_path: str = "submission.csv") -> pd.DataFrame:
        if self.best_lr_model is None:
            raise ValueError("Linear regression model has not been trained. Please call train_linear_regression_model first.")

        output_file = Path(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_file.with_name(f"{output_file.stem}_{timestamp}{output_file.suffix}")

        predictions = self.best_lr_model.predict(self.test_data)
        submission = pd.DataFrame({
            "Id": self.test_id,
            "target": predictions
        })
        submission.to_csv(output_file, index=False)
        self.last_submission_path = output_file
        print(f"Saved submission to {output_file}")
        return submission

    def ridge_regression_model(self):
        ridge_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge())
        ])
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        ridge_hyperparams = {
            "ridge__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "ridge__fit_intercept": [True],
            "ridge__solver": ["auto"]
        }
        ridge_grid = GridSearchCV(ridge_pipeline, ridge_hyperparams,
                                  scoring={
                                      "neg_mean_squared_error": "neg_mean_squared_error",
                                      "r2": "r2"
                                  },
                                  refit="neg_mean_squared_error",
                                  cv=kf, n_jobs=-1, verbose=1)
        ridge_grid.fit(self.x_train, self.y_train)
        print(ridge_grid.best_params_)
        print(f"best score: {ridge_grid.best_score_}")
        print(f"best r2 score: {ridge_grid.cv_results_['mean_test_r2'][ridge_grid.best_index_]}")
        self.best_ridge_model = ridge_grid.best_estimator_
        print(self.best_ridge_model)
        return self.best_ridge_model

    def save_ridge_regression_predictions(self, output_path: str = "submission.csv") -> pd.DataFrame:
        if self.best_ridge_model is None:
            raise ValueError("Ridge regression model has not been trained. Please call ridge_regression_model first.")
        if self.test_data is None:
            raise ValueError("test_data is empty. Read the test CSV first.")
        if self.test_id is None:
            raise ValueError("test_id is empty. Read the test CSV first.")

        output_file = Path(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_file.with_name(f"{output_file.stem}_{timestamp}{output_file.suffix}")

        predictions = self.best_ridge_model.predict(self.test_data)
        submission = pd.DataFrame({
            "Id": self.test_id,
            "target": predictions
        })
        submission.to_csv(output_file, index=False)
        self.last_submission_path = output_file
        print(f"Saved submission to {output_file}")
        return submission

    def random_forest_model(self):
        rf_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(random_state=42))
        ])
        kf = OutlierFilteredKFold(n_splits=10, shuffle=True, random_state=42)
        rf_hyperparams = {
            'rf__n_estimators': [700, 1000, 1500],
            'rf__max_depth': [10, 15, 20],
            'rf__min_samples_split': [5, 10],
            'rf__min_samples_leaf': [4],
            'rf__max_features': ['sqrt', 'log2', 0.3]
        }
        rf_grid = GridSearchCV(rf_pipeline, rf_hyperparams,
                               scoring={
                                   'neg_mean_squared_error': 'neg_mean_squared_error',
                                   'r2': 'r2'
                               },
                               refit='neg_mean_squared_error',
                               cv=kf, n_jobs=-1,verbose=2)
        rf_grid.fit(self.x_train, self.y_train)
        print(rf_grid.best_params_)
        print(f"best score: {rf_grid.best_score_}")
        print(f"best r2 score: {rf_grid.cv_results_['mean_test_r2'][rf_grid.best_index_]}")
        self.best_rf_model = rf_grid.best_estimator_
        print(self.best_rf_model)
        return self.best_rf_model

    def save_random_forest_predictions(self, output_path: str = "submission.csv") -> pd.DataFrame:
        if self.best_rf_model is None:
            raise ValueError("best_rf_model is empty. Train the random forest model first.")
        if self.test_data is None:
            raise ValueError("test_data is empty. Read the test CSV first.")
        if self.test_id is None:
            raise ValueError("test_id is empty. Read the test CSV first.")

        output_file = Path(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_file.with_name(f"{output_file.stem}_{timestamp}{output_file.suffix}")

        predictions = self.best_rf_model.predict(self.test_data)
        submission = pd.DataFrame({
            "Id": self.test_id,
            "target": predictions
        })
        submission.to_csv(output_file, index=False)
        self.last_submission_path = output_file
        print(f"Saved submission to {output_file}")
        return submission

    def xgboost_model(self):
        xgb_pipeline = Pipeline([
            ("xgb", XGBRegressor(
                objective="reg:squarederror",
                device = "cuda",
                random_state=42,
                tree_method="hist"
            ))
        ])
        kf = OutlierFilteredKFold(n_splits=10, shuffle=True, random_state=42)
        xgb_hyperparams = {
            "xgb__n_estimators": [1500, 2500, 5000],
            "xgb__max_depth": [3, 5],
            "xgb__learning_rate": [0.03],
            "xgb__subsample": [1.0],
            "xgb__colsample_bytree": [0.8],
            "xgb__reg_lambda": [10]
        }
        xgb_grid = GridSearchCV(xgb_pipeline, xgb_hyperparams,
                                scoring='neg_mean_squared_error',
                                cv=kf, n_jobs=1, verbose=1)
        xgb_grid.fit(self.x_train, self.y_train)
        print(xgb_grid.best_params_)
        print(f"best score: {xgb_grid.best_score_}")
        self.best_xgb_model = xgb_grid.best_estimator_
        print(self.best_xgb_model)
        return self.best_xgb_model

    def save_xgboost_predictions(self, output_path: str = "submission.csv") -> pd.DataFrame:
        if self.best_xgb_model is None:
            raise ValueError("best_xgb_model is empty. Train the XGBoost model first.")
        if self.test_data is None:
            raise ValueError("test_data is empty. Read the test CSV first.")
        if self.test_id is None:
            raise ValueError("test_id is empty. Read the test CSV first.")

        output_file = Path(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_file.with_name(f"{output_file.stem}_{timestamp}{output_file.suffix}")

        predictions = self.best_xgb_model.predict(self.test_data)
        submission = pd.DataFrame({
            "Id": self.test_id,
            "target": predictions
        })
        submission.to_csv(output_file, index=False)
        self.last_submission_path = output_file
        print(f"Saved submission to {output_file}")
        return submission

    def svr_model(self):
        # self.best_rbf_svr_model, self.svr_scores["rbf"] = self._svr_model_by_kernel(
        #     "rbf",
        #     {
        #         "svr__kernel": ["rbf"],
        #         "svr__C": [0.1, 1, 10, 100],
        #         "svr__gamma": ["scale", "auto", 0.01, 0.1],
        #         "svr__epsilon": [0.01, 0.1, 0.2, 0.5]
        #     }
        # )
        # self.best_linear_svr_model, self.svr_scores["linear"] = self._svr_model_by_kernel(
        #     "linear",
        #     {
        #         "svr__kernel": ["linear"],
        #         "svr__C": [0.1, 1, 10, 100],
        #         "svr__epsilon": [0.01, 0.1, 0.5]
        #     }
        # )
        return self.svr_poly_model()

    def svr_poly_model(self):
        self.best_poly_svr_model, self.svr_scores["poly"] = self._svr_model_by_kernel(
            "poly",
            {
                "svr__kernel": ["poly"],
                "svr__degree": [3],
                "svr__coef0": [0],
                "svr__C": [100, 150],
                "svr__epsilon": [0.1, 1, 10],
                "svr__gamma": ["scale"]
            }
        )
        self.best_svr_kernel = "poly"
        self.best_svr_model = self.best_poly_svr_model
        print(f"overall best SVR kernel: {self.best_svr_kernel}")
        print(f"overall best SVR score: {self.svr_scores[self.best_svr_kernel]}")
        print(f"overall best SVR model: {self.best_svr_model}")
        return {
            "poly": self.best_poly_svr_model,
            "best": self.best_svr_model
        }

    def svr_rbf_model(self):
        self.best_rbf_svr_model, self.svr_scores["rbf"] = self._svr_model_by_kernel(
            "rbf",
            {
                "svr__kernel": ["rbf"],
                "svr__C": [10000],
                "svr__gamma": ["scale", 0.01, 0.1, 0.5, 1, 1.5, 5, 8, 10],
                "svr__epsilon": [1.0]
            }
        )
        self.best_svr_kernel = "rbf"
        self.best_svr_model = self.best_rbf_svr_model
        return self.best_rbf_svr_model

    def svr_rbf_elasticnet_model(self):
        svr_elasticnet_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("stack", StackingRegressor(
                estimators=[
                    ("svr", SVR(kernel="rbf")),
                    ("elasticnet", ElasticNet(max_iter=10000, random_state=42))
                ],
                final_estimator=ElasticNet(max_iter=10000, random_state=42),
                cv=5
            ))
        ])
        kf = OutlierFilteredKFold(n_splits=10, shuffle=True, random_state=42)
        svr_elasticnet_hyperparams = {
            "stack__svr__C": [10000],
            "stack__svr__gamma": ["scale"],
            "stack__svr__epsilon": [1.0],
            "stack__elasticnet__alpha": [0.1, 0.2, 0.5],
            "stack__elasticnet__l1_ratio": [0.05, 0.1, 0.2, 0.5, 0.9],
            "stack__final_estimator__alpha": [0.001, 0.01, 0.05, 0.1],
            "stack__final_estimator__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        svr_elasticnet_grid = GridSearchCV(svr_elasticnet_pipeline,
                                           svr_elasticnet_hyperparams,
                                           scoring='neg_mean_squared_error',
                                           cv=kf, n_jobs=-1, verbose=1)
        svr_elasticnet_grid.fit(self.x_train, self.y_train)
        print(svr_elasticnet_grid.best_params_)
        print(f"best score: {svr_elasticnet_grid.best_score_}")
        self.best_rbf_elasticnet_svr_model = svr_elasticnet_grid.best_estimator_
        self.best_svr_model = self.best_rbf_elasticnet_svr_model
        self.best_svr_kernel = "rbf_elasticnet"
        print(self.best_rbf_elasticnet_svr_model)
        return self.best_rbf_elasticnet_svr_model

    def _svr_model_by_kernel(self, kernel_name: str, svr_hyperparams: dict):
        svr_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svr", SVR())
        ])
        kf = OutlierFilteredKFold(n_splits=10, shuffle=True, random_state=42)
        svr_grid = GridSearchCV(svr_pipeline, svr_hyperparams,
                                scoring={
                                    'neg_mean_squared_error': 'neg_mean_squared_error',
                                    'r2': 'r2'
                                },
                                refit='neg_mean_squared_error',
                                cv=kf, n_jobs=-1, verbose=1)
        svr_grid.fit(self.x_train, self.y_train)
        print(f"{kernel_name} best params: {svr_grid.best_params_}")
        print(f"{kernel_name} best score: {svr_grid.best_score_}")
        print(f"{kernel_name} best r2 score: {svr_grid.cv_results_['mean_test_r2'][svr_grid.best_index_]}")
        print(f"{kernel_name} best model: {svr_grid.best_estimator_}")
        return svr_grid.best_estimator_, svr_grid.best_score_

    def save_svr_predictions(self, output_path: str = "submission.csv", model_name: str = "rbf") -> pd.DataFrame:
        svr_models = {
            "rbf": self.best_rbf_svr_model,
            "linear": self.best_linear_svr_model,
            "poly": self.best_poly_svr_model
        }
        if model_name not in svr_models:
            raise ValueError("model_name must be one of: rbf, linear, poly.")

        model = svr_models[model_name]
        if model is None:
            raise ValueError(f"best_{model_name}_svr_model is empty. Train the SVR model first.")
        if self.test_data is None:
            raise ValueError("test_data is empty. Read the test CSV first.")
        if self.test_id is None:
            raise ValueError("test_id is empty. Read the test CSV first.")

        output_file = Path(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_file.with_name(f"{output_file.stem}_{model_name}_{timestamp}{output_file.suffix}")

        predictions = model.predict(self.test_data)
        submission = pd.DataFrame({
            "Id": self.test_id,
            "target": predictions
        })
        submission.to_csv(output_file, index=False)
        self.last_submission_path = output_file
        print(f"Saved submission to {output_file}")
        return submission

    def save_all_svr_predictions(self, output_path: str = "submission.csv") -> dict[str, pd.DataFrame]:
        return {
            "rbf": self.save_svr_predictions(output_path, model_name="rbf"),
            "linear": self.save_svr_predictions(output_path, model_name="linear"),
            "poly": self.save_svr_predictions(output_path, model_name="poly")
        }

    def save_best_svr_predictions(self, output_path: str = "submission.csv") -> pd.DataFrame:
        if self.best_svr_kernel is None:
            raise ValueError("best_svr_kernel is empty. Train the SVR model first.")

        return self.save_svr_predictions(output_path, model_name=self.best_svr_kernel)

    def save_svr_rbf_elasticnet_predictions(self, output_path: str = "submission.csv") -> pd.DataFrame:
        if self.best_rbf_elasticnet_svr_model is None:
            raise ValueError("best_rbf_elasticnet_svr_model is empty. Train the model first.")
        if self.test_data is None:
            raise ValueError("test_data is empty. Read the test CSV first.")
        if self.test_id is None:
            raise ValueError("test_id is empty. Read the test CSV first.")

        output_file = Path(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_file.with_name(
            f"{output_file.stem}_rbf_elasticnet_{timestamp}{output_file.suffix}"
        )

        predictions = self.best_rbf_elasticnet_svr_model.predict(self.test_data)
        submission = pd.DataFrame({
            "Id": self.test_id,
            "target": predictions
        })
        submission.to_csv(output_file, index=False)
        self.last_submission_path = output_file
        print(f"Saved submission to {output_file}")
        return submission

    def plot_column_distributions(
        self,
        train_data: pd.DataFrame | None = None,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        bins: int = 40,
        max_categories: int = 30,
    ) -> None:
        """Plot value distributions for each column in the training data.

        Numeric columns are plotted as histograms using only values between the
        lower and upper quantiles, which keeps extreme outliers from dominating
        the scale. Non-numeric columns are plotted as bar charts of their most
        common values.
        """
        data = train_data if train_data is not None else self.train_data
        if data is None:
            raise ValueError("No train_data provided and self.train_data is empty.")

        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")

        for column in data.columns:
            series = data[column].dropna()
            if series.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))

            if pd.api.types.is_numeric_dtype(series):
                lower = series.quantile(lower_quantile)
                upper = series.quantile(upper_quantile)
                filtered = series[(series >= lower) & (series <= upper)]

                if filtered.empty:
                    plt.close(fig)
                    continue

                ax.hist(filtered, bins=bins, color="steelblue", edgecolor="black")
                ax.set_xlim(lower, upper)
                ignored_count = len(series) - len(filtered)
                ax.set_title(f"{column} distribution ({ignored_count} outliers ignored)")
                ax.set_xlabel(column)
                ax.set_ylabel("Frequency")
            else:
                counts = series.astype(str).value_counts().head(max_categories)
                ax.bar(counts.index, counts.values, color="steelblue", edgecolor="black")
                ax.set_title(f"{column} distribution")
                ax.set_xlabel(column)
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.show()
