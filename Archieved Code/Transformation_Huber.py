from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt

class Polynomialx9Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, x_train):
        X_copy = x_train.copy()
        if 'x9' in X_copy.columns:
            X_copy['x9_cubed'] = X_copy['x9'] ** 3
            X_copy['x9_abs_prod'] = X_copy['x9'] * X_copy['x9'].abs()
        else:
            print("Warning: 'x9' not found in features. Skipping polynomial transformation for x9.")
        return X_copy

class FeatureSelectorAndWeightedScaler(BaseEstimator, TransformerMixin):
    def __init__(self, extended_correlation_series, selection_threshold=0.2, x_columns=None):
        self.extended_correlation_series = extended_correlation_series
        self.selection_threshold = selection_threshold
        self.x_columns = x_columns
        self.features_to_keep_ = None
        self.feature_weights_ = None  # Stores weights as a Series

    def fit(self, X, y=None):
        # Identify features present in X and in the correlation series
        common_features = [f for f in self.x_columns if f in self.extended_correlation_series.index]
        current_extended_corrs = self.extended_correlation_series[common_features]
        valid_correlations = current_extended_corrs.dropna()
        self.features_to_keep_ = valid_correlations[valid_correlations.abs() >= self.selection_threshold].index.tolist()
        if not self.features_to_keep_:
            raise ValueError(
                f"No features left after correlation-based selection with threshold {self.selection_threshold}. "
                "Adjust threshold or check extended_correlation_series."
            )
        # print(f"Features kept after selection (threshold {self.selection_threshold}): {self.features_to_keep_}")
        kept_correlations_abs = valid_correlations[self.features_to_keep_].abs()

        temp_feature_weights_raw = {}
        for feature in self.features_to_keep_:
            temp_feature_weights_raw[feature] = np.abs(valid_correlations[feature])
            # if feature in ['x9_cubed', 'x9_abs_prod']:
            #     # Assign the absolute correlation of 'x9' as their base weight value
            #     if 'x9' in valid_correlations.index:
            #         temp_feature_weights_raw[feature] = np.abs(valid_correlations['x9'])
            #     else:
            #         # Fallback if x9 itself wasn't kept but its derivatives were - use derivative's own correlation
            #         print(f"Warning: '{feature}' kept but 'x9' not found in valid_correlations. Using '{feature}'s own correlation for weighting.")
            #         temp_feature_weights_raw[feature] = np.abs(valid_correlations[feature])
            # else:
            #     temp_feature_weights_raw[feature] = np.abs(valid_correlations[feature])

        # Convert to Series and normalize
        self.feature_weights_ = pd.Series(temp_feature_weights_raw)
        if not self.feature_weights_.empty:
            self.feature_weights_ = self.feature_weights_ / self.feature_weights_.sum()
        else:
            raise ValueError("No features to weight after selection.")

        if 'x9' in self.feature_weights_.index:
            self.feature_weights_['x9_cubed'] = self.feature_weights_['x9']
            self.feature_weights_['x9_abs_prod'] = self.feature_weights_['x9']
        # print(f"Calculated feature weights:\n{self.feature_weights_}")
        # print(f"Features to keep: {self.features_to_keep_} and {self.features_to_keep_}")
        return self

    def transform(self, X):
        if self.features_to_keep_ is None or self.feature_weights_ is None:
            raise RuntimeError("FeatureSelectorAndWeightedScaler must be fitted before transforming.")
        # Select only the features that were kept
        feature_to_keep_mask = []
        for i in self.features_to_keep_:
            feature_to_keep_mask.append(self.x_columns.get_loc(i))
        feature_to_keep_mask.append(len(self.x_columns))
        feature_to_keep_mask.append(len(self.x_columns) + 1)
        # print(f"Selected features: {self.features_to_keep_}, indices: {feature_to_keep_mask}")
        X_selected = X[:, feature_to_keep_mask].copy()
        # Apply weights to the selected features
        # Ensure weights are aligned with the columns of X_selected
        # print(f"X_selected shape: {X_selected.shape}")
        # print(f"feature_weights_ shape: {self.feature_weights_.shape}{self.feature_weights_.dtype}")
        weighted_X = X_selected * self.feature_weights_.values

        return weighted_X

class TransformationFeatureEngineering:
    def __init__(self):
        self.train_data = None
        self.x_train = None
        self.y_train = None
        self.train_id = None
        self.test_data = None
        self.test_id = None
        self.best_huber_regressor = None
        self.predictions = None

    def read_train_csv(self, file_path: str) -> pd.DataFrame:
        self.train_data = pd.read_csv(file_path)
        self.train_id = self.train_data['Id']
        self.x_train = self.train_data.drop(columns=['target', 'Id'])
        self.y_train = self.train_data['target']

    def read_test_csv(self, file_path: str) -> pd.DataFrame:
        self.test_data = pd.read_csv(file_path)
        self.test_id = self.test_data['Id']
        self.test_data = self.test_data.drop(columns=['Id'])

    def huber_model(self, extended_correlation_series: pd.Series):
        pipeline_steps = [
            ('poly_x9', Polynomialx9Transformer()),  # Add polynomial features first
            ('imputer', SimpleImputer(strategy='median')),  # Then handle NaNs
            ('scaler', RobustScaler()),  # Then scale the features
            ('selector_weigher', FeatureSelectorAndWeightedScaler(
                extended_correlation_series=extended_correlation_series,
                selection_threshold=0.02,
                x_columns=self.x_train.columns
            )),  # Then select and weight features
            ('huber', HuberRegressor(max_iter=10000))
        ]

        pipeline = Pipeline(steps=pipeline_steps)
        param_grid = {
            "huber__epsilon": [13.5],
            "huber__alpha": [0.001, 0.01, 0.1, 1]
        }
        grid_search = GridSearchCV(pipeline, param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(self.x_train, self.y_train)

        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation nMSE: {grid_search.best_score_}")
        self.best_huber_regressor = grid_search.best_estimator_

    def save_huber_predictions(self, filename):
        output_file = Path(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_file.with_name(f"{output_file.stem}_{timestamp}{output_file.suffix}")

        self.predictions = self.best_huber_regressor.predict(self.test_data)
        submission = pd.DataFrame({
            "Id": self.test_id,
            "target": self.predictions
        })
        submission.to_csv(output_file, index=False)
        print(f"Saved submission to {output_file}")


def main():
    train_data_path = "./spring2026_kaggle_linear_regression_challenge_train.csv"
    test_data_path = "./spring2026_kaggle_linear_regression_challenge_test.csv"

    huber_trans = TransformationFeatureEngineering()
    huber_trans.read_train_csv(train_data_path)
    huber_trans.read_test_csv(test_data_path)

    temp_poly_transformer = Polynomialx9Transformer()
    x_train_extended_for_corr = temp_poly_transformer.fit_transform(huber_trans.x_train)
    extended_correlation_series = x_train_extended_for_corr.corrwith(huber_trans.y_train)
    print("Extended Correlation Series with Target:\n")
    print(extended_correlation_series.sort_values(ascending=False))

    huber_trans.huber_model(extended_correlation_series)
    y_pred = huber_trans.best_huber_regressor.predict(huber_trans.x_train)
    R2 = r2_score(huber_trans.y_train, y_pred)
    print("R2 score: ", R2)

    huber_trans.save_huber_predictions("Transformation_Huber_MSE_submission.csv")

    plt.figure(figsize=(10, 6))
    plt.scatter(huber_trans.test_data['x9'], huber_trans.predictions)
    plt.title(f'x9 vs predicted Target')
    plt.xlabel('x9')
    plt.ylabel('Target')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()