import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, SplineTransformer, FunctionTransformer
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
from sklearn.metrics import r2_score
from datetime import datetime
from pathlib import Path

class HuberRegression_SpineTransformation:
    def __init__(self):
        self.train_data = None
        self.x_train = None
        self.y_train = None
        self.train_id = None
        self.test_data = None
        self.test_id = None
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

    def huber_model(self):
        numerical_features = self.x_train.columns.tolist()
        x9_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('spline', SplineTransformer())#,
            #('weight_adjust', FunctionTransformer(lambda X, factor=1.0: X * factor, kw_args={'factor': 1.0}, validate=False))
        ])
        other_numerical_features = [col for col in numerical_features if col != 'x9']
        other_numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        # Combine these preprocessing steps using ColumnTransformer
        # 'remainder="passthrough"' ensures any columns not explicitly transformed are kept.
        preprocessor = ColumnTransformer(
            transformers=[
                ('x9_transform', x9_pipeline, ['x9']),
                ('other_num_transform', other_numeric_pipeline, other_numerical_features)
            ],
            remainder='passthrough'
        )

        # Create the full pipeline including the preprocessor and the HuberRegressor
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', HuberRegressor(max_iter=10000))
        ])
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        # Define the parameter grid for GridSearchCV
        param_grid = {
            'preprocessor__x9_transform__spline__n_knots': [4],
            'preprocessor__x9_transform__spline__degree': [3],
            #'preprocessor__x9_transform__weight_adjust__kw_args': [{'factor': f} for f in [0.01, 0.1, 0.5, 1.0, 2.0]],
            'regressor__epsilon': [13.5],#[1.0, 1.35, 1.5, 2.0],  # 1.35 is the default value
            'regressor__alpha': [0.0001]  # 0.0001 is the default value
        }

        # Instantiate GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=kf,  # 10-fold cross-validation as requested
            scoring='neg_mean_squared_error',  # Using negative mean squared error for optimization
            n_jobs=-1,  # Use all available CPU cores
            verbose=1  # For more detailed output during fitting
        )

        # Fit GridSearchCV on the training data
        grid_search.fit(self.x_train, self.y_train)

        # Print the best parameters and best score found
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation negative MSE: {grid_search.best_score_}")
        self.best_huber_regressor = grid_search.best_estimator_

    def save_huber_predictions(self, output_path: str = "submission.csv") -> pd.DataFrame:
        output_file = Path(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_file.with_name(f"{output_file.stem}_{timestamp}{output_file.suffix}")

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

    hr = HuberRegression_SpineTransformation()
    hr.read_train_csv(train_data_path)
    hr.read_test_csv(test_data_path)
    hr.huber_model()
    y_pred = hr.best_huber_regressor.predict(hr.x_train)
    R2 = r2_score(hr.y_train, y_pred)
    print("R2 score: ", R2)

    hr.save_huber_predictions("Huber_SpineTransformation_MSE_submission.csv")


if __name__ == "__main__":
    main()