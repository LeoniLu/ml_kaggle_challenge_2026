from two_stage_classifier_regressor import TwoStageClassifierRegressor
from sklearn.metrics import r2_score

def main():
    train_data_path = "./spring2026_kaggle_linear_regression_challenge_train.csv"
    test_data_path = "./spring2026_kaggle_linear_regression_challenge_test.csv"

    ts = TwoStageClassifierRegressor()
    ts.read_train_csv(train_data_path)
    ts.read_test_csv(test_data_path)
    ts.extreme_points_classifier()

    ts.classifier_prediction()
    ts.core_svr_model()
    ts.ext_rf_model()

    ts.two_stage_prediction()
    r2 = r2_score(ts.y_train, ts.final_preds)
    print(f"Overall R2 score on train data is: {r2}")
    #ts.save_predictions("rfclf_svr_rf_submission.csv")

if __name__ == "__main__":
    main()