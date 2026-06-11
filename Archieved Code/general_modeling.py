from data_processor import DataProcessor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def main():
    train_data_path = "./spring2026_kaggle_linear_regression_challenge_train.csv"
    test_data_path = "./spring2026_kaggle_linear_regression_challenge_test.csv"

    dp = DataProcessor()
    dp.read_train_csv(train_data_path)
    dp.read_test_csv(test_data_path)

    dp.random_forest_model()
    r2 = r2_score(dp.y_train, dp.best_rf_model.predict(dp.train_data))
    print(f"Overall R2 score on train data is: {r2}")
    #dp.save_random_forest_predictions("random_forest_MSE_submission.csv")

    #dp.linear_regression_model()
    #dp.save_linear_regression_predictions("linear_regression_MSE_remove_outlyer_submission.csv")

    #dp.ridge_regression_model()
    #dp.save_ridge_regression_predictions("ridge_regression_MSE_submission.csv")

    #dp.svr_poly_model()
    #dp.save_best_svr_predictions("svr_poly_MSE_remove_outlyer_submission.csv")
    # tuning hyperparameters good
    #dp.svr_rbf_model()
    #dp.save_svr_predictions("svr_MSE_submission.csv", model_name="rbf")
    #dp.svr_rbf_elasticnet_model()
    #dp.save_svr_rbf_elasticnet_predictions("svr_elasticnet_MSE_submission.csv")

    #dp.xgboost_model()
    #dp.save_xgboost_predictions("xgboost_MSE_submission.csv")

if __name__ == "__main__":
    main()