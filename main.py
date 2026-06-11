import os
import shutil
from sklearn.metrics import r2_score

from data_parser import X9Mapper, DataAugment, DataParser
from runners import SVRRunner


dir = os.path.dirname(__file__)
output_dir = os.path.join(dir, 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def clear_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def print_r2(data, predictor):
    x = data.train_x()
    y = data.train_y()
    y_predicted = predictor.predict(x)
    r2 = r2_score(y, y_predicted)
    print(f"Overall R2 score on train data is: {r2}")

def linear_scann(data, output_dir, transform=None):
    clear_dir(output_dir)
    name = "linear"
    svr_runner = SVRRunner(data, output_dir=output_dir, cmin=20, cmax=2000, emin=20, emax=2000, fold=3, steps=32,
                           scaler="Robust", name=name)
    svr_runner.transform(transform)
    submission = svr_runner.run()
    submission.to_csv(f"{output_dir}/{name}_output_3fold.csv", index=False)
    print_r2(data, svr_runner.last_model)

def expo_scann(data, output_dir, transform=None):
    clear_dir(output_dir)
    c = []
    e = []
    dirs =[]
    for pow in range(1, 6):
        coef = 0.25 * (10**pow)
        dirs.append(f"{output_dir}/param_coef_{pow}")
        os.makedirs(dirs[-1])
        c.append(coef)
        e.append(coef)

    for idx in range(len(dirs)-1):
        cmax = c[idx+1]
        cmin = c[idx]
        emax = e[idx+1]
        emin = e[idx]
        run_dir = dirs[idx]
        for scaler in ["Robust", "Standard", "NoScale"]:
            name = f"{idx}_{scaler}"
            print("**** Running ", name)
            svr_runner = SVRRunner(data, output_dir=run_dir, cmin=cmin, cmax=cmax, emin=emin, emax=emax, fold=3, steps=10, scaler=scaler, name=name)
            svr_runner.transform(transform)
            submission = svr_runner.run()
            submission.to_csv(f"{run_dir}/{name}_output.csv", index=False)

def main():
    train_data_path = f"{dir}/spring2026_kaggle_linear_regression_challenge_train.csv"
    test_data_path = f"{dir}/spring2026_kaggle_linear_regression_challenge_test.csv"

    data = DataParser()
    #data, outlier = X9Mapper(), False
    data.parse(train_data_path, test_data_path)

    # first run with expo_scann to get an extimate range on what C,e behaves the best
    #expo_scann(data, output_dir+"_rough_3fold")

    # after setting the min/max params in linear_scann from expo_scann's result,
    # run with linear_scann for second iteration for more fine tuned best param
    linear_scann(data, output_dir+"_fine_3fold")

if __name__ == "__main__":
    main()