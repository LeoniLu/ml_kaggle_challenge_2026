import os, shutil
import pandas as pd
from data_parser import DataParser
from svr import SVRTrainer

dir = os.path.dirname(__file__)
output_dir = os.path.join(dir, 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def clear_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

train_data_path = f"{dir}/spring2026_kaggle_linear_regression_challenge_train.csv"
test_data_path = f"{dir}/spring2026_kaggle_linear_regression_challenge_test.csv"

data = DataParser()
data.parse(train_data_path, test_data_path)

C = [2000.0]
e = [379.2741935483871]
g = ['auto']
k = ['sigmoid']
svr = SVRTrainer(data, 3, scalor="Robust", refit=True)
svr.set_C(C)
svr.set_kernel(k)
svr.set_epsilon(e)
svr.set_gamma(g)
svr.search()
svr.pretty_print()
svr.save(f'{output_dir}/svr_reprod_best.json')
predx = data.test_x().fillna(data.train_x().mean(numeric_only=True))
predicted = svr.predict(predx)
submission = pd.DataFrame({
    data.ID: data.test_data_raw[data.ID],
    data.Y: predicted
})
submission.to_csv(f"{output_dir}/svr_repro_best.csv", index=False)

