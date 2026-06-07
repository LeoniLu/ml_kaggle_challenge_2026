import json
from collections import defaultdict

from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR


class BaseTrainer:
    def __init__(self, fold_splits, scoring, impute_methods=("median", "mean"), refit=False):
        self.fold = KFold(n_splits=fold_splits, shuffle=True, random_state=42)
        self.scoring = scoring
        self.IMPUTER = "imputer"
        self.SCALER = "scaler"
        self.ESTIMATOR = "estimator"
        self.TRANSFORMER = "transformer"
        self.model = None
        self.best_params = None
        self.best_score = None
        self.impute_methods = impute_methods
        self.refit = refit

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump({
                "hyperparameters": self.best_params,
                "score": self.best_score,
            }
            , f, indent=2)

    def pipeline(self, estimator, scaler, transformer=None):
        args = [
            (self.IMPUTER, SimpleImputer())]
        if transformer is not None:
            args.append((self.TRANSFORMER, transformer))
        if scaler is not None:
            args.append((self.SCALER, scaler))
        args.append((self.ESTIMATOR, estimator))

        pl = Pipeline(args)
        return pl

    def pipline_param_name(self, step_name, param_name):
        return step_name + "__" + param_name

    def pretty_print_param(self, hyperparams):
        print("Searched Params:")
        info = defaultdict(dict)
        for k, v in hyperparams.items():
            step, param = k.split("__")
            info[step][param] = v
        for k, v in info.items():
            print(f"    Step {k}: ")
            for param, value in v.items():
                print(f"        {param}: {value}")

    def pretty_print_results(self):
        print("Results:")
        print(f"    Score: {self.best_score:0.6f}")
        print(f"    Params:")
        for k, v in self.best_params.items():
            print(f"        {k}: {v}")


    def search_grid(self, hyperparams, refit):
        pl = self.pipeline(self.estimator_factory(), self.scaler_factory(), self.transformer_factory())
        grid = GridSearchCV(pl, hyperparams, cv=self.fold, n_jobs=-1, verbose=True, refit=refit)
        return grid

    def search(self, x, y, hyperparams):
        grid = self.search_grid(hyperparams, self.refit)
        grid.fit(x, y)
        self.best_score = grid.best_score_
        self.best_params = grid.best_params_
        if self.refit:
            self.model = grid

    def train(self, x, y, hyperparams):
        self.model = self.estimator_factory()
        self.model.set_params(**hyperparams)
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def estimator_factory(self):
        raise NotImplementedError()

    def scaler_factory(self):
        raise NotImplementedError()

    def transformer_factory(self):
        raise NotImplementedError()

    def gen_params(self, params):
        if params is None:
            params = {}
        params[self.pipline_param_name(self.IMPUTER, "strategy")] = []
        for m in self.impute_methods:
            params[self.pipline_param_name(self.IMPUTER, "strategy")].append(m)
        return params
