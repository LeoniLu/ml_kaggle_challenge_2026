from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.preprocessing import FunctionTransformer

from data_parser import DataParser
from  trainer import BaseTrainer

class SVRTrainer(BaseTrainer):
    def __init__(self, data: DataParser, fold_splits=10, scoring="neg_mean_squared_error", refit=False, scalor="Robust", transform_func=None):
        super(SVRTrainer, self).__init__(fold_splits=fold_splits, scoring=scoring, refit=refit)
        self.data = data
        self.C = []
        self.gamma = []
        self.kernel = []
        self.epsilon = []
        self.scaler = scalor
        self.transform_func = transform_func

    def estimator_factory(self):
        return SVR(degree=7)
    def scaler_factory(self):
        if self.scaler == "Robust":
            return RobustScaler(quantile_range=(10,90))
        if self.scaler == "Standard":
            return StandardScaler()
        return None

    def transformer_factory(self):
        if self.transform_func is None:
            return None
        return FunctionTransformer(func=self.transform_func, validate=False)

    def set_C(self, C):
        self.C = C

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_kernel(self, kernel):
        self.kernel = kernel

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def gen_params(self, params=None):
        params = super(SVRTrainer, self).gen_params(params=params)
        params[self.pipline_param_name(self.ESTIMATOR, "C")] = []
        for c in self.C:
            params[self.pipline_param_name(self.ESTIMATOR, "C")].append(c)

        params[self.pipline_param_name(self.ESTIMATOR, "gamma")] = []
        for g in self.gamma:
            params[self.pipline_param_name(self.ESTIMATOR, "gamma")].append(g)

        params[self.pipline_param_name(self.ESTIMATOR, "kernel")] = []
        for k in self.kernel:
            params[self.pipline_param_name(self.ESTIMATOR, "kernel")].append(k)

        params[self.pipline_param_name(self.ESTIMATOR, "epsilon")] = []
        for e in self.epsilon:
            params[self.pipline_param_name(self.ESTIMATOR, "epsilon")].append(e)
        return params

    def search(self, x=None, y=None, params=None):
        hyper_params:dict[str] = self.gen_params()
        if x is None:
            x = self.data.train_x()
        if y is None:
            y = self.data.train_y()
        if params is not None:
            hyper_params.update(params)
        super(SVRTrainer, self).search(x, y, hyper_params)

    def pretty_print(self):
        print("==============SVR==============")
        self.pretty_print_param(self.gen_params())
        self.pretty_print_results()

