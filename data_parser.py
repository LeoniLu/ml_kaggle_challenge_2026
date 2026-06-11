import random

import pandas as pd


class DataParser:
    def __init__(self):
        self.training_data_path = None
        self.test_data_path = None
        self.training_data_raw :pd.DataFrame | None = None
        self.test_data_raw :pd.DataFrame | None = None
        self.Y = "target"
        self.ID = "Id"
        self.X = []

    def parse(self, training_data_path, test_data_path):
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.training_data_raw = pd.read_csv(self.training_data_path)
        self.test_data_raw = pd.read_csv(self.test_data_path)
        self.X = [x for x in self.training_data_raw.columns if x != self.Y and x != self.ID]

    def test_x(self):
        return self.test_data_raw[self.X]

    def train_x(self):
        return self.training_data_raw[self.X]

    def train_y(self):
        return self.training_data_raw['target']

class DataAugment(DataParser):
    def parse(self, training_data_path, test_data_path):
        super().parse(training_data_path, test_data_path)
        training = self.training_data_raw.copy()
        self.training_data_raw = pd.concat([self.training_data_raw, training], ignore_index=True, axis=0)

class X9Mapper(DataParser):
    # mapping = func(df)->df with mapped value
    def __init__(self):
        super().__init__()
        self.mapped_training_x = None
        self.mapped_testing_x = None
        self.X9="x9"


    def train_x(self):
        if self.mapped_training_x is None:
            self.map_train()
        return self.mapped_training_x

    def test_x(self):
        if self.mapped_testing_x is None:
            self.map_test()
        return self.mapped_testing_x

    def map_train(self):
        self.mapped_training_x = super().train_x()
        x9max = self.training_data_raw[self.X9].max()
        x9min = self.training_data_raw[self.X9].min()
        def map_const(row):
            y = row[self.Y]
            x9 = row[self.X9]
            if (-200 > y > 200) or (x9 != x9):
                return random.uniform(x9min, x9max)
            else:
                return x9
        def map_cubic(row):
            y = row[self.Y]
            x9 = row[self.X9]
            if (-200 < y) or ( y < 200) or (x9 != x9):
                return 0.0
            else:
                return x9
        self.mapped_training_x[self.X9+"const"] = self.training_data_raw.apply(map_const, axis=1)
        self.mapped_training_x[self.X9+"cubic"] = self.training_data_raw.apply(map_cubic, axis=1)
        self.mapped_training_x = self.mapped_training_x.drop(columns=[self.X9])

    def map_test(self):
        self.mapped_testing_x = super().test_x()
        x9max = self.training_data_raw[self.X9].max()
        x9min = self.training_data_raw[self.X9].min()
        def map_const(row):
            x9 = row[self.X9]
            if (x9 != x9):
                return random.uniform(x9min, x9max)
            else:
                return x9
        def map_cubic(row):
            x9 = row[self.X9]
            if (x9 != x9):
                return 0.0
            else:
                return x9
        self.mapped_testing_x[self.X9+"const"] = self.test_data_raw.apply(map_const, axis=1)
        self.mapped_testing_x[self.X9+"cubic"] = self.test_data_raw.apply(map_cubic, axis=1)
        self.mapped_testing_x = self.mapped_testing_x.drop(columns=[self.X9])
