import pandas as pd

from svr import SVRTrainer


def range_float(min, max, num):
    step = (max - min) / (num - 1)
    return [min + x * step for x in range(num)]

class SVRRunner:

    def __init__(self, data, output_dir, cmin, cmax, emin, emax, fold=5, steps=8, scaler="Robust", name="run"):
        self.output_dir = output_dir
        self.data = data
        self.cmin = cmin
        self.cmax = cmax
        self.emin = emin
        self.emax = emax
        self.fold = fold
        self.steps = steps
        self.scaler = scaler
        self.name = name
        self.transform_func = None
        self.last_model = None

    def run(self):
        svr = self.iter_svc(0, self.data, self.cmin, self.cmax, self.emin, self.emax, float("-inf"), -1, ([], [], []))
        predx = self.data.test_x().fillna(self.data.train_x().mean(numeric_only=True))
        predicted = svr.predict(predx)
        submission = pd.DataFrame({
            self.data.ID: self.data.test_data_raw[self.data.ID],
            self.data.Y: predicted
        })
        self.last_model = svr
        return submission

    def transform(self, func):
        self.transform_func = func
        return self

    def run_once(self):
        c = range_float(self.cmin, self.cmax, self.steps)
        g = ["scale", "auto"]
        # g.extend(range_float(gmin, gmax, steps))
        e = range_float(self.emin, self.emax, self.steps)
        svr = SVRTrainer(self.data, self.fold, scalor=self.scaler, refit=True, transform_func=self.transform_func)
        svr.set_C(c)
        svr.set_gamma(g)
        svr.set_epsilon(e)
        svr.set_kernel(["rbf", "sigmoid"])
        svr.search()
        svr.pretty_print()
        svr.save(f"{self.output_dir}/svr_{self.name}.json")
        predx = self.data.test_x().fillna(self.data.train_x().mean(numeric_only=True))
        predicted = svr.predict(predx)
        submission = pd.DataFrame({
            self.data.ID: self.data.test_data_raw[self.data.ID],
            self.data.Y: predicted
        })
        self.last_model = svr
        return submission

    def iter_svc(self, idx, data, cmin, cmax, emin, emax, best, bestidx, bestparam):
        print(f"#### Starting iteration {idx} #### current best: {best} (run {bestidx})")
        c = range_float(cmin, cmax, self.steps)
        g = ["scale", "auto"]
        # g.extend(range_float(gmin, gmax, steps))
        e = range_float(emin, emax, self.steps)
        svr = SVRTrainer(data, self.fold, scalor=self.scaler, transform_func=self.transform_func)
        svr.set_C(c)
        svr.set_gamma(g)
        svr.set_epsilon(e)
        svr.set_kernel(["rbf", "sigmoid"])
        svr.search()
        svr.pretty_print()
        svr.save(f"{self.output_dir}/svr_{self.name}_{idx}.json")

        def ex():
            svr = SVRTrainer(data, self.fold, scalor=self.scaler, refit=True, transform_func=self.transform_func)
            svr.set_C(bestparam[0])
            svr.set_gamma(bestparam[1])
            svr.set_epsilon(bestparam[2])
            svr.set_kernel(["rbf", "sigmoid"])
            svr.search()
            svr.pretty_print()
            svr.save(f"{self.output_dir}/svr_{self.name}_best.json")
            self.last_model = svr
            return svr

        if idx >= 30:
            print("Maximun iteration reached")
            return ex()
        if svr.best_score > best:
            best = svr.best_score
            bestidx = idx
            bestparam = (c, g, e)
        else:
            print("Best score found")
            return ex()
        if svr.best_params["estimator__C"] > (cmax + cmin) / 2:
            newcmin = (cmax + cmin) / 2
            newcmax = cmax
        else:
            newcmin = cmin
            newcmax = (cmax + cmin) / 2
        if svr.best_params["estimator__epsilon"] > (emax + emin) / 2:
            newemin = (emax + emin) / 2
            newemax = emax
        else:
            newemin = emin
            newemax = (emax + emin) / 2
        return self.iter_svc(idx + 1, data, newcmin, newcmax, newemin, newemax, best, bestidx, bestparam)



class HubertRunner:
    pass