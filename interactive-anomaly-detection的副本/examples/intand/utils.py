from collections import defaultdict
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import re
#import torch as pt


def minimize(cost_func, parameters, lr=0.1, iterations=2000):
    optimizer = pt.optim.Adam(parameters, lr=lr)
    for _ in range(iterations):
        optimizer.zero_grad()
        cost_func().backward()
        optimizer.step()


def feedback_loop(model, y, budget):
    for _ in range(budget):
        i = model.get_top1()
        feedback = y[i]  # this is the expert feedback
        model.update(i, feedback)
    if model.plot: plt.show()


class Evaluation:
    def __init__(self, y, expname=None, budget=None):
        self.y = y
        self.expname = expname if expname is not None else str(time.time())
        self.n_anomalies = (y[y == 1]).sum()
        self.budget = min(2 * self.n_anomalies, 100) if budget is None else budget
        self.precision_results = defaultdict(list)
        self.effort_results = defaultdict(list)

    def run(self, cmodels):
        names = [model.__class__.__name__ for model in cmodels]
        names = [v + str(names[:i].count(v) + 1) if names.count(v) > 1 else v for i, v in enumerate(names)]

        for name, model in zip(names, cmodels):
            feedback_loop(model, self.y, self.budget)
            self.precision_results[name].append(model.get_precision(self.n_anomalies))
            self.effort_results[name].append(model.get_expert_effort())

        results = {"precision":self.precision_results, "effort":self.effort_results}
        json.dump(results, open(self.expname + '.json', 'w'))


    @staticmethod
    def plot(dic_results, xlabel, ylabel):
        colour = {}
        for method, runs in sorted(dic_results.items()):
            mu = np.mean(runs, axis=0)
            std = np.std(runs, axis=0) / np.sqrt(len(runs))
            z = [int(s) for s in re.findall(r'\d+',method)]
            if (z == [] or z == [1] ):
                p = plt.errorbar(range(len(mu)), mu, yerr=std, label=method, errorevery=len(mu) // 10)
                if (z == [1]):
                    colour.update({method[:-1]:p[0].get_color()})
            else:
                plt.errorbar(range(len(mu)), mu, yerr=std, label=method, errorevery=len(mu) // 10, c = colour[method[:-1]],ls='--')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

    @staticmethod
    def plot_results(filename, savefig=True):
        results = json.load(open(filename))

        plt.figure(0)
        plt.ylim((0, 105))
        Evaluation.plot(results["precision"], "Nbr. of Feedbacks (Budget)", "Ratio of Anomalies Detected")
        if savefig: plt.savefig(filename + "_precision" + ".png")
        else: plt.show()
        plt.close()

        plt.figure(1)
        Evaluation.plot(results["effort"], "Nbr. of Feedbacks (Budget)", "Expert Effort")
        if savefig: plt.savefig(filename + "_effort" + ".png")
        else: plt.show()
        plt.close()
        
    @staticmethod
    def plot_precision(filename, savefig=True):
        results = json.load(open(filename))
        p = results["precision"]
        #p_end = p[-1]
        return p
    @staticmethod
    def plot_effort(filename, savefig=True):
        results = json.load(open(filename))
        e = results["effort"]
        #e_end = e[-1]
        return e

