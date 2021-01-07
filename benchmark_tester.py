""" This file performs benchmark test using the Tuebingen CEP.
"""
import numpy as np
import multiprocessing
from cdt.data import load_dataset
from cisc_combined import CISCC
from utils import plot_multiline, reverse_argsort
from utils import plot_confidence_tcep, plot_accuracy_with_decision_rate_tcep
from utils import plot_conf_v_p_value
from matplotlib import pyplot as plt
from math import isinf, isnan
from joblib import Parallel, delayed

class BenchmarkTest():

    def __init__(self):
        self.data, self.labels = load_dataset('tuebingen')
        self.no_pairs = len(self.data)
        #self.remove_multivariate_pairs()
        self.ciscc_output = None
        self.ciscc_model = CISCC(1e3, 0.1)

    def remove_multivariate_pairs(self):
        """ This method removes multivariate pairs. """
        self.data = self.data.drop(['pair52', 'pair53', 'pair54',
                                    'pair55', 'pair71'])
        self.labels = self.labels.drop(['pair52', 'pair53', 'pair54',
                                        'pair55', 'pair71'])
        self.no_pairs = self.no_pairs - 5

    def evaluate_pair(self, index):
        """ This method evaluates a given pair. """
        x = self.data['A'][index]
        y = self.data['B'][index]
        self.ciscc_model.add_data(x, y)
        output = np.asarray((self.ciscc_model.compute_cisc()))

        return output

    def evaluate_pair_with_ensemble(self,
                                    index,
                                    no_base_learners):
        """ This method evaluates a given pair using an ensemble based on this
        paper https://arxiv.org/pdf/2006.03231.pdf. Didn´t work well. Maybe won´t
        leave it in. """
        x = self.data['A'][index]
        y = self.data['B'][index]

        if len(x) <= 500:
            sample_fraction = 0.75
        elif 500 < len(x) and len(x) <= 1000:
            sample_fraction = 0.5
        elif 1000 < len(x) and len(x) <= 2000:
            sample_fraction = 0.25
        elif 2000 < len(x) and len(x) <= 10000:
            sample_fraction = 0.125
        elif 10000 < len(x):
            sample_fraction = 0.1

        sample_fraction = 0.2
        sample_size = np.max([50, int(sample_fraction*len(x))])
        results = np.zeros((no_base_learners, 3))
        num_cores = multiprocessing.cpu_count()

        learners = range(no_base_learners)

        def compute_cisc(i, x, y):
            random_indices = np.random.choice(range(len(x)), sample_size, replace=False)
            x_sample = x[random_indices]
            y_sample = y[random_indices]
            self.ciscc_model.add_data(x_sample, y_sample)
            return np.asarray((self.ciscc_model.compute_cisc()))

        par_out = Parallel(n_jobs=num_cores)(delayed(compute_cisc)(learner, x, y) for learner in learners)

        for learner in learners:
            causal_score = par_out[learner][0]
            confidence = par_out[learner][1]
            p_value = par_out[learner][2]
            causal_score = causal_score*confidence
            results[learner, :] = [causal_score, confidence, p_value]

        output = [0, 0, 0]
        output[0] = np.sum(results[:, 0])
        output[1] = np.mean(results[:, 1])
        output[2] = np.mean(results[:, 2])

        return output

    def evaluate_all_pairs(self):
        """ This method evaluates all pairs. """
        results = np.zeros((self.no_pairs, 3))
        for index in range(self.no_pairs):
            results[index, :] = self.evaluate_pair(index)
        #results[:, 1] = (1 - np.exp(-results[:, 1]))/(1 + np.exp(-results[:, 1]))
        #(results[:, 1] - np.min(results[:, 1]))/(
        #                np.max(results[:, 1]) - np.min(results[:, 1]))

        return results

    def evaluate_all_pairs_with_ensemble(self, no_base_learners):
        """ This method evaluates all pairs with ensemble. """
        results = np.zeros((self.no_pairs, 3))
        for index in range(self.no_pairs):
            results[index, :] = self.evaluate_pair_with_ensemble(index,
                                                                 no_base_learners)

        results[:, 1] = (results[:, 1] - np.min(results[:, 1]))/(
                        np.max(results[:, 1]) - np.min(results[:, 1]))

        return results


if __name__ == '__main__':

    benchmark_test = BenchmarkTest()

    #results = benchmark_test.evaluate_all_pairs_with_ensemble(50)
    results = benchmark_test.evaluate_all_pairs()

    plot_confidence_tcep(results, 0.001)
    plot_accuracy_with_decision_rate_tcep(benchmark_test, results,
                                         confidence=True)
    #plot_conf_v_p_value(results, True)
