""" This file performs benchmark test using the Tuebingen CEP.
"""
import numpy as np
from cdt.data import load_dataset
from cisc_combined import CISCC
from utils import plot_multiline, reverse_argsort
from utils import plot_confidence_tcep, plot_accuracy_with_decision_rate_tcep
from utils import plot_conf_v_p_value
from matplotlib import pyplot as plt
from math import isinf, isnan


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

    def evaluate_all_pairs(self):
        """ This method evaluates all pairs. """
        results = np.zeros((self.no_pairs, 4))
        count = 0
        for index in range(self.no_pairs):
            results[index, :] = self.evaluate_pair(index)

        ## Remove infs and nans
        max_conf = np.max(results[:, 1][np.logical_not(np.isinf(results[:, 1]))])
        results[:, 1][np.isinf(results[:, 1])] = max_conf
        results[:, 2][np.isnan(results[:, 2])] = 0

        return results

# Make some plotting functions and then port them out to utils


if __name__ == '__main__':

    benchmark_test = BenchmarkTest()
    results = benchmark_test.evaluate_all_pairs()

    #plot_confidence_tcep(results, 0.001, normalised=False)
    plot_accuracy_with_decision_rate_tcep(benchmark_test, results,
                                         confidence=True, normalised=False)
    plot_conf_v_p_value(results, False, True)
