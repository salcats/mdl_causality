""" This file performs benchmark test using the Tuebingen CEP.
"""
import numpy as np
from cdt.data import load_dataset
from cisc_combined import CISCC
from utils import plot_multiline, reverse_argsort
from matplotlib import pyplot as plt
import matplotlib
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
        results = np.zeros((self.no_pairs, 5))
        for index in range(self.no_pairs):
            results[index, :] = self.evaluate_pair(index)

        return results


if __name__ == '__main__':

    benchmark_test = BenchmarkTest()
    results = benchmark_test.evaluate_all_pairs()
    results[:, 0] = results[:, 0] == 1
    accuracy = np.sum(results[:, 0] == 1)/benchmark_test.no_pairs
    indices = reverse_argsort(results[:, 1])
    sorted_labels = [results[:, 0][i] for i in indices]
    sorted_conf = [results[:, 1][i] for i in indices]
    sorted_p = [results[:, 3][i] for i in indices]

    cmap = matplotlib.colors.ListedColormap(['red', 'green'])

    plt.scatter(range(len(sorted_conf)), sorted_conf, s = 2, c=sorted_labels, cmap=cmap)

    plt.show()
    quit()
    sorted_p_values = [results[:, 3][i] for i in indices]

    '''

    conf = [x for x in results[:, 1] if not isinf(x)]
    pv = [x for x in results[:, 3] if x != 0]
    plt.scatter(conf, -np.log(pv))
    plt.show()

    conf = [x for x in results[:, 2] if not isnan(x)]
    pv = [x for x in results[:, 4] if not isnan(x)]
    plt.scatter(conf, -np.log(pv))
    plt.show()
    '''
    dec_rate = np.arange(0.02, 1.01, 0.01)
    accs = []
    for r in dec_rate:
        maxIdx = int(r * benchmark_test.no_pairs)
        labels = sorted_labels[:maxIdx]
        accs.append(sum(labels) / len(labels))

    plot_multiline([accs], dec_rate, [
                    "COMBINED: " + str(np.round(accuracy, 4))], "decision rate",
                   "accuracy", "decision rate versus accuracy")
