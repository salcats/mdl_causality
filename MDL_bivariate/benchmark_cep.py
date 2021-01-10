""" This file performs benchmark test using the Tuebingen CEP.
"""
import numpy as np
import multiprocessing
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from cdt.data import load_dataset
from cisc_combined import CISCC
from utils import plot_multiline, reverse_argsort
from utils import plot_confidence_tcep, plot_accuracy_with_decision_rate_tcep
from utils import plot_conf_v_p_value, extract_cep_data, plot_accuracy_comparison
from matplotlib import pyplot as plt
from math import isinf, isnan
from joblib import Parallel, delayed
from cdt.causality.pairwise import RECI, IGCI, CDS, ANM, RCC
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

class BenchmarkTestCep():

    def __init__(self):
        self.data = extract_cep_data()
        self.data_no_weights = self.data.drop(columns=['weights'])
        self.no_pairs = len(self.data)
        self.ciscc_output = None
        self.ciscc_model = CISCC(1e3, 0.1)

    def evaluate_pair(self, index, preprocess):
        """ This method evaluates a given pair. """
        x = self.data['A'][index]
        y = self.data['B'][index]
        self.ciscc_model.add_data(x, y)
        output = np.asarray((self.ciscc_model.compute_cisc(preprocess)))
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

    def evaluate_all_pairs(self, preprocess):
        """ This method evaluates all pairs. """
        results = np.zeros((self.no_pairs, 3))
        for index in range(self.no_pairs):
            results[index, :] = self.evaluate_pair(index, preprocess)

        #results[:, 1] = (1 - np.exp(-results[:, 1]))/(1 + np.exp(-results[:, 1]))
        #(results[:, 1] - np.min(results[:, 1]))/(
        #                np.max(results[:, 1]) - np.min(results[:, 1]))

        return results


    def evaluate_all_pairs_method(self, method, preprocess):

        if preprocess:
            for pair in range(len(self.data_no_weights['A'])):
                x = self.data_no_weights['A'][pair]
                y = self.data_no_weights['B'][pair]

                x = (x - np.max(x)) / (np.max(x)-np.min(x))
                y = (y - np.max(y)) / (np.max(y)-np.min(y))
                # Standardise the data
                x = (x-np.mean(x)) / np.std(x)
                y = (y-np.mean(y)) / np.std(y)

                self.data_no_weights['A'][pair] = x
                self.data_no_weights['B'][pair] = y

        if method == 'RECI':
            obj = RECI(degree=2)
            output = obj.predict(self.data_no_weights)
            results = np.zeros((len(output), 2))
            results[:, 1] = np.abs(output)
            for i in range(len(output)):
                if output[i] > 0:
                    cause = 1
                else:
                    cause = -1
                results[i, 0] = cause
        elif method == 'IGCI':
            obj = IGCI()
            output = obj.predict(self.data_no_weights)
            results = np.zeros((len(output), 2))
            results[:, 1] = np.abs(output)
            for i in range(len(output)):
                if output[i] > 0:
                    cause = 1
                else:
                    cause = -1
                results[i, 0] = cause
        elif method == 'IGCI':
            obj = IGCI()
            output = obj.predict(self.data_no_weights)
            results = np.zeros((len(output), 2))
            results[:, 1] = np.abs(output)
            for i in range(len(output)):
                if output[i] > 0:
                    cause = 1
                else:
                    cause = -1
                results[i, 0] = cause
        elif method == 'CDS':
            obj = CDS()
            output = obj.predict(self.data_no_weights)
            results = np.zeros((len(output), 2))
            results[:, 1] = np.abs(output)
            for i in range(len(output)):
                if output[i] > 0:
                    cause = 1
                else:
                    cause = -1
                results[i, 0] = cause
        elif method == 'ANM':
            obj = ANM()
            output = obj.predict(self.data_no_weights)
            results = np.zeros((len(output), 2))
            results[:, 1] = np.abs(output)
            for i in range(len(output)):
                if output[i] > 0:
                    cause = 1
                else:
                    cause = -1
                results[i, 0] = cause
        elif method == 'RCC':
            labels = np.ones((len(self.data_no_weights)))
            X_tr, X_te, y_tr, y_te = train_test_split(self.data_no_weights,
                                                      labels,
                                                      train_size=.5)
            obj = RCC()
            obj.fit(X_tr, y_tr)
            output = np.concatenate( obj.predict(self.data_no_weights), axis=0 )
            results = np.zeros((len(output), 2))
            results[:, 1] = np.abs(output)
            for i in range(len(output)):
                if output[i] > 0:
                    cause = 1
                else:
                    cause = -1
                results[i, 0] = cause
        return results

    def compare_methods(self, methods):

        results_CISCC = self.evaluate_all_pairs(True)
        causes = results_CISCC[:, 0]
        scores = results_CISCC[:, 1]
        methods.remove('CISCC')
        for method in methods:

            results = self.evaluate_all_pairs_method(method, True)
            causes = np.column_stack((causes, results[:, 0]))
            scores = np.column_stack((scores, results[:, 1]))

        methods.insert(0, 'CISCC')
        return causes, scores


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

    benchmark_test = BenchmarkTestCep()

    methods = ['CISCC','RECI', 'IGCI', 'CDS']
    causes, scores = benchmark_test.compare_methods(methods)
    print (np.sum(np.sum(causes == 1, axis = 1) > 0)/benchmark_test.no_pairs)
    plot_accuracy_comparison(benchmark_test, causes, scores, methods, True)
    quit()

    '''
    plot_confidence_tcep(results, 0.001)
    plot_accuracy_with_decision_rate_tcep(benchmark_test, results,
                                         confidence=True)
    #plot_conf_v_p_value(results, True)
    '''
