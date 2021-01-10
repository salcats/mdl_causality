"""Causal inference on discrete data using stochastic complexity
of multinomial model class.
"""
import numpy as np
import pandas as pd
from feature_engine.discretisation import EqualFrequencyDiscretiser
from collections import defaultdict
from collections import Counter
from utils import multinomial_with_recurrence
from sklearn.preprocessing import KBinsDiscretizer

__author__ = "Kailash Budhathoki"
__email__ = "kbudhath@mpi-inf.mpg.de"
__copyright__ = "Copyright (c) 2017"
__license__ = "MIT"


class CISCM():

    def __init__(self, x, y, scale_factor, preprocess=False):
        """ Initialise the class with the data. """
        self.x = x
        self.y = y
        self.scale_factor = scale_factor
        self.y_marginals = defaultdict(list)
        self.x_marginals = defaultdict(list)
        if preprocess:
            self.preprocess_data()
        self.discretise_data()
        self.find_marginals()
        self.sc_x = None
        self.sc_y = None
        self.sc_x_cond_y = None
        self.sc_y_cond_x = None
        self.x_causes_y = None
        self.y_causes_x = None
        self.confidence = None
        self.likelihood_ratio = None
        self.significance = None

    def discretise_data(self):
        """ This method discretises the data.
        Current implementation is very simple. """

        self.x = np.multiply(self.x, self.scale_factor).astype(np.int)
        self.y = np.multiply(self.y, self.scale_factor).astype(np.int)

    def preprocess_data(self):
        """ This method normalises and standardises the data. """

        # Normalise the data
        self.x = (self.x - np.max(self.x)) / (np.max(self.x)-np.min(self.x))
        self.y = (self.y - np.max(self.y)) / (np.max(self.y)-np.min(self.y))

        # Standardise the data
        self.x = (self.x-np.mean(self.x)) / np.std(self.x)
        self.y = (self.y-np.mean(self.y)) / np.std(self.y)


    def find_marginals(self):
        """ Not sure exactñy what this does yet. """
        for i, x in enumerate(self.x):
            self.y_marginals[x].append(self.y[i])

        for j, y in enumerate(self.y):
            self.x_marginals[y].append(self.x[j])

    def return_stochastic_complexity(self, data):
        """ This method returns the stochastic complexity
        of the input data according to the multinomial
        model class. """
        freqs = Counter(data)
        n, L = len(data), len(freqs)
        loglikelihood = 0.0
        for freq in freqs.values():
            loglikelihood += freq * (np.log(n) - np.log(freq))

        normalising_term = multinomial_with_recurrence(L, n)

        stochastic_complexity = loglikelihood + np.log(normalising_term)

        return stochastic_complexity

    def return_conditional_stochastic_complexity(self, marginal_data):
        """ This method returns the conditional stochastic complexity
        of x given y or y given x according to the multinomial model
        class. """
        return np.sum(self.return_stochastic_complexity(z) for z in
                   marginal_data.values())

    def compute_cisc(self):
        """ This method infers the causal direction. It also returns
        a confidence score and a signficance score. """
        self.sc_x = self.return_stochastic_complexity(self.x)
        self.sc_y = self.return_stochastic_complexity(self.y)
        self.sc_y_cond_x = self.return_conditional_stochastic_complexity(
                                                        self.y_marginals)
        self.sc_x_cond_y = self.return_conditional_stochastic_complexity(
                                                        self.x_marginals)
        self.x_causes_y = (self.sc_x + self.sc_y_cond_x)
        self.y_causes_x = (self.sc_y + self.sc_x_cond_y)
        self.likelihood_ratio = 0 if self.x_causes_y == self.y_causes_x else self.x_causes_y - self.y_causes_x
        self.confidence = np.abs(self.likelihood_ratio)
        self.significance = np.power(2, -np.abs(
                                    self.confidence)/2)
