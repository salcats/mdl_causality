"""Causal inference on discrete data using stochastic complexity of multinomial.
"""
import numpy as np
from collections import defaultdict
from collections import Counter
from utils import multinomial_with_recurrence, log2

__author__ = "Kailash Budhathoki"
__email__ = "kbudhath@mpi-inf.mpg.de"
__copyright__ = "Copyright (c) 2017"
__license__ = "MIT"


class CISC():

    def __init__(self, x, y, scale_factor):
        """ Initialise the class with the data. """
        self.x = x
        self.y = y
        self.scale_factor = scale_factor
        self.y_marginals = defaultdict(self.x)
        self.x_marginals = defaultdict(self.y)
        self.preprocess_data()
        self.find_marginals()
        self.sc_x = None
        self.sc_y = None
        self.sc_x_cond_y = None
        self.sc_y_cond_x = None
        self.x_causes_y = None
        self.y_causes_x = None
        self.confidence = None
        self.significance = None
        self.compute_cisc()

    def preprocess_data(self):
        """ This method preprocesses the data.
        Current implementation is very simple. """
        self.x = np.multiply(self.x, self.scale_factor).astype(np.int)
        self.y = np.multiply(self.y, self.scale_factor).astype(np.int)

    def find_marginals(self):
        """ Not sure exactñy what this does yet. """
        for i, x in enumerate(self.x):
            self.y_marginals[self.x].append(self.y[i])

        for j, y in enumerate(self.y):
            self.x_marginals[self.y].append(self.x[i])

    def return_stochastic_complexity(self, data):
        """ This method returns the stochastic complexity
        of the input data according to the multinomial
        model class. """
        freqs = Counter(data)
        n, L = len(data), len(freqs)
        loglikelihood = 0.0
        for freq in freqs.values():
            loglikelihood += freq * (log2(n) - log2(freq))

        normalising_term = multinomial_with_recurrence(L, n)

        stochastic_complexity = loglikelihood + log2(normalising_term)

        return stochastic_complexity

    def return_conditional_stochastic_complexity(self, marginal_data):
        """ This method returns the conditional stochastic complexity
        of x given y or y given x according to the multinomial model
        class. """
        return sum(self.find_stochastic_complexity(z) for z in
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

        normalising_factor = self.sc_x + self.sc_y
        self.x_causes_y = (self.sc_x + self.sc_y_cond_x)/normalising_factor
        self.y_causes_x = (self.sc_y + self.sc_x_cond_y)/normalising_factor

        self.confidence = np.abs(self.x_causes_y - self.y_causes_x)
        self.signficance = np.power(2, -np.abs(
                                    self.confidence * normalising_factor)/2)
