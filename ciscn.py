"""Causal inference on discrete data using stochastic complexity of
normal model class.
"""
import numpy as np
from utils import neg_log_mle, log_parametric_complexity
from utils import var_x_cond_y, var_y_cond_x


class CISCN():

    def __init__(self, x, y, m):
        """ Initialise the class with the data. """
        self.x = x
        self.y = y
        self.m = m
        self.n = len(x)
        self.preprocess_data()
        self.corr_xy = np.corrcoef(self.x, self.y)[0, 1]
        self.mean_x = np.mean(self.x)
        self.mean_y = np.mean(self.y)
        self.var_x = np.var(self.x)
        self.var_y = np.var(self.y)
        self.sc_x = None
        self.sc_y = None
        self.sc_x_cond_y = None
        self.sc_y_cond_x = None
        self.x_causes_y = None
        self.y_causes_x = None
        self.confidence = None
        self.normalised_confidence = None
        self.likelihood_ratio = None
        self.significance = None
        self.normalised_significance = None

    def preprocess_data(self):
        """ This method preprocesses the data. """
        # Normalise the data
        self.x = (np.max(self.x) - self.x) / (np.max(self.x)-np.min(self.x))
        self.y = (np.max(self.y) - self.y) / (np.max(self.y)-np.min(self.y))
        # Standardise the data
        self.x = (self.x-np.mean(self.x)) / np.std(self.x)
        self.y = (self.y-np.mean(self.y)) / np.std(self.y)

    def return_stochastic_complexity(self, data, mean, variance):
        """ This method returns the stochastic complexity
        of the input data according to the Normal
        model class. """

        log_loss = neg_log_mle(data, mean, variance, self.n)
        log_pc = log_parametric_complexity(variance, mean, self.m, self.n)

        stochastic_complexity = log_loss + log_pc

        return stochastic_complexity

    def compute_sc_y_cond_x(self):
        """ This method returns the conditional stochastic complexity
        of x conditioned y according to the Normal model
        class. """
        var_y_c_x = var_y_cond_x(self.var_y, self.corr_xy)

        x_1 = np.min(self.x)
        x_2 = np.max(self.x)

        c_1 = (self.corr_xy*self.var_y)/self.var_x
        c_2 = c_1*self.mean_x
        c_3 = 1/(2*var_y_c_x)
        c_4 = (x_2 - x_1)
        n_half = self.n/2

        I_21_term_1 = c_3*c_4
        I_21_term_2 = np.sum(np.power(self.y, 2))
        I_21_term_3 = -2*(self.mean_y - c_2)*np.sum(self.y)
        I_21_term_4 = self.n*(np.power(self.mean_y, 2) - 2*self.mean_y*c_2
                              + np.power(c_2, 2))
        I_21 = I_21_term_1*(I_21_term_2 + I_21_term_3 + I_21_term_4 + np.sum(
                            np.power((var_y_c_x-self.var_y)/self.m, 2)))

        I_22_term_1 = c_3*c_1*(np.sum(self.y) + self.n*self.mean_y - c_2)*(
                      np.power(x_2, 2) - np.power(x_1, 2))
        I_22_term_2 = c_3*np.power(c_1, 2)*(1/3)*(np.power(x_2, 3)
                                                  - np.power(x_1, 3))
        I_22 = I_22_term_1 + I_22_term_2

        I_1_term_1 = c_4*n_half*np.log(2*np.pi*var_y_c_x)
        I_1_term_2 = c_4*log_parametric_complexity(self.var_y, self.mean_y,
                                                   self.m, self.n)
        I_1 = I_1_term_1 + I_1_term_2

        self.sc_y_cond_x = I_1 + I_21 + I_22

    def compute_sc_x_cond_y(self):
        """ This method returns the conditional stochastic complexity
        of y conditioned on x according to the Normal model
        class. """
        var_x_c_y = var_x_cond_y(self.var_x, self.corr_xy)

        y_1 = np.min(self.y)
        y_2 = np.max(self.y)

        c_1 = (self.corr_xy*self.var_x)/self.var_y
        c_2 = c_1*self.mean_y
        c_3 = 1/(2*var_x_c_y)
        c_4 = (y_2 - y_1)

        n_half = self.n/2

        I_21_term_1 = c_3*c_4
        I_21_term_2 = np.sum(np.power(self.x, 2))
        I_21_term_3 = -2*(self.mean_x - c_2)*np.sum(self.x)
        I_21_term_4 = self.n*(np.power(self.mean_x, 2) - 2*self.mean_x*c_2
                              + np.power(c_2, 2))
        I_21 = I_21_term_1*(I_21_term_2 + I_21_term_3 + I_21_term_4
                            + np.sum(np.power((var_x_c_y-self.var_x)
                                     / self.m, 2)))
        I_22_term_1 = c_3*c_1*(np.sum(self.x) + self.n*self.mean_x - c_2
                               )*(np.power(y_2, 2) - np.power(y_1, 2))
        I_22_term_2 = c_3*np.power(c_1, 2)*(1/3)*(np.power(y_2, 3)
                                                  - np.power(y_1, 3))
        I_22 = I_22_term_1 + I_22_term_2

        I_1_term_1 = c_4*n_half*np.log(2*np.pi*var_x_c_y)
        I_1_term_2 = c_4*log_parametric_complexity(self.var_x, self.mean_x,
                                                   self.m, self.n)
        I_1 = I_1_term_1 + I_1_term_2

        self.sc_x_cond_y = I_1 + I_21 + I_22

    def compute_cisc(self):
        """ This method infers the causal direction. It also returns
        a confidence score and a signficance score. """
        self.sc_x = self.return_stochastic_complexity(self.x, self.mean_x,
                                                      self.var_x)
        self.sc_y = self.return_stochastic_complexity(self.y, self.mean_y,
                                                      self.var_y)
        self.compute_sc_y_cond_x()
        self.compute_sc_x_cond_y()

        normalising_factor = self.sc_x + self.sc_y
        self.x_causes_y = (self.sc_x + (1/self.n)*self.sc_y_cond_x)
        self.y_causes_x = (self.sc_y + (1/self.n)*self.sc_x_cond_y)
        self.likelihood_ratio = self.x_causes_y - self.y_causes_x
        self.confidence = np.abs(self.x_causes_y - self.y_causes_x)
        self.normalised_confidence = np.abs(self.x_causes_y - self.y_causes_x
                                            )/normalising_factor
        self.normalised_significance = np.power(2, -np.abs(
                                    self.normalised_confidence)/2)
        self.significance = np.power(2, -np.abs(
                                    self.confidence)/2)
