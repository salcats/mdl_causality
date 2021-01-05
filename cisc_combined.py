"""Causal inference on discrete data using stochastic complexity
of multinomial or normal model class.
"""

import numpy as np
from ciscn import CISCN
from ciscm import CISCM
from scipy.stats import normaltest


class CISCC():
    """ This class combines ciscn and ciscm to infer causal
    direction between x and y. It chooses to use ciscn or
    ciscm based on characteristics of the data. """

    def __init__(self, scale_factor, m):

        self.x = None
        self.y = None
        self.scale_factor = scale_factor
        self.m = m
        self.conditions = np.zeros((5, 1))

    def add_data(self, x, y):
        self.x = x
        self.y = y

    def compute_handmade_conditions(self,
                                    normality_alpha=0.001,
                                    data_size=2000,
                                    correlation=0.75,
                                    var_diff=1e4):
        """ This method computes the value of the conditionals
        used to determine which model class to use. These conditions
        were established by hand. """

        k2, p_x = normaltest(self.x)
        k2, p_y = normaltest(self.y)
        corr_xy = np.abs(np.corrcoef(self.x, self.y)[0, 1])

        condition_1 = p_x > normality_alpha or p_y > normality_alpha
        condition_2 = len(self.x) > data_size
        condition_3 = (self.x[0].is_integer() and \
                        not(self.y[0].is_integer())) or \
                        (self.y[0].is_integer() and not(self.x[0].is_integer()))
        condition_4 = corr_xy > correlation
        condition_5 = np.abs(np.var(self.x)-np.var(self.y)) > var_diff and \
                        not(self.x[0].is_integer() and self.y[0].is_integer())

        self.conditions[0] = condition_1
        self.conditions[1] = condition_2
        self.conditions[2] = condition_3
        self.conditions[3] = condition_4
        self.conditions[4] = condition_5

    def compute_cisc(self):
        """ This method infers the causal direction. It chooses which
        model class to encode the data with and then computes the cisc
        score with that model class.
        0 = Multinomial
        1 = Normal
        """
        self.compute_handmade_conditions()

        ciscm_model = CISCM(self.x, self.y, self.scale_factor)
        ciscn_model = CISCN(self.x, self.y, self.m)

        ciscm_model.compute_cisc()
        ciscn_model.compute_cisc()

        model_class_choice = 0

        if np.isnan(ciscm_model.normalised_confidence):
            model_class_choice = 1
        else:
            confidence_scores = [ciscm_model.normalised_confidence, ciscn_model.normalised_confidence]
            ind = np.argmax(confidence_scores)
            if ind == 0:
                model_class_choice = 0
            else:
                model_class_choice = 1

        if self.conditions[0]:
            model_class_choice = 1
        if self.conditions[1]:
            model_class_choice = 1
        if self.conditions[2]:
            model_class_choice = 0
        if self.conditions[3]:
            model_class_choice = 1
        if self.conditions[4]:
            model_class_choice = 1

        if model_class_choice == 0:
            if ciscm_model.likelihood_ratio < 0:
                cause = 1
            else:
                cause = 2
            output = [cause, ciscm_model.confidence, ciscm_model.normalised_confidence,
                      ciscm_model.significance]
        else:
            if ciscn_model.likelihood_ratio < 0:
                cause = 1
            else:
                cause = 2

            output = [cause, ciscn_model.confidence, ciscn_model.normalised_confidence,
                    ciscn_model.significance]

        '''
        if ciscm_model.likelihood_ratio < 0:
            output = [1, ciscm_model.confidence, ciscm_model.normalised_confidence,
                                  ciscm_model.significance]
        elif ciscn_model.likelihood_ratio < 0:
            output = [cause, ciscn_model.confidence, ciscn_model.normalised_confidence,
            ciscn_model.significance]
        More experiments with how to choose the right model class.
        '''
        return output
