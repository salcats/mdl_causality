#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Computes the stochastic complexity of multinomial distribution.
http://www.sciencedirect.com/science/article/pii/S0020019007000944
"""
from __future__ import division
from collections import Counter
from math import ceil, e, factorial, log, pi, sqrt, isinf

__author__ = "Kailash Budhathoki"
__email__ = "kbudhath@mpi-inf.mpg.de"
__copyright__ = "Copyright (c) 2017"
__license__ = "MIT"


log2 = lambda n: log(n or 1, 2)
fact = lambda n: factorial(n)


def composition(n):
    half = int(n / 2)
    for i in range(half + 1):
        r1, r2 = i, n - i
        yield r1, r2


def multinomial_with_recurrence(L, n):
    total = 1.0
    b = 1.0
    d = 10   # seven digit precision
    bound = int(ceil(2 + sqrt(2 * n * d * log(10))))  # using equation (38)
    for k in range(1, bound + 1):
        b = (n - k + 1) / n * b
        total += b

    old_sum = 1.0
    for j in range(3, L + 1):
        new_sum = total + (n * old_sum) / (j - 2)
        old_sum = total
        total = new_sum

    if L == 1:
        total = 1.0

    return total

def stochastic_complexity(X):
    freqs = Counter(X)
    n, L = len(X), len(freqs)
    loglikelihood = 0.0
    for freq in freqs.values():
        loglikelihood += freq * (log2(n) - log2(freq))

    normalising_term = multinomial_with_recurrence(L, n)

    sc = loglikelihood + log2(normalising_term)

    return sc


def stochastic_complexity_slow(X):
    freqs = Counter(X)
    n, K = len(X), len(freqs)
    likelihood = 1.0
    for freq in freqs.values():
        likelihood *= (freq / n) ** freq

    prev1 = 1.0
    prev2 = 0.0
    n_fact = fact(n)
    for r1, r2 in composition(n):
        prev2 += (n_fact / (fact(r1) * fact(r2))) * \
            ((r1 / n) ** r1) * ((r2 / n) ** r2)

    for k in range(1, K - 1):
        temp = prev2 + n * prev1 / k
        prev1, prev2 = prev2, temp

    loglikelihood = 0.0
    for freq in freqs.values():
        loglikelihood += freq * (log2(n) - log2(freq))

    sc = loglikelihood + log2(prev2)
    return sc
