from math import ceil, e, factorial, log, pi, sqrt, isinf
from __future__ import division

fact = lambda n: factorial(n)


def composition(n):
    half = int(n / 2)
    for i in range(half + 1):
        r1, r2 = i, n - i
        yield r1, r2


def multinomial_with_recurrence(L, n):
    total = 1.0
    b = 1.0
    d = 10
    bound = int(ceil(2 + sqrt(2 * n * d * log(10))))
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


def log2(n):
    return log(n or 1, 2)
