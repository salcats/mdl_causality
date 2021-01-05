from __future__ import division
import numpy as np
import os, re, glob
from math import ceil, e, factorial, log, pi, sqrt, isinf, isnan
from scipy.special import erf
from matplotlib import pyplot as plt
import matplotlib

fact = lambda n: factorial(n)
plt.style.use('ggplot')

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


def neg_log_mle(x, mu, var, n):
    """ This function computes the negative log likelihood of the
    maximum likelihood estimator for the Normal distribution. """
    n_half = n/2
    var_half_inv = (1/(2*var))

    return (n_half*np.log(2*np.pi*var) + var_half_inv*np.sum(
                                                          np.power(x-mu, 2)))


def log_parametric_complexity(var, mu, m, n):
    """ This function computes the log parametric complexity of the
    Normal distribution premultiplied by the luckiness function. """

    n_half = n/2
    term_1 = n_half*np.log(n_half)
    term_2 = -n_half
    term_3 = np.log(np.sqrt(np.pi)*np.abs(m)*(erf(var/np.abs(m))+1))
    term_4 = np.log(2*var)
    term_5 = term_1 - n_half + 0.5*np.log((2*np.pi)/n_half)

    return term_1+term_2+term_3-term_4-term_5


def var_x_cond_y(var_x, corr_xy):
    """ This function computes variance of x conditional on y. """
    return (1-np.power(corr_xy, 2))*var_x


def var_y_cond_x(var_y, corr_xy):
    """ This function computes variance of y conditional on x. """
    return (1-np.power(corr_xy, 2))*var_y


def mu_x_cond_y(mu_x, mu_y, var_x, var_y, corr_xy, y):
    """ This function computes mean of x conditional on y. """

    return mu_x + (np.sqrt(var_x)/np.sqrt(var_y))*corr_xy*(y-mu_y)


def mu_y_cond_x(mu_x, mu_y, var_x, var_y, corr_xy, x):
    """ This function computes mean of y conditional on x. """

    return mu_y + (np.sqrt(var_y)/np.sqrt(var_x))*corr_xy*(x-mu_x)


def plot_multiline(Ys, X, labels, xlabel, ylabel, title, fig_name=None):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i, Y in enumerate(Ys):
        plt.plot(X, Y, label=labels[i], linewidth=2.0,
                 dash_joinstyle="bevel", solid_capstyle="round", markeredgecolor="none", linestyle="-")
        plt.legend(loc="upper left")
    plt.ylim(0.0, 1.001)
    if fig_name:
        plt.savefig("results/%s" % fig_name)
    else:
        plt.show()
    plt.cla()


def reverse_argsort(X):
    indices = list(range(len(X)))
    indices.sort(key=X.__getitem__, reverse=True)
    return indices


def plot_confidence_tcep(results, alpha, normalised=False):

    results[:, 0] = results[:, 0] == 1
    if normalised:
        indices = reverse_argsort(results[:, 2])
        sorted_confidence = np.asarray(([results[:, 2][i] for i in indices]))
    else:
        indices = reverse_argsort(results[:, 1])
        sorted_confidence = np.asarray(([results[:, 1][i] for i in indices]))

    sorted_significance = np.asarray(([results[:, 3][i] for i in indices]))
    sorted_labels = np.asarray(([results[:, 0][i] for i in indices]))

    cmap = matplotlib.colors.ListedColormap(['red', 'green'])

    fig, ax = plt.subplots(figsize=(12, 4))

    x = np.linspace(1, len(sorted_confidence), len(sorted_confidence))

    ax.scatter(x, sorted_confidence,
               s = 2, c=sorted_labels, cmap=cmap)

    significance_mask = np.asarray((sorted_significance)) >= alpha

    x_ann = x[significance_mask]
    y_ann = sorted_confidence[significance_mask]

    arrow_length = 0.1*np.max(sorted_confidence)
    print (arrow_length)

    for i in range(len(x_ann)):
        ax.annotate('', xy=(x_ann[i], y_ann[i]),
                    xytext=(x_ann[i], y_ann[i] + arrow_length),
                    arrowprops=dict(linewidth=0.00001, facecolor='black'))

    plt.show()



def plot_accuracy_with_decision_rate_tcep(benchmark_test, results,
                                          confidence=True, normalised=False):

    results[:, 0] = results[:, 0] == 1

    if confidence:
        if normalised:
            indices = reverse_argsort(results[:, 2])
            sorted_confidence = np.asarray(([results[:, 2][i] for i in indices]))
            sorted_labels = np.asarray(([results[:, 0][i] for i in indices]))
        else:
            indices = reverse_argsort(results[:, 1])
            sorted_confidence = np.asarray(([results[:, 1][i] for i in indices]))
            sorted_labels = np.asarray(([results[:, 0][i] for i in indices]))
    else:
        if normalised:
            indices = reverse_argsort(results[:, 4])
            sorted_confidence = np.asarray(([results[:, 4][i] for i in indices]))
            sorted_labels = np.asarray(([results[:, 0][i] for i in indices]))
        else:
            indices = reverse_argsort(results[:, 3])
            sorted_confidence = np.asarray(([results[:, 3][i] for i in indices]))
            sorted_labels = np.asarray(([results[:, 0][i] for i in indices]))

    accuracy = np.sum(results[:, 0])/benchmark_test.no_pairs
    dec_rate = np.arange(0.02, 1.01, 0.01)
    accs = []
    for r in dec_rate:
        maxIdx = int(r * benchmark_test.no_pairs)
        labels = sorted_labels[:maxIdx]
        accs.append(sum(labels) / len(labels))

    plot_multiline([accs], dec_rate, [
                    "COMBINED: " + str(np.round(accuracy, 4))], "decision rate",
                   "accuracy", "decision rate versus accuracy")


def plot_conf_v_p_value(results, normalised=False, log=False):

    if normalised:
        conf = [x for x in results[:, 2]]
        pv = [x for x in results[:, 3]]
    else:
        conf = [x for x in results[:, 1]]
        pv = [x for x in results[:, 3]]

    if log:
        plt.scatter(conf, -np.log(pv))
        plt.ylabel('-log p-value')
    else:
        plt.scatter(conf, (pv))
        plt.ylabel('p-value')
    plt.xlabel('Confidence')
    plt.show()
