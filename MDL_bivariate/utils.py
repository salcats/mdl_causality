from __future__ import division
import numpy as np
import os, re, glob
import pandas as pd
from math import ceil, e, factorial, log, pi, sqrt, isinf, isnan
from scipy.special import erf
from matplotlib import pyplot as plt
from cdt.data import CausalPairGenerator
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


def log_parametric_complexity(var, m, n):
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

def plot_single(Ys, X, labels, xlabel, ylabel, title, fig_name=None):

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

def plot_multiline(Ys, X, labels, xlabel, ylabel, title, fig_name=None):

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i, (k, v) in enumerate(Ys.items()):
        plt.plot(X, v, label=labels[i], linewidth=2.0,
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


def plot_confidence_tcep(results, alpha):

    results[:, 0] = results[:, 0] == 1
    indices = reverse_argsort(results[:, 1])
    sorted_confidence = np.asarray(([results[:, 1][i] for i in indices]))

    sorted_significance = np.asarray(([results[:, 2][i] for i in indices]))
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
    '''
    for i in range(len(x_ann)):
        ax.annotate('', xy=(x_ann[i], y_ann[i]),
                    xytext=(x_ann[i], y_ann[i] + arrow_length),
                    arrowprops=dict(width=1,
                                    headwidth=5,
                                    facecolor='black'))
    '''
    plt.show()


def plot_accuracy_with_decision_rate_tcep(benchmark_test, results,
                                          confidence=True):

    results[:, 0] = results[:, 0] == 1

    if confidence:
        indices = reverse_argsort(results[:, 1])
        sorted_labels = np.asarray(([results[:, 0][i] for i in indices]))
    else:
        indices = reverse_argsort(results[:, 2])
        sorted_labels = np.asarray(([results[:, 0][i] for i in indices]))

    accuracy = np.sum(results[:, 0])/benchmark_test.no_pairs
    dec_rate = np.arange(0.02, 1.01, 0.01)
    accs = []
    for r in dec_rate:
        maxIdx = int(r * benchmark_test.no_pairs)
        labels = sorted_labels[:maxIdx]
        accs.append(sum(labels) / len(labels))

    plot_single([accs], dec_rate, [
                    "COMBINED: " + str(np.round(accuracy, 4))],
                    "decision rate",
                    "accuracy",
                    "decision rate versus accuracy")


def plot_conf_v_p_value(results, log=False):

    conf = [x for x in results[:, 1]]
    pv = [x for x in results[:, 2]]

    if log:
        plt.scatter(conf, -np.log(pv))
        plt.ylabel('-log p-value')
    else:
        plt.scatter(conf, (pv))
        plt.ylabel('p-value')
    plt.xlabel('Confidence')
    plt.show()

xy_TRUTH_PATTERN = re.compile("x\s*-+\s*-*\s*>\s*y", re.IGNORECASE)
yx_TRUTH_PATTERN = re.compile("y\s*-+\s*-*\s*>\s*x", re.IGNORECASE)


def extract_cep_truths():
    cep_pairs_dir = os.path.join(os.path.dirname(
        __file__), "data", "cep")
    cep_pairs = glob.glob(
        cep_pairs_dir + os.sep + "pair[0-9][0-9][0-9][0-9]_des.txt")

    truths = []
    for pair in cep_pairs:
        with open(pair, "r") as fp:
            desc = fp.read()
            if xy_TRUTH_PATTERN.findall(desc):
                truths.append(("x", "y"))
            else:
                truths.append(("y", "x"))
    return truths


def extract_cep_data():

    truths = extract_cep_truths()
    cep_pairs_dir = os.path.join(os.path.dirname(
        __file__), "data", "cep")
    cep_pairs = glob.glob(
        cep_pairs_dir + os.sep + "pair[0-9][0-9][0-9][0-9].txt")
    multivariate_pairs = [51, 52, 53, 54, 70, 101, 102, 103, 104, 105, 106, 107]
    cep_df = pd.DataFrame(columns = ['A', 'B', 'weights'],
                          index = range(len(cep_pairs)))
    weights = np.loadtxt(cep_pairs_dir + os.sep + "pairmeta.txt")[:, 5]

    for pair in range(len(cep_pairs)):
        xy_data = np.loadtxt(cep_pairs[pair])
        x = xy_data[:, 0]
        y = xy_data[:, 1]

        if truths[pair] == ("x", "y"):
            cep_df['A'][pair] = x
            cep_df['B'][pair] = y
        else:
            cep_df['A'][pair] = y
            cep_df['B'][pair] = x

        cep_df['weights'][pair] = weights[pair]

    cep_df = cep_df.drop(multivariate_pairs, axis = 0)
    cep_df = cep_df.reset_index(drop=True)

    return cep_df


def extract_sim_data(data_type):

    sim_pairs_dir = os.path.join(os.path.dirname(
        __file__), "data", data_type)
    sim_pairs = glob.glob(
        sim_pairs_dir + os.sep + "pair[0-9][0-9][0-9][0-9].txt")
    sim_df = pd.DataFrame(columns = ['A', 'B'],
                          index = range(len(sim_pairs)))

    meta_data = np.loadtxt(sim_pairs_dir + os.sep + "pairmeta.txt")
    for pair in range(len(sim_pairs)):
        xy_data = np.loadtxt(sim_pairs[pair])
        if meta_data[pair][1] == 1:
            sim_df['A'][pair] = xy_data[:, 0]
            sim_df['B'][pair] = xy_data[:, 1]
        else:
            sim_df['A'][pair] = xy_data[:, 1]
            sim_df['B'][pair] = xy_data[:, 0]

    return sim_df


def extract_other_sim_data(data_type):

    sim_pairs_dir = os.path.join(os.path.dirname(
        __file__), "data", data_type)
    sim_pairs = glob.glob(
        sim_pairs_dir + os.sep + "pair_[0-9].txt")
    sim_pairs = sim_pairs + glob.glob(
        sim_pairs_dir + os.sep + "pair_[0-9][0-9].txt")

    sim_df = pd.DataFrame(columns = ['A', 'B'],
                          index = range(len(sim_pairs)))

    meta_data = np.loadtxt(sim_pairs_dir + os.sep + "pairs_gt.txt")
    for pair in range(len(sim_pairs)):
        xy_data = np.loadtxt(sim_pairs[pair], delimiter=',', skiprows=1,
                              usecols= (1, 2))

        if meta_data[pair] == 1:
            sim_df['A'][pair] = xy_data[:, 0]
            sim_df['B'][pair] = xy_data[:, 1]
        else:
            sim_df['A'][pair] = xy_data[:, 1]
            sim_df['B'][pair] = xy_data[:, 0]

    return sim_df


def generate_data(no_points, noise_coeff, mechanism):

    generator = CausalPairGenerator(mechanism, noise_coeff=noise_coeff)
    data, labels = generator.generate(100, npoints=no_points, rescale=True)

    for i in range(0, 100):
        if labels['label'][i] == -1:
            x = data['A'][i]
            y = data['B'][i]
            data['A'][i] = y
            data['B'][i] = x

    return data


def plot_accuracy_comparison(benchmark_test,
                             causes,
                             scores,
                             methods,
                             use_weights=False):

    causes = causes == 1
    if use_weights == True:
        weights = benchmark_test.data['weights']
    else:
        weights = np.ones((len(benchmark_test.data['A'])))
    dec_rate = np.arange(0.02, 1.01, 0.01)
    sorted_accs = dict.fromkeys(methods)
    overall_acc_strings = []

    for m_ind in range(len(methods)):
        indices = reverse_argsort(scores[:, m_ind])
        sorted_causes = np.asarray(([causes[:, m_ind][i] for i in indices]))
        sorted_weights = np.asarray(([weights[i] for i in indices]))

        accuracy = np.sum(causes[:, m_ind]*weights)/np.sum(weights)
        accs = []
        for r in dec_rate:
            maxIdx = int(r * benchmark_test.no_pairs)
            top_k_causes = sorted_causes[:maxIdx]
            top_k_weights = sorted_weights[:maxIdx]
            accs.append(np.sum(top_k_causes*top_k_weights) / np.sum(top_k_weights))

        sorted_accs[methods[m_ind]] = accs
        overall_acc_strings.append(' ' + str(np.round(accuracy, 4)))
    methods = np.char.add(methods, overall_acc_strings)
    plot_multiline(sorted_accs, dec_rate,
                    methods,
                    "decision rate",
                    "accuracy",
                    "decision rate versus accuracy")
