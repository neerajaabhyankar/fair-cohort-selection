import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def plot_objective_values(eval_objectives, eval_ground, setcolor, groundcolor=None):
    """
    As we greedily build out the set,
    we plot the value of the objective f(S_t)
    at every stage t
    :param eval_objectives: f(S_t) for 0 \leq t < k
    :param eval_ground: f(V)
    :param setcolor: color to plot f(S_t) with
    :param groundcolor: color to plot f(V) with
    """
    k = len(eval_objectives)
    plt.plot(np.arange(k), eval_objectives, "o", c=setcolor, label="$f(S_t)$")
    if groundcolor is not None:
        plt.plot(np.arange(k), [eval_ground,]*k, "--", c=groundcolor, label="$f(V)$")
    plt.xlabel("$|S_t|$")
    plt.legend(loc=2)


def plot_membership_histogram(
    memberships, budgets,
    S,
    setcolor, groundcolor,
    budgetlabel="budget", setlabel="selection", value=-100
):
    """ Given a membership assignment matrix (n x p) and optionally group budgets (min or max),
        A selection of indices S of size k < n
        Plot the selection's group-wise distribution
        :param memberships: n rows (members), p columns (0 or 1 membership bools)
        :param budgets: indices in [n], length k
        :param setcolor: color to plot f(S_t) with
        :param groundcolor: color to plot f(V) with
    """
    
    [n, p] = memberships.shape
    k = len(S)
    
    Vcounts = np.sum(memberships, axis=0)
    Scounts = np.sum(memberships[S], axis=0)
    if budgetlabel == "quotas":
        assert budgets is not None
        unmet = np.where(Scounts < budgets)[0]
    elif budgetlabel == "capacities":
        assert budgets is not None
        unmet = np.where(Scounts > budgets)[0]
    else:
        unmet = None
    
    plt.bar(np.arange(p), Vcounts, color=groundcolor, label="all")  # all members
    plt.bar(np.arange(p), Scounts, color=setcolor, label=setlabel)  # selected members
    if budgets is not None:
        plt.scatter(np.arange(p), budgets, c="white", s=16, zorder=8, label=budgetlabel)  # all budgets
    if unmet is not None:
        plt.scatter(unmet, budgets[unmet], c="red", s=2, zorder=10, label="unmet "+budgetlabel)  # unmet budgets
    plt.xlabel("groups")
    plt.ylabel("membership counts")
    if value >= 0:
        plt.title(f"objective value = {value}")
    plt.legend()
