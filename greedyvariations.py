import os
import math
import numpy as np


def random_vanilla(oracle, k):
    """ 
    For a given ground set, a feature matrix and mixture weights which define the submodular objective,
    Returns the greedy selection and step-wise objective values
    Over the addition of k items
    :param k: total capacity
    """
    
    V = np.arange(oracle.n).tolist()
    objs = []
    objs.append(0)
    
    for kk in range(k):
        
        greedy_winner = np.random.choice(V)
        
        # move winner from V to A
        V.remove(greedy_winner)
        oracle.add_element(greedy_winner)
        objs.append(oracle.current_objective)
    
    return oracle.set, objs


def greedy_vanilla(oracle, k):
    """ 
    For a given ground set, a feature matrix and mixture weights which define the submodular objective,
    Returns the greedy selection and step-wise objective values
    Over the addition of k items
    :param k: total capacity
    """
    
    V = np.arange(oracle.n).tolist()
    objs = []
    objs.append(0)
    
    for kk in range(k):
        
        gains = [oracle.compute_gain(ii) for ii in V]
        greedy_winner = V[np.argmax(gains)]
        
        # move winner from V to A
        V.remove(greedy_winner)
        oracle.add_element(greedy_winner)
        objs.append(oracle.current_objective)
    
    return oracle.set, objs


def greedy_quota(oracle, memberships, unfilled_quotas, k, verbose=False):
    """
    :param memberships: n rows (members), p columns (0 or 1 membership bools)
    :param unfilled_quotas: indices in [n], numpy array of length p
    :param k: total capacity
    """

    assert memberships.shape[0] == oracle.X.shape[0]
    assert memberships.shape[1] == len(unfilled_quotas)
    [n, p] = memberships.shape
    
    V = np.arange(oracle.n).tolist()
    objs = []
    objs.append(0)
    
    for jj in range(p):
        if np.sum(memberships[:,jj]) < unfilled_quotas[jj]:
            print("Not enough members in group {}, infeasible problem.".format(jj))
            return None, None
    
    """ Quota-filling stage """
    
    unfilled_groups = np.where(np.array(unfilled_quotas)>0)[0]
    V_quota_candidates = [ii for ii in V if np.sum(memberships[ii][unfilled_groups]) > 0]
    
    while np.sum(unfilled_quotas) > 0:
        
        unfilled_groups = np.where(unfilled_quotas>0)[0]
        V_quota_candidates = [ii for ii in V if np.sum(memberships[ii][unfilled_groups]) > 0]
        if len(V_quota_candidates) <= 0:
            # we ran out of quota candidates
            break
        
        gains = [oracle.compute_gain(ii) for ii in V_quota_candidates]
        greedy_winner = V_quota_candidates[np.argmax(gains)]
        
        # move winner from V to A
        V.remove(greedy_winner)
        oracle.add_element(greedy_winner)
        objs.append(oracle.current_objective)
        
        # decrement unfilled quotas
        unfilled_quotas = unfilled_quotas - memberships[greedy_winner]
        unfilled_quotas.clip(0, k)
        
        if verbose:
            print("unfilled_quotas = ", unfilled_quotas)
    
    """ Regular greedy stage """
    
    for kk in range(k - len(oracle.set)):
        
        gains = [oracle.compute_gain(ii) for ii in V]
        greedy_winner = V[np.argmax(gains)]
        
        # move winner from V to A
        V.remove(greedy_winner)
        oracle.add_element(greedy_winner)
        objs.append(oracle.current_objective)
    
    return oracle.set, objs


def greedy_capacity(oracle, memberships, unfilled_capacities, k, verbose=False):
    """
    :param memberships: n rows (members), p columns (0 or 1 membership bools)
    :param unfilled_capacities: indices in [n], numpy array of length p
    :param k: total capacity
    """

    assert memberships.shape[0] == oracle.X.shape[0]
    assert memberships.shape[1] == len(unfilled_capacities)
    [n, p] = memberships.shape
    
    V = np.arange(oracle.n).tolist()
    objs = []
    objs.append(0)
    
    unfilled_groups = np.where(np.array(unfilled_capacities)>0)[0]
    V_capacity_candidates = [ii for ii in V if np.sum(memberships[ii][unfilled_groups]) > 0]
    
    for kk in range(k):
        
        unfilled_groups = np.where(unfilled_capacities>0)[0]
        V_capacity_candidates = [ii for ii in V if np.sum(memberships[ii][unfilled_groups]) > 0]
        if len(V_capacity_candidates) <= 0:
            # we ran out of candidates
            return oracle.set, objs
        
        gains = [oracle.compute_gain(ii) for ii in V_capacity_candidates]
        greedy_winner = V_capacity_candidates[np.argmax(gains)]
        
        # move winner from V to A
        V.remove(greedy_winner)
        oracle.add_element(greedy_winner)
        objs.append(oracle.current_objective)
        
        # decrement unfilled quotas
        unfilled_capacities = unfilled_capacities - memberships[greedy_winner]
        unfilled_capacities.clip(0, k)
        
        if verbose:
            print("unfilled_capacities = ", unfilled_capacities)
    
    return oracle.set, objs
