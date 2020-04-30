#%% Imports

import os
import math

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#%% Marginal gain oracle of the utility

def submodgains(X, modA, fA, aa, mixw):
    """ Returns f(A+a) - f(A)
        Where f(A) = \sum_{j=1}^m (w_j * \phi(\sum_{i \in A} X_{ij}))
    """
    
    modAa = modA + X[aa,:] if aa is not None else modA
    
    # options: modA**0.5, np.log(1+modA), (1-np.exp(-modA)), modA/(1+modA)
    fAa = np.dot(mixw, modAa**0.2)
    
    return fAa - fA


#%% Plain old simple Greedy

def greedygains_submod(V, X, mixw, k):
    """ For a given ground set, a feature matrix and mixture weights which define the objective
        (submodular),
        Returns the greedy selection and step-wise objective values
        Over the addition of k items
    """
    
    [n,m] = X.shape
    
    if V is None:
        V = np.arange(n)
    
    objs = np.empty(k+1)
    
    A = np.empty(0, int)
    modA = np.sum(X[A,:], axis=0)
    ff = 0 # assume normalized for now
    objs[0] = ff
    
    for ii in range(k):
        
        maxgain = -100
        greedyv = np.random.choice(V)
        
        for vidx in range(len(V)):
            gain = submodgains(X, modA, ff, V[vidx], mixw)
            if gain > maxgain:
                maxgain = gain
                greedyv = V[vidx]
        
        # add element to A, remove from V, update gains
        A = np.append(A, greedyv)
        modA += X[greedyv,:]
        V = V[V!=greedyv]
        ff += maxgain
        objs[ii+1] = ff
    
    return A, objs

#%% Greedy for DMQ

def greedyDMquota_submod(V, X, mixw, Memvec, quo, k, verbose=False):
    """ For the disjoint membership quota.
        Memvec is an n x p one-hot matrix (exactly one 1 per row).
        quo is a p x 1 vector.
        Output a subset that satisfies the quotas.
    """
    
    [n,m] = X.shape
    p = Memvec.shape[1]
    
    if V is None:
        V = np.arange(n)
    
    for grp in range(p):
        if np.sum(Memvec[:,grp]) < quo[grp]:
            print("Not enough members in group {}, infeasible problem.".format(grp))
            return None, None
        
    objs = np.empty(k+1)
    
    A = np.empty(0, int)
    modA = np.sum(X[A,:], axis=0)
    ff = 0 # assume normalized for now
    objs[0] = ff
    
    """ Quota-filling stage """
    
    ii = 0
    Vsat = np.copy(V) # only for use in the quota-filling stage
    
    # remove from Vsat groups with zero quota
    for grp in range(p):
        if quo[grp] < 1:
            Vsat = np.delete(Vsat, np.argwhere(Memvec[Vsat,grp].flatten()))
    
    while ii < np.sum(quo):
        
        maxgain = -100
        greedyv = np.random.choice(Vsat)
        
        for vidx in range(len(Vsat)):
            gain = submodgains(X, modA, ff, Vsat[vidx], mixw)
            if gain > maxgain:
                maxgain = gain
                greedyv = Vsat[vidx]
        
        # add element to A, update gains
        A = np.append(A, greedyv)
        modA += X[greedyv,:]
        ff += maxgain
        objs[ii+1] = ff
        
        # remove from V, remove all from Vsat if quota filled
        V = V[V!=greedyv]
        grp = np.argwhere(Memvec[greedyv])

        if verbose:
            print("selected element", greedyv)
            print("lies in group", grp)
            #print("new A", A)

        if np.sum(Memvec[A,grp]) >= quo[grp]:
            if verbose:
                print("\n Quota for group {} satisfied by set {} \n".format(
                        grp, A[Memvec[A,grp].flatten().astype(bool)])
                     )
                print("Deleting {}".format(Vsat[Memvec[Vsat,grp].flatten().astype(bool)]))
            Vsat = np.delete(Vsat, np.argwhere(Memvec[Vsat,grp].flatten()))
        else:
            if verbose:
                print("Only deleting", Vsat[Vsat==greedyv])
            Vsat = Vsat[Vsat!=greedyv]
        
        ii += 1
        
    if verbose:
        print("Quotas filled.")
        print("quotas : ", quo)
        print("Representatives : ", [np.sum(Memvec[A,jj]) for jj in range(p)])
    
    
    """ Regular greedy stage """
    
    while ii < k:
        
        maxgain = -100
        greedyv = np.random.choice(V)
        
        for vidx in range(len(V)):
            gain = submodgains(X, modA, ff, V[vidx], mixw)
            if gain > maxgain:
                maxgain = gain
                greedyv = V[vidx]
        
        # add element to A, remove from V, update gains
        A = np.append(A, greedyv)
        modA += X[greedyv,:]
        V = V[V!=greedyv]
        ff += maxgain
        objs[ii+1] = ff
        
        if verbose:
            print("selected element", greedyv)
            print("lies in group", np.argwhere(Memvec[greedyv]))
            #print("new A", A)
        
        ii += 1
    
    return A, objs

#%% Greedy for IMQ

def greedyIMquota_submod(V, X, mixw, Memvec, quo, k, verbose=False):
    """ For the intersecting membership quota.
        Memvec is an n x p membership matrix (any number of ones per row).
        quo is a p x 1 vector.
        Output a subset that satisfies the quotas.
    """
    
    [n,m] = X.shape
    p = Memvec.shape[1]
    
    if V is None:
        V = np.arange(n)
    
    for grp in range(p):
        if np.sum(Memvec[:,grp]) < quo[grp]:
            print("Not enough members in group {}, infeasible problem.".format(grp))
            return None, None
        
    objs = np.empty(k+1)
    
    A = np.empty(0, int)
    modA = np.sum(X[A,:], axis=0)
    ff = 0 # assume normalized for now
    objs[0] = ff
    
    """ Quota-filling stage """
    
    Memsat = np.copy(Memvec) # look at this like a "saturation potential"
    sat = np.ones_like(quo) # to keep track of what is satisfied
    
    # ignore groups with zero quota
    for grp in range(p):
        if quo[grp] < 1:
            Memsat[:,grp] = 0
            sat[grp] = 0
    
    Vsat = V[np.sum(Memsat[V,:], axis=1)>0] # only for use in the quota-filling stage
    ii = 0

    while (np.sum(sat) > 0) & (ii < k):
        
        maxgain = -100
        greedyv = np.random.choice(Vsat)
        
        for vidx in range(len(Vsat)):
            gain = submodgains(X, modA, ff, Vsat[vidx], mixw)
            if gain > maxgain:
                maxgain = gain
                greedyv = Vsat[vidx]
        
        # add element to A, update gains
        A = np.append(A, greedyv)
        modA += X[greedyv,:]
        ff += maxgain
        objs[ii+1] = ff
        
        grps = np.argwhere(Memsat[greedyv]).flatten() # note that grp can be a list
        
        if verbose:
            print("selected element", greedyv)
            print("lies in unsatisfied groups", grps)
            #print("new A", A)
        
        # remove from V and Vsat
        V = V[V!=greedyv]
        Memsat[greedyv,:] = 0
        Vsat = Vsat[Vsat!=greedyv]
        
        # also remove from Vsat those who have no uniqueness to offer
        for grp in grps:
            if np.sum(Memvec[A,grp]) >= quo[grp]:
                sat[grp] = 0
                Memsat[:,grp] = 0
                if verbose:
                    print("\n Quota for group {} satisfied by set {} \n".format(
                            grp, A[Memvec[A,grp].flatten().astype(bool)]
                        ))
                    print("Deleting {}".format(
                            Vsat[np.sum(Memsat[Vsat,:], axis=1)==0]
                        ))
                Vsat = Vsat[np.sum(Memsat[Vsat,:], axis=1)>0]
        
        ii += 1
        
    if verbose:
        print("Quotas filled.")
        print("quotas : ", quo)
        print("Representatives : ", [np.sum(Memvec[A,jj]) for jj in range(p)])
    
    """ Regular greedy stage """
    
    while ii < k:
        
        maxgain = -100
        greedyv = np.random.choice(V)
        
        for vidx in range(len(V)):
            gain = submodgains(X, modA, ff, V[vidx], mixw)
            if gain > maxgain:
                maxgain = gain
                greedyv = V[vidx]
        
        # add element to A, remove from V, update gains
        A = np.append(A, greedyv)
        modA += X[greedyv,:]
        V = V[V!=greedyv]
        ff += maxgain
        objs[ii+1] = ff
        
        if verbose:
            print("selected element", greedyv)
            print("lies in group", np.argwhere(Memvec[greedyv]))
            #print("new A", A)
        
        ii += 1
    
    return A, objs

#%%
