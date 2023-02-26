#%% Imports

import os
import math

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#%% Plot gain curves

def objplot(objs, ground, setcolor, setcolorground=None):
    krange = np.arange(len(objs))
    
    plt.plot(krange, objs, "o", c=setcolor, label="f(S)")
    if setcolorground is not None:
        plt.plot(krange, [ground,]*len(objs), "--", c=setcolorground, label="ground set eval")
    plt.xlabel("set size")
    plt.legend(loc=2)

def logobjplot(objs, ground, setcolor, setcolorground=None):
    krange = np.arange(len(objs))
    
    plt.plot(krange, np.log(objs), "o", c=setcolor, label="log(f(S))")
    if setcolorground is not None:
        plt.plot(krange, [np.log(ground),]*len(objs), "--", c=setcolorground, label="ground set eval")
    plt.xlabel("set size")
    plt.legend(loc=2)

#%% Plot TSNE of subset selection

def viztsne(X, S, setcolor, setlabel, perp):
    """ Given an nxm feature matrix X
        A selection of indices S of size k < n
        Plot the selection's 2d TSNE
    """
    
    [n,m] = X.shape
    
    Xemb = TSNE(n_components=2, random_state=256).fit_transform(X)
    
    plt.scatter(Xemb[:,0], Xemb[:,1], c="lightskyblue")
    plt.scatter(Xemb[S,0], Xemb[S,1], facecolors="none", edgecolors=setcolor, linewidth=2, label=setlabel)
    plt.legend()
    
#%% Plot PCA reduction of subset selection -- with Discrete Memberships

def vizpca(X, S, setcolor, setlabel):
    """ Given an nxm feature matrix X
        A selection of indices S of size k < n
        An n x p Membership matrix for p groups
        Plot the selection's 2d PCA
    """
    
    [n,m] = X.shape
    
    Xemb = PCA(n_components=m).fit(X).transform(X)[:,:2]
    
    plt.scatter(Xemb[:,0], Xemb[:,1], c="lightskyblue")
    plt.scatter(Xemb[S,0], Xemb[S,1], facecolors="none", edgecolors=setcolor, linewidth=2, label=setlabel)
    plt.legend()
    
#%% Plot TSNE of subset selection -- with Discrete Memberships

def viztsne_DMQ(X, Memvec, memcolors, S, setcolor, setlabel, perp):
    """ Given an nxm feature matrix X
        A selection of indices S of size k < n
        An n x p Membership matrix for p groups
        Plot the selection's 2d TSNE
    """
    
    [n,m] = X.shape
    p = Memvec.shape[1]
    
    Xemb = TSNE(n_components=2, random_state=256).fit_transform(X)
    
    for jj in range(p):
        memgrp = np.argwhere(Memvec[:,jj])
        plt.scatter(Xemb[memgrp,0], Xemb[memgrp,1], c=memcolors[jj])
    plt.scatter(Xemb[S,0], Xemb[S,1], facecolors="none", edgecolors=setcolor, linewidth=2, label=setlabel)
    plt.legend()

#%% Plot PCA reduction of subset selection -- with Discrete Memberships

def vizpca_DMQ(X, Memvec, memcolors, S, setcolor, setlabel):
    """ Given an nxm feature matrix X
        A selection of indices S of size k < n
        An n x p Membership matrix for p groups
        Plot the selection's 2d PCA
    """
    
    [n,m] = X.shape
    p = Memvec.shape[1]
    
    Xemb = PCA(n_components=m).fit(X).transform(X)[:,:2]
    
    for jj in range(p):
        memgrp = np.argwhere(Memvec[:,jj])
        plt.scatter(Xemb[memgrp,0], Xemb[memgrp,1], c=memcolors[jj])
    plt.scatter(Xemb[S,0], Xemb[S,1], facecolors="none", edgecolors=setcolor, linewidth=2, label=setlabel)
    plt.legend()

#%% Plot TSNE of subset selection -- with Intersecting Memberships

def viztsne_IMQ(X, Memvec, memcolors, S, setcolor, setlabel, perp):
    """ Given an nxm feature matrix X
        A selection of indices S of size k < n
        An n x p Membership matrix for p groups
        Plot the selection's 2d TSNE
    """
    
    [n,m] = X.shape
    p = Memvec.shape[1]
    
    Xemb = TSNE(n_components=2, random_state=256).fit_transform(X)
    
    plt.figure(figsize=(p*5, 5))
    for jj in range(p):
        plt.subplot(1,p,jj+1)
        plt.scatter(Xemb[:,0], Xemb[:,1], c=memcolors[0])
        
        memgrp = np.argwhere(Memvec[:,jj])
        plt.scatter(Xemb[memgrp,0], Xemb[memgrp,1], c=memcolors[1], label="group{}".format(jj+1))
        
        plt.scatter(Xemb[S,0], Xemb[S,1], facecolors="none", edgecolors=setcolor, linewidth=2, label=setlabel)
        plt.legend()
    

#%% Plot TSNE of subset selection -- with Feature Quotas

#


#%% Plot Group Membership histograms of selections -- Membership Quota

def vizbalance_MQ(V, Memvec, vcolor, quo, S, setcolor, setlabel="selection"):
    """ Given a ground set (n)
        With Membership Assignments (n x p)
        And optionally a quota
        A selection of indices S of size k < n
        Plot the selection's 2d PCA
    """
    
    [n,p] = Memvec.shape
    if V is None:
        V = np.arange(n)
    
    Vgroups = np.zeros(p)
    Sgroups = np.zeros(p)
    unsat = np.empty((0)).astype(int)
    
    for jj in range(p):
        Vgroups[jj] = np.sum(Memvec[V,jj])
        Sgroups[jj] = np.sum(Memvec[S,jj])
        if Sgroups[jj] < quo[jj]:
            unsat = np.append(unsat, jj)
    
    plt.bar(np.arange(p), Vgroups, color=vcolor)
    plt.bar(np.arange(p), Sgroups, color=setcolor, label=setlabel)
    plt.plot(np.arange(p), quo, marker="o", linestyle="--", c="white", markersize=7)
    plt.scatter(unsat, quo[unsat], c="red", s=7, zorder=10)
    plt.legend()

#%%

#

