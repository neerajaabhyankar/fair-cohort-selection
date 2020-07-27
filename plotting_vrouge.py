# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:06:15 2017

@author: neeraja
"""

from six.moves import cPickle
from scipy import io
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import math
import funcs
#from plotting_vrouge_func import plotvr

##############################################

def measureDist(w_learnt, w_true):
    """
    A measure of the distance between two probability distributions
    Here, the coefficients of our submod mixture.
    
    Input w_learnt is a matrix with each ROW being a time instance
    (has 628 columns)
    (w_true is of size 628)
    """
    N_samples = len(w_learnt)
    
    w_learnt = np.array([w_learnt[t]/np.sum(w_learnt[t]) for t in range(N_samples)])
    
    #2norm
    #dist = np.array([np.linalg.norm(w_learnt[t] - w_true) for t in range(N_samples)])
    
    #bhattacharyya
    dist = np.array([0.0 for t in range(N_samples)])
    for t in range(N_samples):
        for i in range(len(w_true)):
            dist[t] += math.sqrt(w_learnt[t, i]*w_true[i])
        dist[t] = -math.log(dist[t])

    #hellinger
#    dist = np.array([0.0 for t in range(N_samples)])
#    for t in range(N_samples):
#        for i in range(len(w_true)):
#            dist[t] += math.sqrt(w_learnt[t, i]*w_true[i])
#        dist[t] = math.sqrt(1 - dist[t])
        
    return dist

################################################

def measureRandEvals(Data_Mat, w_learned_array, idx, ftrue, feature_based_func_type, gamma, beta_Mat):
    """
    A measure of a learnt function:
    The average normed difference between evaluations of "l" random subsets by f_true and f_learnt
    procdata contains all features & rouge info
    w_learned is the thing we're evaluating (at every iter) ## N_samples x N_dim
    ftrue is what we're evaluating it against
    """
    #N_dim = 60
    #Data_Mat = proc_data['all_Feature_Vec'][0][idx]
    #Data_Mat = np.transpose(Data_Mat) # the feature matrix ## 628 x 100
    
    V = np.arange(100) #ground set
    K = 10 #proc_data['K_budget'][0][0]
    N_samples = len(w_learned_array)
    lrange = 20
    
    dist = np.array([0.0 for t in range(N_samples)])
    
    #for each iteration n
    for n in range(N_samples): #N_samples = len(w_learned_array)
        print("jdx = ", n)
        
        for l in range(lrange):
            S = np.random.choice(V, K, replace=False)
            
            #if conc == 'log' or any of those \phi s
            learned_y = funcs.truescore_image_summarization(w_learned_array[n], Data_Mat, S, feature_based_func_type, gamma)
            
            #if conc == minfeatp
            #learned_y = funcs.minfeatscore_image_summarization(w_learned_array[n], beta_Mat, Data_Mat, S)
            
            if type(ftrue) == str:
                if ftrue == 'vrouge':
                    proc_data = {} #empty dictionary
                    io.loadmat('processed_data.mat', mdict=proc_data)
                    true_y = funcs.score_image_summarization(proc_data['all_subset'][0][idx], np.transpose(Data_Mat), S)[0]
            else:
                w_true = ftrue
                true_y = funcs.truescore_image_summarization(w_true, Data_Mat, S, feature_based_func_type, gamma)
                
            dist[n] += abs(true_y - learned_y)/float(lrange)
            
    return dist
        
################################################

N_samples = 1000
#alphachoice_str = ['inf', 'auto']
alphastr = 'auto'
alphastr = 'inf'
measure = 'Deviation from true weights'

#data_dir = 'synthetic_normed-y_poisson_w_folder/'
data_dir = 'vrouge_FINAL/'
p = 0

#featfile = open('feat'+str(N_dim)+'_maxvar' + '.save', 'rb')
#data = cPickle.load(featfile)
#featfile.close()
#Data_Mat = data[idx]
#Data_Mat = np.transpose(Data_Mat) # the feature matrix ## N_dim x 100



savedict = {}

# TAKE MEAN AND STD DEV OF RANDOM SEPARATELY

#algo = 'random'
#rand_runs = np.empty([0, N_samples], dtype=float)
#
#for idx in range(3):
#    for r in range(len(alphachoice_str)):
#        outfile = 'Algo_' + algo + '_Nsmps_' + str(N_samples) + '_L_' + str(N_bins) + '_alpha_' + alphachoice_str[r] + '_const-thresh' + '_collection_' + str(idx+1) + '_human_' + 'true'
#        opfile = open(data_dir + outfile + '.save', 'rb')
#        w_vec = cPickle.load(opfile)['Learned_Vec']
#        ## difference between w_true and each ROW of w_vec
#        ## yields a N_samples dimension vector
#        dist = measureDist(w_vec, w_true)
#        rand_runs = np.vstack([rand_runs, dist])
#        opfile.close()
#rand_mean = np.mean(rand_runs, axis=0)
#rand_err_mean =  np.std(rand_runs, axis=0)

rand_runs = 5
N_dim = 100
N_ground = 100
idxrange = 14
K = 10

L = 5

a_arr = np.array([[0 for i in range(N_samples)] for r in range(rand_runs)], dtype=float)  

for r in range(rand_runs):
    outfile = 'Algo_' + 'random' + str(r) + '_Nsmps_' + str(N_samples) + '_L_' + str(L) + '_alpha_' + alphastr + '_collections_14' + '_human_' + 'vrouge_small' + str(N_dim)
    opfile = open(data_dir + outfile + '.save', 'rb')     
    a = cPickle.load(opfile)
    a_arr[r] += a['Performance']
    opfile.close()
    
b_mean = np.array([0 for i in range(N_samples)], dtype=float)  

outfile = 'Algo_' + 'greedy0' + '_Nsmps_' + str(N_samples) + '_L_' + str(L) + '_alpha_' + alphastr + '_collections_14' + '_human_' + 'vrouge_small' + str(N_dim)
opfile = open(data_dir + outfile + '.save', 'rb')
b = cPickle.load(opfile)
b_mean += b['Performance']
opfile.close()

c_mean = np.array([0 for i in range(N_samples)], dtype=float)  

outfile = 'Algo_' + 'dsopt0' + '_Nsmps_' + str(N_samples) + '_L_' + str(L) + '_alpha_' + alphastr + '_collections_14' + '_human_' + 'vrouge_small' + str(N_dim)
opfile = open(data_dir + outfile + '.save', 'rb')
c = cPickle.load(opfile)
c_mean += c['Performance']
opfile.close()

### PLOT

#plt.figure()

s=range(N_samples)#; s=range()

#plt.errorbar(s, a_mean, yerr=a_std, color="lightpink", fmt="o", markersize = 1, markeredgecolor="lightpink", markeredgewidth=0.3, linewidth = 0.4, capsize = 1)

for r in range(rand_runs):
    A, = plt.plot(s, a_arr[r], color="mediumblue", linestyle="-", linewidth = 0.8, label='Random Querying', alpha=0.4)

B, = plt.plot(s, b_mean, color="orange", linestyle="-", linewidth = 1.2, label='Greedy optimized ACS')

C, = plt.plot(s, c_mean, color="crimson", linestyle="-", linewidth = 1.2, label='DS optimized ACS')

axes = plt.gca()
axes.set_xlim([0,200])
axes.set_ylim([0.75, 0.93])

fs = 12
#plt.locator_params(axis='y', nticks=4)
#plt.locator_params(axis='x', nticks=6)
plt.tick_params(axis='both', labelsize=fs)

plt.xlabel('Learning Rounds', fontsize=fs)
plt.ylabel('Performance with estimated weights', fontsize=fs)
plt.title('V-rouge Experiments : 5-discretized feedback', fontsize=fs)

for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(1)
#axes.legend().get_frame().set_linewidth(0.5)

leg = plt.legend(handles=[B,C,A], fontsize=fs, loc=4, bbox_to_anchor=(1.0,0.08))
leg.get_frame().set_edgecolor('white')

plt.tight_layout(pad=0.5)
plt.axes().set_aspect(750)
#plt.subplots_adjust(top=0.85)
#plt.suptitle('Online learning Realizable case : Fixed Threshold', fontsize=10)
plotname = 'perf-plot-alphainf-L'+str(L)+'-bigfont'
plt.savefig(data_dir + plotname + '.pdf', dpi=800)

#plotfile = open(data_dir + plotname + '.save', 'wb')
#cPickle.dump(savedict, plotfile)
#plotfile.close()

######################################

