#Author : Neeraja Abhyankar
#Date : May 2018

from scipy import io
import numpy as np
import sys, os
import math
from six.moves import cPickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticks

#sys.path.insert(0, os.getcwd()+'/static/pycode/')
import funcs
import GreedySubmodKnapsackWeighted
import DSoptCardinalityConstrained
import GreedyCardinalityConstrained
import GreedyOnVrouge

#import thresholdsL

def create_results(n_dim, algo, randrunnum, L):
    """
    This script runs the online learning scenario where
    the uncertainty sampling objective is optimized using various algorithms.
    
    Parameter descriptions:
        N_samples: number of rounds in the interactive experiment
        feature_based_func_type: the type of the concave function you want to use
            1: x.^gamma with 0 < gamma < 1;
            2: log (1+gamma*x) with gamma >0;
            3: 1 - gamma^(-x) with gamma > 1;
            4: 1/(1+exp^(-gamma * x)) - 0.5;
        gamma: the parameter defining the concave function
        N_dim: number of modular features used
        noise_level: the parameter associated with the linear regression problem
        N_bins: feedback quantization
        eta: additive noise to the feedback in SNR to simulate quantization
    """
    
    # SOME VARIABLES -- FIXED
    
    N_ground = 100 # this number should be fixed in this case
    V = np.arange(1, N_ground+1)
    N_dim = 628 # this number should also be fixed
    K_budget = 10 # this number should be fixed
    
    feature_based_func_type_feedback = 2 # this is the function giving us feedback (in place of v-rouge)
    gamma_feedback = 100
    feature_based_func_type = 2 # this is the function we're modeling it as
    gamma = 100
    noise_level = 0.1
    threshold_zero_flag = 1
    #lambda_regul = float(1)/pow(noise_level,2)
    lambda_regul = 1000
    #noise_offset = 0.0
    #less_items_flag = 1 #20choose5 instead of 100choose10
    #if less_items_flag == 1:
        #N_ground = 
        #K_budget = 
    
    # RANDOM VARIABLES
    #algo = 'random' ###############################
    alpha_ind = 0 ##################################
    alphachoice_str = ['0', '0_1', '1', '10', 'inf', 'lin', 'auto']
    #idx = 0 #######################################
    eta = 100 #SNR in dB/bit
    
    # THRESHOLD LEARNING PARAMETERS
    #L = 2 #number of BITS
    #thresh_learn_init = thresholdsL.thresh[L] ##2
    
    # FIXED FOR NOW, potentially changeable
    N_samples = 1000
    idxrange = 14
    
    #data_dir = ''
    data_dir = 'vrouge_FINAL/'
    
    #NUMBER OF FEATURES
    if n_dim != 0:
        N_dim = n_dim
    
    outfile = alphachoice_str[alpha_ind]
    outfile = 'Algo_' + algo + randrunnum + '_Nsmps_' + str(N_samples) + '_L_' + str(L) + '_alpha_' + alphachoice_str[alpha_ind] + '_collections_14' + '_human_' + 'vrouge_small' + str(N_dim)
    print('creating', outfile)
    
    while os.path.exists(data_dir + outfile + '.save') == True:
        outfile = outfile + '_repeated' #ensures unique names 
    
    savedict = {}
    savedict['Performance'] = np.array([0.0 for i in range(N_samples)])
    savedict['Learned_Vec'] = {}
    savedict['Thresh'] = {}
    savedict['Queried_Summaries'] = {}
    savedict['Queried_Scores'] = {}
    savedict['Unquant_Scores'] = {}
    
    for idx in range(idxrange):   
    
        # FEATURE MATRIX
        All_Data_Mat = {} #empty dictionary
        io.loadmat('data-reduceddim-facloc-full.mat', mdict=All_Data_Mat);
        Data_Mat_Complete = All_Data_Mat[str(idx)]
        Data_Mat = Data_Mat_Complete[:N_dim, :]
        
        alphachoice = [0, 0.1, 1, 10, None]
        if alpha_ind < 5:
            alpha = alphachoice[alpha_ind]
            
        if alpha_ind == 6:
            alpha = 0.001 ###############to start with
    
        # LOADING PROCESSED DATASET INFO
        
        proc_data = {} #empty dictionary
        io.loadmat('processed_data.mat', mdict=proc_data);
        
        subset = proc_data['all_subset'][0][idx]
        
        #preproc_rand_summ_sc = {} #empty dictionary
        #io.loadmat('preprocessed_random_summaries_and_scores.mat', mdict=preproc_rand_summ_sc);
        
        """
        A whole bunch of variables have been added from the preprocessed data
        
        proc_data.keys()
        ['K_budget', '__header__', 'idx', '__globals__', 'imagecollection', 'all_random_scores', 'all_human_score', 'V_rouge_optimized_summary', 'K', 'N_dim', 'all_Feature_Vec', 'all_subset', 'Best_score', 'V', 'Data_Mat', 'N_images', 'V_rouge_optimized_summary_score', '__version__', 'Best_SummarySet', 'test_image_collection']
        
        preproc_rand_summ_sc.keys()
        ['__header__', 'Collected_All_Random_Rouge_Scores', 'Collected_All_Random_Summaries', '__globals__', '__version__']
        
        """

        # PRIORS
        #thresh_learn = thresh_learn_init[idx]
        mu_0 = np.array([1.0 for i in range(N_dim)]) # initialize the mean
        mu_0 = mu_0/np.sum(mu_0)
        sigma_0 = np.identity(N_dim) # initialize the covariance matrix
        # ESTIMATES TO UPDATE
        Cinv = lambda_regul * np.linalg.inv(sigma_0)
        Yvec = np.transpose(np.atleast_2d(np.dot(Cinv, mu_0))) ## N_dim x 1
        # LEARNT
        w_vec = np.squeeze(np.dot(np.linalg.inv(Cinv), Yvec)) ## N_dim
        mu_0 = None
        sigma_0 = None
            
        # STUFF TO STORE
        perf_vec = np.array([0.0 for i in range(N_samples)]) ## 100
        thresh_learn_vec = np.array([0.0 for i in range(N_samples)]) ## 100
        coeff_vec = np.array([[0.0,]*(np.size(Data_Mat, axis = 0)) for i in range(N_samples)]) ## 100 x 628
        unquantized_score_vec = np.array([0.0 for i in range(N_samples)]) ## 100
        Uncertainty_Queried_Summaries = np.array([[0.0,]*K_budget for i in range(N_samples)], dtype=int) ## 100 x 10
        UQSummaries_dict = {} #for searching purposes : keys=sets, values=indexes
        Uncertainty_Queried_Scores = np.array([0.0 for i in range(N_samples)]) ## 100
        
        # USED FOR NORMALIZING
        # run greedy on Vrouge itself -- use access to the chosen summaries "subset"
        max_opt_summary = GreedyOnVrouge.GreedyOnVrouge(subset, np.transpose(Data_Mat_Complete), V, K_budget, [1 for i in range(N_ground)], 1)
        # best possible y
        ymax = funcs.score_image_summarization(subset, np.transpose(Data_Mat_Complete), max_opt_summary)[0]
        # worst possible y
        ymin = 1000
        for randmin in range(200):
            SummarySet = np.random.choice(range(N_ground), K_budget, replace=False)
            ymintemp = funcs.score_image_summarization(subset, np.transpose(Data_Mat_Complete), SummarySet)[0]
            ymin = min(ymin, ymintemp)


        for jdx in range(N_samples):
            print('jdx = ', jdx)
            
            if alpha_ind == 5:
                # VARY LINEARLY
                if jdx < N_samples/3:
                    alpha = 0.1
                elif jdx < 2*N_samples/3:
                    alpha = 1
                else:
                    alpha = 10
                    
            # FIND QUERY
            #IF REPEATED, RECORD OLD SCORE AND PROCEED
            #IF TOO HARD, JUST SELECT A RANDOM SUMMARY
            
            uniquesummflag = 0 #haven't found a unique summary yet
            uniqueiters = 0 #tried this many times
            while uniquesummflag == 0:
                
                #find a summary
                #w_mediate = np.array([0.0 for i in range(N_dim)])
                w_mediate = w_vec
                 
                if algo == 'random':
                    SummarySet = np.random.choice(range(N_ground), K_budget, replace=False)
                    
                if algo == 'dsopt':
                    SummarySet = DSoptCardinalityConstrained.DSoptCardinalityConstrained(np.transpose(Data_Mat), feature_based_func_type, w_mediate, np.linalg.inv(Cinv), alpha, V, K_budget, gamma)
            
                if algo == 'greedy':
                    SummarySet = GreedyCardinalityConstrained.GreedyCardinalityConstrained(np.transpose(Data_Mat), feature_based_func_type, w_mediate, np.linalg.inv(Cinv), alpha, V, K_budget, gamma)
                    
                #see if it's unique
                #dict has been used to compare
                uniquesummflag = 1 #temporarily we think this is unique
                if frozenset(SummarySet) in UQSummaries_dict.keys():
                    print("repeated")
                    uniqueiters += 1
                    jjdx = UQSummaries_dict[frozenset(SummarySet)]
                    uniquesummflag = 0

                    #copy score from jjdx and update Cinv, Yvec, w_vec
                    #do not store summary in log
                
                    #training data
                    x = funcs.featurize_data(Data_Mat, feature_based_func_type, gamma, SummarySet)
                    x = np.atleast_2d(x) ## 1 x N_dim
                    y = Uncertainty_Queried_Scores[jjdx]
                    #learn
                    Cinv += float(1)/pow(noise_level,2) * np.dot(np.transpose(x), x)
                    Yvec += np.transpose(float(1)/pow(noise_level,2) * np.dot(y, x))
                    w_vec = np.squeeze(np.dot(np.linalg.inv(Cinv), Yvec))
                    if threshold_zero_flag == 1:
                        w_vec = np.array([max(i, 0) for i in w_vec])
                    w_vec = w_vec/np.sum(w_vec)
                    
                    if uniqueiters > 20:
                        #quit trying -- speed is important
                        SummarySet = np.random.choice(range(N_ground), K_budget, replace=False)
                        uniquesummflag = 1
                        print("switched to random selection for this round")
                
                #if it is indeed unique
                #the loop will automatically be exited
                        
            #here, we now have a unique SummarySet
            #proceed as before
            
            Uncertainty_Queried_Summaries[jdx, :] = SummarySet
            UQSummaries_dict[frozenset(SummarySet)] = jdx

            # GET THE SCORE (for this jdx)
            y = funcs.score_image_summarization(subset, np.transpose(Data_Mat_Complete), SummarySet)[0]
            y = (y-ymin)/(ymax-ymin) #this CAN be negative
            unquantized_score_vec[jdx] = y
            
#            # ADD NOISE TO FEEDBACK
#            #amp_signal ~ 0.05
#            amp_signal = 0.03
#            amp_noise = amp_signal/math.pow(10, eta/20)
#            y += np.random.normal(noise_offset, amp_noise)
                        
#            if N_bins != 0: #discretized feedback
#                ydisc = 0
#                for t in range(L-1):
#                    ydisc += float(y >= thresh_learn[t])
#                y = ydisc
            
            if L != 0: #discretized feedback
                y = np.floor(L*y)
                
            Uncertainty_Queried_Scores[jdx] = y
            
            if alpha_ind == 6:
                if alpha != None: #still increasing
                    if alpha > 100000: #if large
                        alpha = None #set to None to avoid reaching inf
                    elif y > 0.5: #not yet large
                        alpha = 10*alpha #make larger
            
            # UPDATE AND LEARN
            #training matrix
            x = funcs.featurize_data(Data_Mat, feature_based_func_type, gamma, SummarySet)
            x = np.atleast_2d(x) ## 1 x N_dim

            #learn
            Cinv += float(1)/pow(noise_level,2) * np.dot(np.transpose(x), x) ## N_dim x N_dim
            Yvec += np.transpose(float(1)/pow(noise_level,2) * np.dot(y, x)) ## N_dim x 1
            w_vec = np.squeeze(np.dot(np.linalg.inv(Cinv), Yvec)) ## N_dim
            
            if threshold_zero_flag == 1:
                w_vec = np.array([max(i, 0) for i in w_vec])
                
            #print('Sum of w_vec : ', np.sum(w_vec))
            w_vec = w_vec/np.sum(w_vec)
    
            
            # EVALUATE
            #run greedy
            optimized_summary = GreedySubmodKnapsackWeighted.GreedySubmodKnapsackWeighted(np.transpose(Data_Mat), feature_based_func_type_feedback, w_vec, V, K_budget, [1 for i in range(N_ground)], 1, gamma_feedback)
            optimized_y = funcs.score_image_summarization(subset, np.transpose(Data_Mat_Complete), optimized_summary)[0]
            
            optimized_y = optimized_y/ymax #prevented from being negative

            perf_vec[jdx] = optimized_y
            coeff_vec[jdx] = w_vec
            
        # SAVING

        # all are of length N_samples
        savedict['Performance'] += perf_vec/idxrange
        savedict['Learned_Vec'][idx] = coeff_vec
        savedict['Thresh'][idx] = thresh_learn_vec
        savedict['Queried_Summaries'][idx] = Uncertainty_Queried_Summaries
        savedict['Queried_Scores'][idx] = Uncertainty_Queried_Scores
        savedict['Unquant_Scores'][idx] = unquantized_score_vec

    opfile = open(data_dir + outfile + '.save', 'wb')
    cPickle.dump(savedict, opfile)
    opfile.close()

    return None


################################################

# RUN!!!!!!!!!!!!!!!!!!!!!!!!!!!!

ndimlist = np.array([5,10,20,50,100,150,200,300,500,0])
ndimlist = np.array([100])
L = 5

for n_dim in ndimlist:
    create_results(n_dim, 'dsopt', '0', L)
    create_results(n_dim, 'greedy', '0', L)
    create_results(n_dim, 'random', '0', L)
    create_results(n_dim, 'random', '1', L)
    create_results(n_dim, 'random', '2', L)
    create_results(n_dim, 'random', '3', L)
    create_results(n_dim, 'random', '4', L)