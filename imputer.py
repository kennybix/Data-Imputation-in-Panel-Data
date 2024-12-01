# import important libraries

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt 
import seaborn as sns 
import pickle
from joblib import Parallel, delayed

from numpy.linalg import LinAlgError 
from functools import reduce 

from sklearn.neighbors import KernelDensity, KNeighborsRegressor
from utils import *

class Imputer:
    def __init__(self, masked_rank_chars):
        """ 
        masked_rank_chars : panel data with missing points
        """
        self.masked_rank_chars = masked_rank_chars

        T, N, L = masked_rank_chars.shape 

        self.T, self. N, self.L = T, N, L 

        # set defaults for the different methods

        # em
        self.em_params = {
            'max_iter': 20,
            'eps': 1e-03,
            'min_xs_obs': 1
        }

        # xs-median
        self.xs_median_params = {
            'min_xs_obs': 1
        }

        # xs
        self.xs_params = {
            'K': L,
            'time_varying_loadings': True,
            'reg_param': 0.01/L,
            'eval_weight_lmbda': True,
            'shrink_lmbda': False,
            'min_xs_obs': 1
        }

        # b-xs
        self.b_xs_params = {
            'K': L,
            'time_varying_loadings': True,
            'reg_param': 0.01/L,
            'eval_weight_lmbda': True,
            'shrink_lmbda': False,
            'min_xs_obs': 1
        }

    def impute_with_em(self, params=""):

        # check if params is empty

        if params == "":
            params = self.em_params 
        
        else:
            params = params 

        max_iter = params['max_iter']
        eps = params['eps']
        min_xs_obs = params['min_xs_obs']

        T,N,L = self.masked_rank_chars.shape 
        min_chars = min_xs_obs # minimum number of characteristics that must be observed
        char_panel = np.copy(self.masked_rank_chars)
        missing_mask_overall = np.isnan(char_panel)
        self.missing_mask_overall = missing_mask_overall 
        return_mask = np.sum(~missing_mask_overall, axis=2) >= min_chars 

        char_panel[np.sum(~np.isnan(missing_mask_overall), axis=2) < min_chars] = np.nan 

        em_imputations = np.zeros((T,N,L)) # T X N X L 
        em_imputations[:,:] = np.nan 

        imputations = [impute_em(char_panel[t]) for t in range(T)]

        for t in range(T):
            em_imputations[t, return_mask[t]] = imputations[t]['X_imputed'][return_mask[t]] # copying imputations into em_imputations

        self.rank_imputed_chars = em_imputations 

        return self.rank_imputed_chars 
    
    def impute_with_xs_median(self, params=""):
        # check if params is empty

        if params == "":
            params = self.xs_median_params 

        else:
            params = params 

        min_chars = params['min_xs_obs']

        # run the baseline code
        char_panel = np.copy(self.masked_rank_chars)
        missing_mask_overall = np.isnan(char_panel)

        char_panel[np.sum(~np.isnan(missing_mask_overall), axis=2) < min_chars] = np.nan 

        # get the missing mask after min_chars is enforced 
        missing_mask_overall = np.isnan(char_panel)
        self.missing_mask_overall = missing_mask_overall 

        return_mask = np.sum(~missing_mask_overall, axis=2) >= min_chars 

        imputed_chars = np.copy(char_panel) # this is the rank data 

        new_imputation = xs_median_impute(imputed_chars)

        # revisit this --- perhaps using the return_mask is best
        imputed_chars[missing_mask_overall] = new_imputation[missing_mask_overall]

        self.rank_imputed_chars = imputed_chars

        return self.rank_imputed_chars 
    
    def impute_with_forward_filling(self, params=""):
        """ 
        Make Imputations using forward filling
        """

        # run the baseline code
        char_panel = np.copy(self.masked_rank_chars)
        missing_mask_overall = np.isnan(char_panel)

        self.missing_mask_overall = missing_mask_overall

        imputed_chars = np.copy(char_panel) # rank data
        
        new_imputation = simple_impute(imputed_chars)

        # revisit this --- perhaps using the return_mask is best
        imputed_chars[missing_mask_overall] = new_imputation[missing_mask_overall]

        self.rank_imputed_chars = imputed_chars

        return self.rank_imputed_chars 


    def impute_with_xs(self, params=""):
        """ 
        Make Imputations with cross-sectional approach
        """

        if params == "": # check if params is empty
            params = self.xs_params
        else:
            params = params 

        self.min_xs_obs = params['min_xs_obs']
        self.K = params['K']
        self.time_varying_loadings = params['time_varying_loadings']
        self.reg_param = params['reg_param']
        self.eval_weight_lmbda = params['eval_weight_lmbda']
        self.shrink_lmbda = params['shrink_lmbda']


        # run the baseline code
        char_panel = np.copy(self.masked_rank_chars)
        min_chars = self.min_xs_obs
        missing_mask_overall = np.isnan(char_panel)

        char_panel[np.sum(~np.isnan(missing_mask_overall), axis=2) < min_chars] = np.nan 

        # obtain modified missing mask after enforcing min_chars
        
        missing_mask_overall = np.isnan(char_panel)
        self.missing_mask_overall = missing_mask_overall 
        return_mask = np.sum(~missing_mask_overall, axis=2) >= min_chars

        imputed_chars = np.copy(char_panel) # rank data
        
        mu = np.nanmean(imputed_chars, axis=1) # important for the algorithm stability

        lmbda, cov_mat = estimate_lambda(imputed_chars, self.T, self.K,self.min_xs_obs, 
                                        self.time_varying_loadings,
                                        reg=self.reg_param, eval_weight_lmbda = self.eval_weight_lmbda,
                                        shrink_lmbda=self.shrink_lmbda)

        assert np.sum(np.isnan(lmbda)) == 0, f"lambda should contain no nans, {np.argwhere(np.isnan(lmbda))}"

        gamma_ts = np.zeros((char_panel.shape[0], char_panel.shape[1], self.K))  # T X N X K 
        gamma_ts[:,:] = np.nan 

        def get_gamma_t(ct, present, to_impute, lmbda, time_varying_lambdas, t):

            if time_varying_lambdas:
                gamma_t = lmbda[t].T.dot(ct.T).T # gamma_t = ct @ lmbda[t]
                gamma_t = get_optimal_A(lmbda[t].T, gamma_t, present, ct, L=self.L, 
                                        idxs=to_impute, reg=self.reg_param, mu=mu[t])
            else:
                gamma_t = lmbda[t].T.dot(ct.T).T # gamma_t = ct @ lmbda[t]
                gamma_t = get_optimal_A(lmbda.T, gamma_t, present, ct, L=self.L, 
                                        idxs=to_impute, reg=self.reg_param, mu=mu[t])
            
            return gamma_t 
        
        gammas = [get_gamma_t(
            ct=char_panel[t],
            present= ~np.isnan(char_panel[t]),
            to_impute= np.argwhere(return_mask[t]).squeeze(),
            lmbda=lmbda,
            time_varying_lambdas=self.time_varying_loadings, t=t,
        ) for t in range(self.T)]

        for t in range(self.T):
            gamma_ts[t, return_mask[t]] = gammas[t][return_mask[t]] # copying gamma into gamma_ts 

        self.rank_imputed_chars = imputed_chars

        return self.rank_imputed_chars 
    

    def impute_with_b_xs(self, params=""):
        """ 
        Cross-sectional imputation + backward time series information
        """

        if params == "": # check if params is empty
            params = self.b_xs_params
        else:
            params = params 

        self.min_xs_obs = params['min_xs_obs']
        self.K = params['K']
        self.time_varying_loadings = params['time_varying_loadings']
        self.reg_param = params['reg_param']
        self.eval_weight_lmbda = params['eval_weight_lmbda']
        self.shrink_lmbda = params['shrink_lmbda']

        # run the xs function
        xs_imputation = self.impute_with_xs(params)

        # run the baseline code
        char_panel = np.copy(self.masked_rank_chars)
        min_chars = self.min_xs_obs
        missing_mask_overall = np.isnan(char_panel)

        char_panel[np.sum(~np.isnan(missing_mask_overall), axis=2) < min_chars] = np.nan 

        # obtain modified missing mask after enforcing min_chars
        
        missing_mask_overall = np.isnan(char_panel)
        self.missing_mask_overall = missing_mask_overall 
        return_mask = np.sum(~missing_mask_overall, axis=2) >= min_chars

        imputed_chars = np.copy(char_panel) # rank data
        
        mu = np.nanmean(imputed_chars, axis=1) # important for the algorithm stability

        # compute the residuals
        residuals = simple_impute(imputed_chars) - xs_imputation

        local_bw = impute_chars(char_panel, xs_imputation, residuals, suff_stat_method='last_val', constant_beta=False)

        imputed_chars[missing_mask_overall] = local_bw[missing_mask_overall]

        local_bw_missing_mask = np.isnan(imputed_chars) # obtain the missing mask in the already filled data

        # improving data quality by imputing from the xs_imputation
        imputed_chars[local_bw_missing_mask] = xs_imputation[local_bw_missing_mask]

        self.rank_imputed_chars = imputed_chars

        return self.rank_imputed_chars 
    
    def evaluate_imputations(self, rank_chars):
        """ 
        Adapted function to measure the quality of the imputation
        rank_chars : the complete ranked characteristics
        """
        truth_panel = rank_chars

        tgt = np.copy(truth_panel)
        T, N, L = tgt.shape
        imputed = np.zeros((T,N,L)) # initialized the panel
        imputed[~self.missing_mask_overall] = np.nan # everywhere should be nan
        imputed[self.missing_mask_overall] = self.rank_imputed_chars[self.missing_mask_overall]

        tgt[np.isnan(imputed)] = np.nan # anyone that is still empty in the imputed chars, make them empty in the target

        # compute rmse and r2
        rmse = compute_rmse(tgt, imputed)
        r2 = compute_r2(tgt, imputed)

        metrics = {'rmse': rmse,
                   'r2': r2}
        
        print(metrics)

        return metrics
    