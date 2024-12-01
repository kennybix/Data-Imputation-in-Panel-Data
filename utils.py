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


# helper functions
def percentile_rank(x, UNK=np.nan):
    """ 
    utility method to quantile rank a vector
    """

    mask = np.logical_not(np.isnan(x))
    x_copy = np.copy(x)
    x_mask = x_copy[mask]
    n = len(x_mask)

    if n > 1:
        temp = [(i, x_mask[i]) for i in range(n)]
        temp_sorted = sorted(temp, key=lambda t: t[1])
        idx = sorted([(temp_sorted[i][0], i) for i in range(n)], key=lambda t: t[0])
        x_copy[mask] = np.array([idx[i][1] for i in range(n)]) / (n-1)

    elif n == 1:
        x_copy[mask] = 0.5
    return x_copy 

def percentile_rank_panel(char_panel): # very useful function
    """ 
    utility method to quantile rank the characteristics or features
    """
    ret_panel = np.zeros(char_panel.shape)
    ret_panel[:,:,:] = np.nan
    for t in range(char_panel.shape[0]):
        for i in range(char_panel.shape[2]):
            ret_panel[t,:,i] = percentile_rank(char_panel[t,:,i])


    return ret_panel


def get_cov_mat(char_matrix):
    """
    Calculate the covariance matrix of a partially observed panel using the method from Xiong & Pelger
    Parameters
    ----------
        char_matrix : the panel over which to calculate the covariance N x L
    """
    ct_int = (~np.isnan(char_matrix)).astype(float)
    ct = np.nan_to_num(char_matrix)
    mu = np.nanmean(char_matrix, axis=0).reshape(-1, 1)
    temp = ct.T.dot(ct) 
    temp_counts = ct_int.T.dot(ct_int)
    sigma_t = temp / temp_counts - mu @ mu.T
    return mu,sigma_t


def get_data_panel(data, N_column_name, start_date=None):
    """ 
    N_column_name could be the company column
    fetch data from user input
    """

    if start_date is not None:
        data = data.loc[data.date >= start_date]
    else:
        pass 

    dates = data['date'].unique() 
    dates.sort() # sort the dates
    Ns = data[N_column_name].unique() # obtain the unique Ns

    date_vals = [date for date in dates]
    chars = np.array([char for char in data.columns.tolist() if char not in ['date', N_column_name]])
    chars.sort()

    rank_chars = np.zeros((len(date_vals), Ns.shape[0], len(chars))) # create the panel
    rank_chars[:,:,:] = np.nan 


    N_map = {}
    for i, N_ in enumerate(Ns):
        N_map[N_] = i 

    for i, date in enumerate(dates):
        date_data = data.loc[data['date'] == date].sort_values(by=N_column_name)
        date_Ns = date_data[N_column_name].tolist()
        N_inds_for_date = [N_map[N_] for N_ in date_Ns]
        rank_chars[i, N_inds_for_date,:] = date_data[chars].to_numpy()

    return rank_chars, chars, date_vals, Ns



def get_dataframe_from_panel(rank_chars, chars, date_vals, Ns, N_column_name):
    """
    Converts the panel data back into the original dataframe format.
    
    Parameters:
    rank_chars : numpy.ndarray
        The panel data of shape (len(date_vals), len(Ns), len(chars)).
    chars : list
        List of column names for the data.
    date_vals : list
        List of unique dates.
    Ns : list
        List of unique values for the N_column.
    N_column_name : str
        The name of the N_column in the original dataframe.
        
    Returns:
    pandas.DataFrame
        The reconstructed dataframe.
    """
    # Create empty list to hold data for reconstruction
    rows = []
    
    # Iterate through the panel data and reconstruct each row
    for i, date in enumerate(date_vals):
        for j, N in enumerate(Ns):
            # Get the data values for the given date and N
            values = rank_chars[i, j, :]
            # If all values are NaN, skip this entry
            if np.all(np.isnan(values)):
                continue
            # Append a row with the date, N, and values
            row = [date, N] + values.tolist()
            rows.append(row)
    
    # Create the dataframe from the rows
    columns = ['date', N_column_name] + chars.tolist()
    reconstructed_df = pd.DataFrame(rows, columns=columns)
    
    return reconstructed_df



def estimate_lambda(char_panel, num_days_train, K, min_chars,
                    time_varying_loadings=False, eval_weight_lmbda=True,
                    shrink_lmbda=False, reg=0, window_size=1):
    
    """ 
    Fit the cross-sectional Loadings using XP method
    Parameters
    ----------
        char_panel: the panel over which to fit the model T X N X L
        num_days_train: if fitting a global model, the number of days over which to fit the loadings
        K: the number of cross-sectional factors
        min_chars: the minimum number of observations required for an entity to be included in the sample
        time_varying_loadings = False: whether or not to allow the loadings to vary over time

    Formula
    --------
    \hat{\Lambda^t} = \hat{V^t} (\hat{D^t}^{1/2})

    """
    # create a mask to show when the minimum number of characteristics is observed
    min_char_mask = np.expand_dims(np.sum(~np.isnan(char_panel), axis=2) >= min_chars, axis=2)

    cov_mats = []

    for t in range(num_days_train):
        # cov_mats.append(get_cov_mat(char_panel[t][1])) # Send the N X L to get the characteristic covariance matrix of dim L X L
        cov_mats.append(get_cov_mat(char_panel[t])[1]) # Send the N X L to get the characteristic covariance matrix of dim L X L

    # print(sum(cov_mats)) # first debug
    # print(cov_mats)
    # print(cov_mats.shape)
    # print(1/len(cov_mats))
    # cov_mats_sum = np.mean(cov_mats, axis=0)
    cov_mats_sum = sum(cov_mats) * (1/len(cov_mats)) # Takes the average of the covariance matrix

    if time_varying_loadings: # local-based models have time-varying lambdas
        lmbda = []
        cov_mat = []
        printed = False 

        for t in range(len(cov_mats)): # in this case, we want to avoid look-ahead bias, so we keep cov_mat as it was at day t
            cov_mats_sum = sum(cov_mats[max(0, t-window_size+1): t+1]) * (1/window_size)

            eig_vals, eig_vects = np.linalg.eigh(cov_mats_sum) # obtain the eigenvalues and eigenvectors, from the covariance matrix
            idx = np.abs(eig_vals).argsort()[::-1]
            if eval_weight_lmbda:
                if shrink_lmbda:
                    lmbda.append(eig_vects[:, idx[:K]] *
                                np.maximum(np.sqrt(np.sqrt(np.maximum(eig_vals[idx[:K]].reshape(1,-1),0))) - reg, 0))
                else:
                    lmbda.append(eig_vects[:, idx[:K]] * np.sqrt(np.maximum(eig_vals[idx[:K]].reshape(1,-1), 0)))
            else:
                lmbda.append(eig_vects[:, idx[:K]])
            assert np.all(~np.isnan(lmbda[-1])), lmbda
            cov_mat.append(cov_mats_sum)


    else:
        tgt_mat = cov_mats_sum
        eigh_vals, eig_vects = np.linalg.eigh(tgt_mat)

        idx = np.abs(eig_vals).argsort()[::-1]
        if eval_weight_lmbda:
            if shrink_lmbda:
                lmbda = eig_vects[:, idx[:K]] * np.maximum(np.sqrt(np.sqrt(eig_vals[idx[:K]].reshape(1,-1))) - reg, 0)
            else:
                lmbda = eig_vects[:, idx[:K]] * np.sqrt(np.maximum(0, eig_vals[idx[:K]].reshape(1, -1)))
        else:
            lmbda = eig_vects[:, idx[:K]]
        cov_mat = tgt_mat
    return lmbda, cov_mat 




def get_optimal_A(B, A, present, cl, L, idxs=[], reg=0, mu=None):
    """
    Get optimal A for cl = AB given that X is (potentially) missing data
    Parameters
    ----------
        B : matrix B
        A : matrix A, will be overwritten
        present: boolean mask of present data
        cl: matrix cl
        idxs: indexes which to impute
        reg: optinal regularization penalty
        mu: mean of the partially-observed characteristics
    """
    A[:,:] = np.nan
    for i in idxs:
        present_i = present[i,:]
        Xi = cl[i,:]
        Xi = Xi[present_i]
        Bi = B[:,present_i]
        assert np.all(~np.isnan(Bi)) and np.all(~np.isinf(Bi))
        effective_reg = reg 
        lmbda = effective_reg * np.eye(Bi.shape[1])

        if mu is not None:
            Xi = Xi - mu.T[present_i]
        try:
            A[i,:] = Bi @ np.linalg.lstsq(Bi.T @ Bi / L + lmbda, Xi / L, rcond=0)[0]
        except LinAlgError as e:
            lmbda = np.eye(Bi.shape[1])
            A[i,:] = Bi @ np.linalg.lstsq(Bi.T @ Bi / L + lmbda, Xi / L, rcond=0)[0]
    return A



def get_sufficient_statistics_last_val(characteristics_panel, max_delta=None,
                                      residuals=None):
    """
    Get the last observed value for a panel time series 
    Parameters
    ----------
        characteristics_panel : the time series panel, T x N x L
        max_delta=None : Optional, the maximum lag which is allowed for a previously observed value
        residuals=None : Optional, residuals T x N x L, the residuals the factor model applied to the time
                        series panel
    """
    T, N, L = characteristics_panel.shape
    last_val = np.copy(characteristics_panel[0])
    if residuals is not None:
        last_resid = np.copy(residuals[0])

    lag_amount = np.zeros_like(last_val)
    lag_amount[:] = np.nan
    if residuals is None:
        sufficient_statistics = np.zeros((T,N,L, 1), dtype=float)
    else:
        sufficient_statistics = np.zeros((T,N,L, 2), dtype=float)
    sufficient_statistics[:,:,:,:] = np.nan
    deltas = np.copy(sufficient_statistics[:,:,:,0])
    for t in range(1, T):
        lag_amount += 1
        sufficient_statistics[t, :, :, 0] = np.copy(last_val)
        deltas[t] = np.copy(lag_amount)
        present_t = ~np.isnan(characteristics_panel[t])
        last_val[present_t] = np.copy(characteristics_panel[t, present_t])
        if residuals is not None:
            sufficient_statistics[t, :, :, 1] = np.copy(last_resid)
            last_resid[present_t] = np.copy(residuals[t, present_t])
        lag_amount[present_t] = 0
        if max_delta is not None:
            last_val[lag_amount >= max_delta] = np.nan
    return sufficient_statistics, deltas


def impute_chars(char_data, imputed_chars, residuals=None, 
                 suff_stat_method='None', constant_beta=False):
    """
    run the imputation for a given configuration
    Parameters
    ----------
        char_data : the time series panel, T x N x L
        imputed_chars: the cross-sectional imputation of the time series panel
        residuals=None : Optional, residuals T x N x L, the residuals the factor model applied to the time
                        series panel
        suff_stat_method=None : Optional, the type of information to add to the cross sectional panel in the 
                        imputation
        constant_beta=False: whether or not to allow time variation in the loadings of the model
    """
    if suff_stat_method == 'last_val':
        suff_stats, _ = get_sufficient_statistics_last_val(char_data, max_delta=None,
                                                                           residuals=residuals)
        if len(suff_stats.shape) == 3:
            suff_stats = np.expand_dims(suff_stats, axis=3)
                
    elif suff_stat_method == 'None':
        suff_stats = None
            
    if suff_stats is None:
        return imputed_chars
    else:
        return impute_beta_combined_regression(
            char_data, imputed_chars, sufficient_statistics=suff_stats, 
            constant_beta=constant_beta
        )

def impute_beta_combined_regression(characteristics_panel, xs_imps, sufficient_statistics=None, 
                                    constant_beta=False, get_betas=False, gamma_ts=None, use_factors=False, reg=None):
    """
    run the imputation regression for a given configuration
    Parameters
    ----------
        char_data : the time series panel, T x N x L
        xs_imps: the cross-sectional imputation of the time series panel
        sufficient_statistics=None : Optional, the information to add to the cross sectaial panel in the imputation
        constant_beta=False: whether or not to allow time variation in the loadings of the model
        get_betas=False: whether or not to return the learned betas
        
    """
    T, N, L = characteristics_panel.shape
    K = 0
    if xs_imps is not None:
        K += 1
    if sufficient_statistics is not None:
        K += sufficient_statistics.shape[3]

    betas = np.zeros((T, L, K))
    imputed_data = np.copy(characteristics_panel)
    imputed_data[:,:,:]=np.nan
    
    for l in range(L):
        fit_suff_stats = []
        fit_tgts = []
        inds = []
        curr_ind = 0
        all_suff_stats = []
        
        for t in range(T):
            inds.append(curr_ind)
            
            if xs_imps is not None:
                suff_stat = np.concatenate([xs_imps[t,:,l:l+1], sufficient_statistics[t,:,l]], axis=1)
            else:
                suff_stat = sufficient_statistics[t,:,l]
            
            available_for_imputation = np.all(~np.isnan(suff_stat), axis=1)
            available_for_fit = np.logical_and(~np.isnan(characteristics_panel[t,:,l]),
                                                  available_for_imputation)
            all_suff_stats.append(suff_stat)

            fit_suff_stats.append(suff_stat[available_for_fit, :])
            fit_tgts.append(characteristics_panel[t,available_for_fit,l])
            curr_ind += np.sum(available_for_fit)
        
        
        inds.append(curr_ind)
        fit_suff_stats = np.concatenate(fit_suff_stats, axis=0)
        fit_tgts = np.concatenate(fit_tgts, axis=0)
        
        if constant_beta:

            beta = np.linalg.lstsq(fit_suff_stats, fit_tgts, rcond=None)[0]
                
            betas[:,l,:] = beta.reshape(1, -1)
        else:
            for t in range(T):
                beta_l_t = np.linalg.lstsq(fit_suff_stats[inds[t]:inds[t+1]],
                                       fit_tgts[inds[t]:inds[t+1]], rcond=None)[0]
                betas[t,l,:] = beta_l_t
                if np.any(np.isnan(beta_l_t)):
                    print("should be no nans, t=", t,)
                
        for t in range(T):
            beta_l_t = betas[t,l]
            suff_stat = all_suff_stats[t]
            available_for_imputation = np.all(~np.isnan(suff_stat), axis=1)
            imputed_data[t,available_for_imputation,l] = suff_stat[available_for_imputation,:] @ beta_l_t
            
    if get_betas:
        return imputed_data, betas
    else:
        return imputed_data



def get_oos_estimates_given_loadings(chars, reg, Lmbda, time_varying_lmbda=False, get_factors=False):
    """
    Generate the finite-sample correction to the cross-sectionally imputed data
    Parameters
    ----------
        chars : the time series panel, T x N x L
        Lmbda : the loadings in the Xiong - Pelger model
        time_varying_lmbda=False: whether or the loadings are time varying
        get_factors=False: whether or not to return the factors, or the imputed values        
    """
    C = chars.shape[-1]
    def impute_t(t_chars, reg, C, Lmbda, get_factors=False):
        if not get_factors:
            imputation = np.copy(t_chars) * np.nan
        else:
            imputation = np.zeros((t_chars.shape[0], t_chars.shape[1], Lmbda.shape[1])) * np.nan
        mask = ~np.isnan(t_chars)
        net_mask = np.sum(mask, axis=1)
        K = Lmbda.shape[1]
        for n in range(t_chars.shape[0]):
            if net_mask[n] == 1:
                imputation[n,:] = 0
            elif net_mask[n] > 1:
                for i in range(C):
                    tmp = mask[n, i]
                    mask[n,i] = False
                    y = t_chars[n, mask[n]]
                    X = Lmbda[mask[n], :]
                    L = np.eye(K) * reg
                    params = np.linalg.lstsq(X.T @ X + L, X.T @ y, rcond=None)[0]
                    if get_factors:
                        imputation[n,i] = params
                    else:
                        imputation[n,i] = Lmbda[i] @ params
                    
                    mask[n,i] = tmp
        return np.expand_dims(imputation, axis=0)
    chars = [chars_t for chars_t in chars]
    
    if time_varying_lmbda:
        imputation = list(Parallel(n_jobs=60)(delayed(impute_t)(chars_t, reg, C, l, get_factors=get_factors) 
                                              for chars_t, l in zip(chars, Lmbda)))
    else:
        imputation = list(Parallel(n_jobs=60)(delayed(impute_t)(chars_t, reg, C, Lmbda, get_factors=get_factors)
                                              for chars_t in chars))
    return np.concatenate(imputation, axis=0)


def simple_impute(char_panel):
    """ 
    Imputes using the last value of the characteristic time series - the forward filling
    
    """
    imputed_panel = np.copy(char_panel)
    imputed_panel[:,:,:] = np.nan
    imputed_panel[0] = np.copy(char_panel[0])

    for t in range(1, imputed_panel.shape[0]):
        present_t_l = ~np.isnan(char_panel[t-1])
        imputed_t_1 = ~np.isnan(imputed_panel[t-1])
        imputed_panel[t, present_t_l] = char_panel[t-1, present_t_l]
        imputed_panel[t, np.logical_and(~present_t_l,
                                        imputed_t_1)] = imputed_panel[t-1,
                                                                      np.logical_and(~present_t_l, imputed_t_1)]
        imputed_panel[t, ~np.logical_or(imputed_t_1, present_t_l)] = np.nan 

    return imputed_panel


def xs_median_impute(char_panel):
    """ 
    Imputes using the cross-sectional median for each time period and characteristic
    
    """

    imputed_panel = np.copy(char_panel)
    for t in range(imputed_panel.shape[0]):
        for c in range(imputed_panel.shape[2]):
            present_t_1 = ~np.isnan(char_panel[t,:,c])
            imputed_panel[t,:,c] = np.median(char_panel[t, present_t_1, c])
    return imputed_panel

def impute_em(X, max_iter=50, eps=1e-08):
    """ 
    
    
    """
    nr, nc = X.shape
    C = np.isnan(X) == False 

    # Collect M_i and O_i's 
    one_to_nc = np.arange(1, nc+1, step=1)
    M = one_to_nc * (C==False) - 1
    O = one_to_nc * C - 1

    # Generate Mu_0 and Sigma_0
    Mu = np.nanmean(X, axis=0)
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    S = np.cov(X[observed_rows, ].T)
    if np.isnan(S).any():
        S = np.diag(np.nanvar(X, axis = 0))

    # Start updating
    Mu_tilde, S_tilde = {}, {}
    X_tilde = X.copy()
    no_conv = True 
    iteration = 0 
    while no_conv and iteration < max_iter:
        for i in range(nr):
            S_tilde[i] = np.zeros(nc ** 2).reshape(nc, nc)
            if set(O[i, ]) != set(one_to_nc - 1): # missing component exists
                M_i, O_i = M[i, ][M[i, ] != -1], O[i,][O[i, ] != -1]
                S_MM = S[np.ix_(M_i, M_i)]
                S_MO = S[np.ix_(M_i, O_i)]
                S_OM = S_MO.T 
                S_OO = S[np.ix_(O_i, O_i)]
                Mu_tilde[i] = Mu[np.ix_(M_i)] +\
                    S_MO @ np.linalg.inv(S_OO) @\
                    (X_tilde[i, O_i] - Mu[np.ix_(O_i)])
                X_tilde[i, M_i] = Mu_tilde[i]
                S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
                S_tilde[i][np.ix_(M_i, M_i)] = S_MM_O
        Mu_new = np.mean(X_tilde, axis=0)
        S_new = np.cov(X_tilde.T, bias=1) +\
            reduce(np.add, S_tilde.values()) / nr 
        no_conv = \
            np.linalg.norm(Mu - Mu_new) >= eps or\
            np.linalg.norm(S - S_new, ord=2) >= eps 
        
        Mu = Mu_new
        S = S_new 
        iteration += 1 

    result = {
        'mu': Mu,
        'Sigma': S,
        'X_imputed': X_tilde,
        'C': C,
        'iteration': iteration
    }

    return result 

def project_percentile_data(observed_data, percentile_data):
    # Flatten the input arrays if necessary
    observed_data = np.asarray(observed_data).flatten()
    percentile_data = np.asarray(percentile_data).flatten()

    # Fit the KDE to the observed data
    kde = KernelDensity().fit(observed_data.reshape(-1,1))

    # Estimate the CDF of the observed data
    observed_cdf = np.cumsum(np.exp(kde.score_samples(observed_data.reshape(-1,1))))
    observed_cdf /= observed_cdf[-1]

    # Map the projectile data to the observed data space
    # projected_data = np.interp(percentile_data, observed_cdf, observed_data) # simple way which uses linear interpolation

    # Fit a KNN regressor to map the percentile data
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(np.array(observed_cdf).reshape(-1,1), observed_data.reshape(-1,1))

    # Map the percentile data to the observed data space using KNN
    projected_data = (knn.predict(percentile_data.reshape(-1,1))).reshape(-1,)

    return projected_data


def project_percentile_data_with_knn(observed_data, x_percentile_data, percentile_data):
    # Flatten the input arrays if necessary
    observed_data = np.asarray(observed_data).flatten()
    x_percentile_data = np.asarray(x_percentile_data).flatten()
    percentile_data = np.asarray(percentile_data).flatten()

    # Fit a KNN regressor to map the percentile data
    knn = KNeighborsRegressor(n_neighbors=25)
    knn.fit(np.array(x_percentile_data).reshape(-1,1), observed_data.reshape(-1,1))

    input_percentile_data = percentile_data.copy()
    # get missing mask
    missing_mask = np.isnan(input_percentile_data)

    input_percentile_data = np.nan_to_num(input_percentile_data) # convert all nan to zero

    # Map the percentile data to the observed data space using KNN
    projected_data = (knn.predict(input_percentile_data.reshape(-1,1)))

    # now we return the nan back to the predictions
    projected_data[missing_mask] = np.nan

    return projected_data.reshape(-1,)


def invert_cross_sectional_percentiles_with_knn(percentile_panel, partially_observed_panel):
    T, N, L = percentile_panel.shape
    inverted_panel = np.zeros((T,N,L))
    percentile_panel_missing_mask = np.isnan(percentile_panel) # capturing the missing mask from the panel after imputation
    for t in range(T):
        # Extract the cross-sectional data at time t 
        for l in range(L): # Looping through the periodic characteristics
            observed_data = partially_observed_panel[t,:,l] # might contain missing values
            if np.isnan(observed_data).any():
                missing_mask = np.isnan(observed_data)
                observed_data = observed_data[~missing_mask] # getting only the fully observed_data 
            else: # no missing components
                observed_data = observed_data # do nothing


            percentile_data = percentile_panel[t,:,l]

            x_percentile_data = percentile_data.copy()
            x_percentile_data = x_percentile_data[~missing_mask] # get corresponding percentile of fully observed data


            # look for more missing values in x_percentile_data and effect it in both x_percentile_data and observed data
            more_missing_mask  = np.isnan(x_percentile_data)

            x_percentile_data_ = x_percentile_data.copy()
            x_percentile_data_ = x_percentile_data_[~more_missing_mask]

            observed_data = observed_data[~more_missing_mask]

            input_missing_mask = np.isnan(percentile_data)
            percentile_data_ = percentile_data.copy()
            percentile_data_ = percentile_data[~input_missing_mask]

            inverted_panel[t,:,l][~input_missing_mask] = project_percentile_data_with_knn(observed_data, x_percentile_data_, percentile_data_)

    # fetch the mask and only fill those places
    mask = np.isnan(partially_observed_panel)

    full_data = partially_observed_panel.copy() # just initialize

    full_data[mask] = inverted_panel[mask]

    # return the missing mask in the percentile panel to the data
    full_data[percentile_panel_missing_mask] = np.nan

    return full_data 



# sort dataframe

def sort_dataframe(df, N_column_name):
    """ 
    code to sort the dataframe based on the column names, N column and the date
    - helps with easy reproducibility of the code
    """
    columns_to_exclude = ['date', N_column_name]
    sorted_columns = sorted([col for col in df.columns if col not in columns_to_exclude])
    sorted_columns = ['date', N_column_name] + sorted_columns 
    sorted_df = df[sorted_columns].sort_values(by=['date', N_column_name], ascending=True)
    return sorted_df


# compute performance metrics

def compute_rmse(truth_panel, predicted_panel):
    """ 
    Compute root mean squared error
    """
    resids = truth_panel - predicted_panel
    error = np.sqrt(np.nanmean(np.square(resids)))
    return error 


def compute_r2(truth_panel, predicted_panel):
    """ 
    Compute R^2
    """
    imputed = predicted_panel
    tgt = np.copy(truth_panel)
    tgt[np.isnan(imputed)] = np.nan 

    overall_r2 = np.nanmean(1 - np.nansum(np.square(tgt - imputed), axis=(1,2)) /
                            np.nansum(np.square(tgt), axis=(1,2)), axis=0)
    return overall_r2
 


# Mask generator

def get_random_masks(present_chars, p):
    """ 
    get a fully random mask over observed characteristics
    """
    flipped = np.random.binomial(1,p, size=present_chars.shape) == 1
    flipped = np.logical_and(~np.isnan(present_chars), flipped)
    return flipped 

def generate_MAR_missing_data(rank_chars):
    """ 
    Generate a MAR masked dataset
    --- fixing the value of p at 0.1 # can experiment with different values later
    """
    np.random.seed(42)
    T,N,L = rank_chars.shape 
    update_chars = np.copy(rank_chars)

    random_nan_mask = get_random_masks(update_chars, p=0.1)
    print(random_nan_mask.shape)
    print(np.max(np.sum(random_nan_mask, axis=2)),
          np.sum(random_nan_mask, axis=(0,1)) / (np.sum(~np.isnan(update_chars), axis=(0,1))))
    masked_chars = np.copy(update_chars)
    masked_chars[random_nan_mask] = np.nan 

    masked_lagged_chars = np.copy(rank_chars)
    only_missing_chars = np.copy(rank_chars)
    only_missing_chars[:,:,:] = np.nan 

    for c in range(L):
        for t in range(random_nan_mask.shape[0]):
            missing = random_nan_mask[t,:,c]
            only_missing_chars[t,missing,c] = np.copy(masked_lagged_chars[t,missing,c])
            masked_lagged_chars[t,missing,c] = np.nan 
    only_mimissing_chars = np.copy(rank_chars)
    only_mimissing_chars[~random_nan_mask] = np.nan 

    return masked_lagged_chars, only_mimissing_chars 



# function to generate simulated data

def simulate_nan(X, nan_rate):
    """ 
    
    """

    # Create C matrix; entry is False if missing, and True if observed
    X_complete = X.copy()
    nr, nc = X_complete.shape
    C = np.random.random(nr * nc).reshape(nr, nc) > nan_rate

    # Check for which i's we have all components become missing
    checker = np.where(sum(C.T) == 0)[0]
    if len(checker) == 0:
        # Every X_i has at least one component that is observed,
        # which is what we want
        X_complete[C == False] = np.nan 
    else:
        # Otherwise, randomly "revive" some components in such X_i's 
        for index in checker:
            reviving_components = np.random.choice(
                nc,
                int(np.ceil(nc * np.random.random())),
                replace=False
            )
            C[index, np.ix_(reviving_components)] = True
        X_complete[C == False] = np.nan 

    result = {
        'X': X_complete,
        'C': C,
        'nan_rate': nan_rate,
        'nan_rate_actual': np.sum(C == False) / (nr * nc)
    }

    return result
