# Imports
import os
import h5py
import sys
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, PoissonRegressor
from sklearn.metrics import explained_variance_score
import argparse
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn import metrics, model_selection
import pandas as pd

# Modules
from predict_utils import compute_regression_rates

sys.path.append('../../Code/')
from path_utils import PATH_TO_SPIKE_REGRESS_DATA_PASSIVE

# Paths
PATH_TO_FIGS = '..'

params = {
    'axes.labelsize': 13,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'text.usetex': False,
    'figure.figsize': [7, 7],
    'font.size': 15,
    'lines.markersize': 10
}
plt.rcParams.update(params)

def remove_outlier_all(y_hat, std_range=10):
    """
    Remove outliers in predicted data.
    :param y_hat:
    :return:
    """
    y_hat_tmp = y_hat.copy()
    mean_pp = np.mean(y_hat_tmp.reshape(-1))
    std_pp = np.std(y_hat_tmp.reshape(-1))
    ind_pp = np.where(np.abs(y_hat - mean_pp) > std_pp * std_range)
    for jj in range(len(ind_pp[0])):
        ind_trial, ind_t = ind_pp[0][jj], ind_pp[1][jj]
        if ind_t - 2 < 0:
            ind_init = 0
        else:
            ind_init = ind_t - 2
        if ind_t + 2 > len(y_hat_tmp):
            ind_end = len(y_hat_tmp)
        elif ind_t + 2 < ind_pp[1][-1]:
            ind_end = ind_pp[1][-1] +1
        else:
            ind_end = ind_t + 2
        value_tmp = np.median(np.concatenate((y_hat_tmp[ind_trial][ind_init:ind_t], y_hat_tmp[ind_trial][ind_t + 1:ind_end]), 0))
        if not np.isnan(value_tmp):
            y_hat_tmp[ind_trial][ind_t] = value_tmp
    return y_hat_tmp

def fit_poisson_glm(x_train, x_test, y_train, y_test, neur_idx):
    """
    Fit Poisson GLM of neural spike rates, for one neuron, given train/test splits.
    :param x_train: Input features, train.
    :param x_test: Input features, test.
    :param y_train: Neural spike rates, train.
    :param y_test: Neural spike rates, test.
    :param neur_idx: Neuron index.
    :return: A list of results.
    """

    # Take rates for that neuron
    y_train_idx, y_test_idx = y_train[:, neur_idx], y_test[:, neur_idx]

    # POISSON GENERALIZED LINEAR MODEL
    poisson_glm = PoissonRegressor(fit_intercept=True,
                                   tol=1e-4,
                                   warm_start=False,
                                   max_iter=1000,
                                   alpha=0.0)  # fixed alpha: can't use GridSearchCV with multiprocessing Pool

    poisson_glm.fit(x_train, y_train_idx)

    y_hat_train = poisson_glm.predict(x_train)
    y_hat_test = poisson_glm.predict(x_test)
    d2_score_tr = poisson_glm.score(x_train, y_train_idx)  # /2
    d2_score_te = poisson_glm.score(x_test, y_test_idx)  # /2
    ev_score_train = explained_variance_score(y_train_idx, y_hat_train)
    ev_score_test = explained_variance_score(y_test_idx, y_hat_test)

    # STORE RESULTS
    result_list = [list(y_test),
                   list(y_hat_test.astype(float)),
                   poisson_glm.intercept_.astype(float),
                   list(poisson_glm.coef_.astype(float)),
                   ev_score_train.astype(float),
                   ev_score_test.astype(float),
                   d2_score_tr.astype(float),
                   d2_score_te.astype(float)]

    return result_list

def kf_alpha_reg(x_tr, y_tr, train_ids_act, idx_trial_train_concat, n_splits = 5):
    kf = model_selection.KFold(n_splits=n_splits,shuffle=True,random_state=42)

    alphas_list = np.logspace(-2, 5, num=15)

    all_scores = np.zeros((n_splits,alphas_list.shape[0],y_tr.shape[1]))

    for j,(train_index, test_index) in enumerate(kf.split(train_ids_act)):
        train_index = train_ids_act[train_index]
        test_index = train_ids_act[test_index]
        # print(train_index)
        all_train_idx = [ii for ii in range(len(idx_trial_train_concat)) if idx_trial_train_concat[ii] in train_index]
        all_test_idx = [ii for ii in range(len(idx_trial_train_concat)) if idx_trial_train_concat[ii] in test_index]
        # print(len(all_train_idx)+len(all_test_idx))

        for i,alpha in enumerate(alphas_list):
            
            cv_MIMO = Ridge(alpha = alpha,max_iter=1000, fit_intercept=True, random_state=42)
            cv_MIMO.fit(x_tr[all_train_idx,:], y_tr[all_train_idx,:]) 
            cv_preds = cv_MIMO.predict(x_tr[all_test_idx,:])
            all_scores[j,i,:] = metrics.explained_variance_score(y_tr[all_test_idx,:], cv_preds, multioutput='raw_values')
    idx_alphas = np.argmax(all_scores.mean(axis=0),axis=0)
    return alphas_list[idx_alphas]

def make_layer_predictions(layer_activations, spike_counts, mvt_durations, monkey_name, session_date,
                           model_config=None, layer_idx=None, is_passive=False, align=100, current_temp_factor=None,
                           plot_pred=False):
    """
    Script to make predictions of neural rates using a set of neural network activations i.e. one layer,
    for all neurons.

    :param layer_activations: Input activations of dimension (trials, nspatial, ntemporal, nfilters).
    :param spike_counts: Binned spike firing rates of dimension (trials, ntemporal, neurons).
    :param mvt_durations: Array of durations of each trials.
    :param monkey_name: Name of monkey, capitalized first letter.
    :param session_date: Monkey session date DDMMYYYY.
    :param model_config: Task-driven model configuration dict.
    :param layer_idx: Index of task-driven DNN layer.
    :param is_passive: Whether these are passive predictions.
    :param align: Alignment index in muscle input when running activations.
    :param current_temp_factor: Temporal factor of current layer.
    :param plot_pred: Whether to plot example model predictions, for some trials, for that model.

    :return: A dictionary containing results for linear predictivity models.
    """

    ### Remove outlier for activations
    list_layer_pc = [layer_activations[:,:,pc_idx] for pc_idx in range(layer_activations.shape[2])]
    layer_activations = map(remove_outlier_all,list_layer_pc)
    layer_activations = np.asarray(list(layer_activations)).swapaxes(0,1).swapaxes(1,2)

    ## TRAIN-TEST TRIAL SPLITS for ACTIVE data FROM SPIKE REGRESSION DATASETS

    ## Get train/test indexes for multiple conditions
    idx_trials_path = os.path.join(PATH_TO_SPIKE_REGRESS_DATA_PASSIVE,
                                   'idx_split_{}_passive.p'.format(monkey_name))

    all_indexes = pd.read_pickle(idx_trials_path)
    train_ids_act = all_indexes['train_idx']
    test_ids_act = all_indexes['test_idx']

    layer_activations_tr = layer_activations[train_ids_act]
    spike_counts_tr = spike_counts[train_ids_act]
    mvt_durations_tr = mvt_durations[train_ids_act]

    layer_activations_te = layer_activations[test_ids_act]
    spike_counts_te = spike_counts[test_ids_act]
    mvt_durations_te = mvt_durations[test_ids_act]

    mvt_durations_tr = np.array(mvt_durations_tr,dtype=np.int)
    mvt_durations_te = np.array(mvt_durations_te,dtype=np.int)

    n_trials = len(layer_activations)  # TODO: this should be obsolete

    print('N trial train: ',layer_activations_tr.shape[0])
    print('N trial test: ',layer_activations_te.shape[0])

    ## CONCATENATE TRAIN/TEST TRIAL DATA
    x_train, y_train, _ = compute_regression_rates(layer_activations_tr,
                                                spike_counts_tr,  # [:n_trials]
                                                mvt_durations_tr,
                                                model_config,
                                                layer_idx,
                                                align,
                                                current_temp_factor)

    x_test, y_test, _ = compute_regression_rates(layer_activations_te,
                                              spike_counts_te,  # [:n_trials]
                                              mvt_durations_te,
                                              model_config,
                                              layer_idx,
                                              align,
                                              current_temp_factor)

    # Init. result storage
    result_dict_lm = dict(test_rates=[], test_preds=[], intercepts=[], weights=[], alphas=[], ev_train=[], ev_test=[])

    ## FIT MULTINEURON REGRESSION & TUNE HYPERPARAMETERS
    print('Regression split sizes:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    ### Get the index trial for each timepoint of the future concatenate trainset
    idx_trial_train_concat = []
    for idx_mvt,t in enumerate(train_ids_act):
        idx_trial_train_concat.extend(mvt_durations_tr[idx_mvt]*[t])
    idx_trial_train_concat = np.array(idx_trial_train_concat)

    # # Standarize features
    scaler = StandardScaler().fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    alpha_array = kf_alpha_reg(x_train, y_train, train_ids_act, idx_trial_train_concat, n_splits = 5)
    ridge_model = Ridge(alpha=np.array(alpha_array),
                                # normalize=True,
                                max_iter=1000,
                                fit_intercept=True,
                                random_state=42)
    ridge_model.fit(x_train, y_train)

    y_hat_train = ridge_model.predict(x_train)
    ev_score_train = explained_variance_score(y_train, y_hat_train, multioutput='raw_values')

    # Test predictions per neuron
    y_hat_test = ridge_model.predict(x_test)
    ev_score_test = explained_variance_score(y_test, y_hat_test, multioutput='raw_values')

    print('Mean EV scores train:', np.mean(ev_score_train))
    print('Mean EV scores test:', np.mean(ev_score_test))

    # PLOTTING FOR LAYER PREDICTIONS FIGURES
    if plot_pred:
        # Create figure folder
        print(model_config.keys())
        model_suffix = str(model_config['name']) +'_'+ str(layer_idx)
        monkey_suffix = monkey_name+'_'+str(session_date)
        saving_path = os.path.join(PATH_TO_FIGS, model_suffix+'_'+monkey_suffix)

        # Make plots
        plot_neural_predictions_layer(test_rates=y_test,
                                      predicted_rates=y_hat_test,
                                      mvt_durations=mvt_durations_te,
                                      temp_factor = current_temp_factor,
                                      saving_path=saving_path)

    # STORE LM RESULTS
    result_dict_lm['test_rates'] = list(y_test)
    result_dict_lm['test_preds'] = list(y_hat_test.astype(float))
    result_dict_lm['intercepts'] = list(ridge_model.intercept_.astype(float))
    result_dict_lm['weights'] = list(ridge_model.coef_.astype(float))
    result_dict_lm['alphas'] = ridge_model.alpha.astype(float) #ridge_cv.alpha_.astype(float)
    result_dict_lm['ev_train'] = list(ev_score_train.astype(float))
    result_dict_lm['ev_test'] = list(ev_score_test.astype(float))

    # Layer LM and GLM results
    result_dict = dict(lm=result_dict_lm)  # ,

    return result_dict


if __name__ == '__make_layer_predictions__':
    parser = argparse.ArgumentParser(
        description='Make neural predictions using single-layer activations.')

    parser.add_argument('--act', type=np.ndarray, help='Input activation array.', required=True)
    parser.add_argument('--spikes', type=np.ndarray, help='Output spike array', required=True)
    args = parser.parse_args()

    make_layer_predictions(args.act, args.spikes)
