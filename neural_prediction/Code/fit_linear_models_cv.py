# Imports
import os
import argparse
import sys
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, explained_variance_score
from collections import defaultdict
from sklearn.linear_model import Ridge, RidgeCV, PoissonRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn import metrics, model_selection

# Modules
from neural_utils import load_monkey_data, get_neuraldata_path, from_name_sortfields, get_result_path, fit_cosine_curve
from neural_utils import collect_tuning_data, get_figfolder_path
sys.path.append(os.path.join(sys.path[0],'../'))
sys.path.append(os.path.join(sys.path[0],'../../neural_prediction/Code/'))
from global_utils import is_CObump_session
from predict_utils import load_monkey_datasets

sys.path.append('../../code/')
from path_utils import PATH_TO_NEURAL_DATA, PATH_TO_DATASPLITS, PATH_TO_MATLAB_DATA, PATH_TO_SAVE_LINEAR


def window(x):
    d = []
    for i in range(x.shape[1]):
        d.append(np.convolve(x[:, i], np.ones(5) / 5, mode='same'))
    return np.asarray(d).T


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def replace_nan_single(y):
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def replace_nan(mus_len_tmp):
    if np.any(np.isnan(mus_len_tmp)):
        idx_x, idx_y = np.where(np.isnan(mus_len_tmp))
        unique_y = np.unique(idx_y)

        for jj in unique_y:
            mus_len_tmp[:, jj] = replace_nan_single(mus_len_tmp[:, jj])
    return mus_len_tmp

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

def main(monkey_name, session_date, active_start, is_short):
    """
    Fit tuning curve models for a monkey session dataset.

    :param monkey_name: Monkey name, capitalized first letter.
    :param session_date: Monkey session date, DDMMYYYY.
    :param active_start: Trial event from which wet get behavioural/neural data (mvt, cue).
    :param is_short: Whether we take fraction of active data, from active start.
    :return:
    """

    if is_short == 'y':
        is_short = True
        print('Short time tuning models in active.')
    elif is_short == 'n':
        print('Whole active reach predictions.')
        is_short = False

    if not is_CObump_session(session_date):
        print(
            'Data does not have passive task - ignore session or adapt script.')  # TODO: temporary message, to fix
        # and adapt
        return

    path_to_save_linear_folder = os.path.join(PATH_TO_SAVE_LINEAR,'active','linear_models')
    if not os.path.exists(path_to_save_linear_folder):
        os.makedirs(path_to_save_linear_folder)

    # Load data
    data_df = load_monkey_data(PATH_TO_MATLAB_DATA,
                               monkey_name,
                               session_date,
                               keep_kinematics=True,
                               keep_spike_data=True,
                               use_new_muscle=True)
    spike_field, _, _ = from_name_sortfields(monkey_name)

    bump_ids = data_df[~np.isnan(data_df.bumpDir)].index

    ## COLLECT DATA
    session_sets = collect_tuning_data(data_df,
                                       spike_field,
                                       active_start=active_start,
                                       is_short=is_short)
    print('Collected:', session_sets._fields)

    # Passive
    rates_pas = session_sets.PassiveSets.rates_trial
    dir_pas_trial = session_sets.PassiveSets.dir_trial
    rates_t_pas = session_sets.PassiveSets.rates_t
    dir_pas = session_sets.PassiveSets.dir
    pos_pas = session_sets.PassiveSets.pos
    pos_pas_norm = session_sets.PassiveSets.pos_norm
    vel_pas = session_sets.PassiveSets.vel
    vel_pas_norm = session_sets.PassiveSets.vel_norm
    acc_pas = session_sets.PassiveSets.acc
    acc_pas_norm = session_sets.PassiveSets.acc_norm
    force_pas = session_sets.PassiveSets.force
    force_pas_norm = session_sets.PassiveSets.force_norm

    m_len_pas = session_sets.PassiveSets.m_len
    m_vel_pas = session_sets.PassiveSets.m_vel

    # Active
    rates_act_start_end = session_sets.ActiveSets.rates_trial
    dir_act_start_end_trial = session_sets.ActiveSets.dir_trial
    rates_t_act_start_end = session_sets.ActiveSets.rates_t
    dir_act_start_end = session_sets.ActiveSets.dir
    pos_act = session_sets.ActiveSets.pos
    pos_act_norm = session_sets.ActiveSets.pos_norm
    vel_act = session_sets.ActiveSets.vel
    vel_act_norm = session_sets.ActiveSets.vel_norm
    acc_act = session_sets.ActiveSets.acc
    acc_act_norm = session_sets.ActiveSets.acc_norm
    force_act = session_sets.ActiveSets.force
    force_act_norm = session_sets.ActiveSets.force_norm

    j_ang_act_start_end = session_sets.ActiveSets.j_ang
    j_vel_act_start_end = session_sets.ActiveSets.j_vel

    m_len_act_start_end = session_sets.ActiveSets.m_len
    m_vel_act_start_end = session_sets.ActiveSets.m_vel

    if monkey_name == 'Chips':
        active_length = 0
        active_start = "mvt"
        align = 100 
        control_dict = {'permut_m': False,
                        'permut_t': False,
                        'constant_input': False}

        datasets = load_monkey_datasets(PATH_TO_NEURAL_DATA,
                                            monkey_name,
                                            session_date,
                                            False,
                                            active_start=active_start,
                                            active_length=active_length,
                                            align=align,
                                            control_dict=control_dict)

        m_len_vel_start_end_all = datasets['active']['muscle_coords'].copy()

        m_len_act_start_end1 = []
        m_vel_act_start_end1 = []
        for ii in range(len(m_len_vel_start_end_all)):
            t_end_tmp = m_len_act_start_end[ii].shape[0]
            m_len_act_start_end1.append(m_len_vel_start_end_all[ii,:,100:100+t_end_tmp,0].T/1000)
            m_vel_act_start_end1.append(m_len_vel_start_end_all[ii,:,100:100+t_end_tmp,1].T/1000)
        m_len_act_start_end = m_len_act_start_end1.copy()
        m_vel_act_start_end = m_vel_act_start_end1.copy()

    # INIT NEURON RESULT CONTAINER
    kin_vars_act = ['dir_active', 'pos_active', 'pos_norm_active', 'vel_active', 'vel_norm_active', 'acc_active',
                    'acc_norm_active', 'force_active', 'force_norm_active',
                    'j_ang_active', 'j_vel_active', 'm_len_active', 'm_vel_active']

    kin_vars_regression = kin_vars_act

    res_dict_lm = defaultdict(list, {key: [] for key in kin_vars_regression})
    res_dict_glm = defaultdict(list, {key: [] for key in kin_vars_regression})
    res_dict_glm_r2 = defaultdict(list, {key: [] for key in kin_vars_regression})

    ## REMOVE TRIALS TO BE EXCLUDED
    bad_trials_path = os.path.join(PATH_TO_DATASPLITS,
                                   'excludedtrials_{}_{}_center_out_spikes_datadriven.p'.format(monkey_name,
                                                                                                           session_date))
    excluded_trials = pd.read_pickle(bad_trials_path)
    excluded_trials_ids = list(excluded_trials['bad_trials']) + list(excluded_trials['too_long'])

    ### TRAIN-TEST TRIAL SPLITS for ACTIVE and PASSIVE
    # GET SPLITS FROM SPIKE REGRESSION DATASETS
    print('Get train/test splits from spike regression datasets')
    # Train splits
    path_dataset = os.path.join(PATH_TO_DATASPLITS,
                                'dataset_train_{}_{}_center_out_spikes_datadriven.hdf5'.format(monkey_name,
                                                                                               session_date))
    file = h5py.File(path_dataset, 'r')
    train_ids_act = file['indices_info'][()]
    file.close()
    # Test splits
    path_dataset = os.path.join(PATH_TO_DATASPLITS,
                                'dataset_test_{}_{}_center_out_spikes_datadriven.hdf5'.format(monkey_name,
                                                                                              session_date))
    file = h5py.File(path_dataset, 'r')
    test_ids_act = file['indices_info'][()]
    file.close()

    fname_end = '_' + active_start
    if is_short:
        fname_end += '_short'
    fname_end += '_goodsplits_new'  # matching train/test splits

    train_ids_act = np.asarray([t for t in train_ids_act if t not in excluded_trials_ids])
    test_ids_act = np.asarray([t for t in test_ids_act if t not in excluded_trials_ids])

    ### Get the index trial for each timepoint of the future concatenate trainset
    idx_trial_train_concat = []
    for t in train_ids_act:
        idx_trial_train_concat.extend(m_len_act_start_end[t].shape[0]*[t])
    idx_trial_train_concat = np.array(idx_trial_train_concat)

    ### LINEAR TUNING MODELS - SINGLE-TYPE VARIABLES

    array_list = [dir_act_start_end, pos_act, pos_act_norm, vel_act, vel_act_norm, acc_act, acc_act_norm,
                  force_act, force_act_norm,
                  j_ang_act_start_end, j_vel_act_start_end,
                  m_len_act_start_end, m_vel_act_start_end]

    print('Fitting single variable models...')
    for kin_array, kin_var in zip(array_list, kin_vars_regression):
        print('Fitting', kin_var)

        # DATA PREPARATION
        # Set split indices and concatenate active trials
        if 'active' in kin_var:

            kin_array_concat_all = []
            target_rates = []
            trial_ids_act = list(train_ids_act) + list(test_ids_act)

            # CORRECT AND STACK DATA
            for ii in trial_ids_act:
                kin_array_tmp = replace_nan(np.asarray(np.vstack(kin_array[ii])))
                kin_array_concat_all.append(kin_array_tmp)

                target_rates.append(rates_t_act_start_end[ii])
            kin_array_concat_all = np.vstack(kin_array_concat_all)
            target_rates = np.vstack(target_rates)

            # INIT. STORAGE
            kin_array_concat_tr = []
            target_rates_concat_tr = []
            kin_array_concat_te = []
            target_rates_concat_te = []

            # CONCATENATE DATA
            for t in range(len(kin_array)):
                if t in train_ids_act:
                    kin_array_concat_tr.extend(kin_array[t])
                    target_rates_concat_tr.extend(rates_t_act_start_end[t])
                elif t in test_ids_act:
                    kin_array_concat_te.extend(kin_array[t])
                    target_rates_concat_te.extend(rates_t_act_start_end[t])
            # RESHAPE INPUT DATA
            # Sinusoidal formulation for directional tuning model
            if 'dir' in kin_var:
                kin_array_tr = np.stack((np.sin(kin_array_concat_tr), np.cos(kin_array_concat_tr)), axis=0).T
                kin_array_te = np.stack((np.sin(kin_array_concat_te), np.cos(kin_array_concat_te)), axis=0).T
            else:
                kin_array_tr = np.asarray(kin_array_concat_tr)
                kin_array_te = np.asarray(kin_array_concat_te)
            # OUTPUT DATA
            y_tr = np.asarray(target_rates_concat_tr)
            y_te = np.asarray(target_rates_concat_te)

        # Reshape based on kinematic feature type, and add constant
        norm_vars = ['pos_norm_active', 'vel_norm_active', 'acc_norm_active', 'force_norm_active']

        # Models with normalized variables
        if kin_var in norm_vars:
            n_predictors = 1
            x_tr = kin_array_tr.reshape(-1, n_predictors)
            x_te = kin_array_te.reshape(-1, n_predictors)

        # Models with x,y components
        else:
            n_predictors = kin_array_tr[0].shape[-1]
            x_tr = kin_array_tr.reshape(-1, n_predictors)
            x_te = kin_array_te.reshape(-1, n_predictors)

        print('Shapes train:', x_tr.shape, y_tr.shape)
        print('Shapes test:', x_te.shape, y_te.shape)

        # Replace some remaining NaNs in 1 muscle length (Snap)
        if kin_var in ['m_len_active','m_vel_active']:
            x_tr[:, 20] = replace_nan_single(x_tr[:, 20])

        scaler = StandardScaler().fit(x_tr)

        x_tr = scaler.transform(x_tr)
        x_te = scaler.transform(x_te)

        alpha_array = kf_alpha_reg(x_tr, y_tr, train_ids_act, idx_trial_train_concat, n_splits = 5)

        # Refit on training using best alpha
        ridge_model = Ridge(alpha=np.array(alpha_array),
                            max_iter=1000,
                            fit_intercept=True,
                            random_state=42)

        ridge_model.fit(x_tr, y_tr)
        y_hat_train = ridge_model.predict(x_tr)
        ev_score_train = explained_variance_score(y_tr, y_hat_train, multioutput='raw_values')

        # Test predictions per neuron
        y_hat_test = ridge_model.predict(x_te)
        ev_score_test = explained_variance_score(y_te, y_hat_test, multioutput='raw_values')
        res_dict_lm[kin_var].append(ev_score_test)

        ### SAVE RESULTS
        # Explained variance
        # Linear

        ### Decomment here
        file_path = os.path.join(path_to_save_linear_folder,
                                'tuning_{}_{}_{}_lm_ev_df{}.pkl'.format(monkey_name, session_date, kin_var, fname_end))
        res_df = pd.DataFrame()
        res_df['kin_var'] = kin_var
        res_df['ev_score_train'] = ev_score_train
        res_df['ev_score_test'] = ev_score_test
        res_df.to_pickle(file_path)
        print('Results saved in :', file_path)


    ### LINEAR TUNING MODELS - MULTI-TYPE VARIABLES
    # Combine variable datasets

    array_list_act = [np.stack((pos_act, vel_act), axis=-1),
                      np.stack((pos_act, acc_act), axis=-1),
                      np.stack((pos_act, force_act), axis=-1),
                      np.stack((vel_act, acc_act), axis=-1),
                      np.stack((vel_act, force_act), axis=-1),
                      np.stack((acc_act, force_act), axis=-1),
                      np.stack((pos_act, vel_act, acc_act), axis=-1),
                      np.stack((pos_act, vel_act, acc_act, force_act), axis=-1),
                      np.stack((j_ang_act_start_end, j_vel_act_start_end), axis=-1),
                      np.stack((m_len_act_start_end, m_vel_act_start_end), axis=-1),
                      np.stack((j_ang_act_start_end, j_vel_act_start_end, m_len_act_start_end, m_vel_act_start_end), axis=-1),
                      ]

    array_list_multi = array_list_act

    kin_vars_act = ['pos_vel_active', 'pos_acc_active', 'pos_force_active', 'vel_acc_active', 'vel_force_active',
                    'acc_force_active', 'pos_vel_acc_active', 'pos_vel_acc_force_active',
                    'j_ang_vel_active', 'm_len_vel_active',
                    'joint_muscle_active']


    kin_vars_regression_multi =  kin_vars_act
    print('Fitting multi variable models...')
    for kin_array, kin_var in zip(array_list_multi, kin_vars_regression_multi):
        print('Fitting', kin_var)
        # DATA PREPARATION
        # Concatenate active trials
        if 'active' in kin_var:

            kin_array_concat_all = []
            target_rates = []
            trial_ids_act = list(train_ids_act) + list(test_ids_act)
            for ii in trial_ids_act:
                kin_array_tmp = replace_nan(np.asarray(np.hstack(kin_array[ii])))
                kin_array_concat_all.append(kin_array_tmp)
                target_rates.append(rates_t_act_start_end[ii])
            kin_array_concat_all = np.vstack(kin_array_concat_all)
            target_rates = np.vstack(target_rates)

            kin_array_concat_tr = []
            target_rates_concat_tr = []
            kin_array_concat_te = []
            target_rates_concat_te = []

            for t in range(len(kin_array)):
                if t in train_ids_act:
                    kin_array_concat_tr.extend(np.asarray(np.hstack(kin_array[t])))
                    target_rates_concat_tr.extend(rates_t_act_start_end[t])
                elif t in test_ids_act:
                    kin_array_concat_te.extend(np.asarray(np.hstack(kin_array[t])))
                    target_rates_concat_te.extend(rates_t_act_start_end[t])

        x_tr = np.asarray(kin_array_concat_tr)
        y_tr = np.asarray(target_rates_concat_tr)
        x_te = np.asarray(kin_array_concat_te)
        y_te = np.asarray(target_rates_concat_te)

        # Reshape and add constant
        n_predictors = x_tr[0].shape[-1]
        x_tr = x_tr.reshape(-1, n_predictors)
        x_te = x_te.reshape(-1, n_predictors)

        print('Shapes train:', x_tr.shape, y_tr.shape)
        print('Shapes test', x_te.shape, y_te.shape)

        # Replace some remaining NaNs in 1 muscle length (Snap)
        if kin_var in ['m_len_vel_active']:
            x_tr[:, 20] = replace_nan_single(x_tr[:, 20])
            x_te[:, 20] = replace_nan_single(x_te[:, 20])
            x_tr[:, 59] = replace_nan_single(x_tr[:, 59])
            x_te[:, 59] = replace_nan_single(x_te[:, 59])
        elif kin_var in ['joint_muscle_active']:
            x_tr[:, 34] = replace_nan_single(x_tr[:, 34])
            x_te[:, 34] = replace_nan_single(x_te[:, 34])
            x_tr[:, 73] = replace_nan_single(x_tr[:, 73])
            x_te[:, 73] = replace_nan_single(x_te[:, 73])

        scaler = StandardScaler().fit(x_tr)
        x_tr = scaler.transform(x_tr)
        x_te = scaler.transform(x_te)

        alpha_array = kf_alpha_reg(x_tr, y_tr, train_ids_act, idx_trial_train_concat, n_splits = 5)

        # Refit on training using best alpha
        ridge_model = Ridge(alpha=np.array(alpha_array),
                            # normalize=True,
                            max_iter=1000,
                            fit_intercept=True,
                            random_state=42)
        
        ridge_model.fit(x_tr, y_tr)
        y_hat_train = ridge_model.predict(x_tr)
        ev_score_train = explained_variance_score(y_tr, y_hat_train, multioutput='raw_values')

        # Test predictions per neuron
        y_hat_test = ridge_model.predict(x_te)
        ev_score_test = explained_variance_score(y_te, y_hat_test, multioutput='raw_values')

        res_dict_lm[kin_var].append(ev_score_test)

        ### SAVE RESULTS
        # Explained variance
        # Linear
        file_path = os.path.join(path_to_save_linear_folder,
                                'tuning_{}_{}_{}_lm_ev_df{}.pkl'.format(monkey_name, session_date, kin_var, fname_end))
        res_df = pd.DataFrame()
        res_df['kin_var'] = kin_var
        res_df['ev_score_train'] = ev_score_train
        res_df['ev_score_test'] = ev_score_test
        res_df.to_pickle(file_path)
        print('Results saved in :', file_path)

    # ### SAVE RESULTS

    # Explained variance
    # Linear

    ## Decomment here
    file_path = os.path.join(path_to_save_linear_folder,
                             'tuning_{}_{}_lm_ev_df{}.pkl'.format(monkey_name, session_date, fname_end))
    res_df = pd.DataFrame.from_dict(res_dict_lm, orient='index')
    res_df = res_df.transpose()
    res_df.to_pickle(file_path)
    print('Results saved in :', file_path)

    print('Done fitting tuning models.')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit tuning curve models for a monkey session dataset.')

    parser.add_argument('--monkey', type=str, help='Which monkey?', required=True)
    parser.add_argument('--session', type=int, help='Which session data?', required=True)
    parser.add_argument('--start', type=str, help='Which active start?', required=True)
    parser.add_argument('--short', type=str, default='n', help='Short active?', required=False)
    args = parser.parse_args()

    main(args.monkey, args.session, args.start, args.short)

    # main('Snap',20190829, 'mvt', 'n')
