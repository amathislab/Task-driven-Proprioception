### IMPORTS
import os
import numpy as np
import pandas as pd
import h5py
import fnmatch
import scipy.io as sio
from scipy import stats
from scipy.optimize import least_squares
from collections import OrderedDict, namedtuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score
import sys

sys.path.append('../../code/')
sys.path.append('../code/')
from path_utils import PATH_MONKEY_PROCESSED_DICT

###---- GENERAL ----####

def get_neuraldata_path():
    return PATH_MONKEY_PROCESSED_DICT

def get_figfolder_path():
    return '..'

def get_result_path():
    return '..'


def get_CN_monkeys():
    '''Get list of monkeys with CN data.'''
    return ['Snap', 'Lando', 'Butter', 'NewButter', 'NewLando']

def get_S1_monkeys():
    ''' Get list of monkeys with S1 data.'''
    return ['Han', 'S1Lando', 'Han_20171122', 'Han_20170203', 'Han_20170105', 'Chips']

def has_passive_task(session_date):
    '''Check whether monkey session data has both passive and active or only active tasks.
    session_date - (int) Date YYYYMMDD of session, also serves as identifier of monkey.'''
    active_only_sessions = [20170203, 20170105]
    actpas_sessions = [20171122, 20190829, 20180326, 20170917, 20170913]
    if session_date in active_only_sessions:
        return False
    elif session_date in actpas_sessions:
        return True
    else:
        print('Session date entered is not valid.')

###---- DATA PREPARATION ----###

def from_name_sortfields(monkey_name):
    ''' Function to sort neural data .mat fields (spikes, timestampts, etc.) wrt. monkey name.
    Returns (in order): spike_field, guide_field, ts_field.
    Arguments:
    monkey_name = name of the monkey written in neural dataframe.
    '''
    spike_field, guide_field, ts_field = None, None, None
    if monkey_name in get_S1_monkeys():
        spike_field = 'LeftS1Area2_spikes'
        guide_field = 'LeftS1Area2_unit_guide'
        ts_field = 'LeftS1Area2_ts'
        if monkey_name == 'S1Lando':
            spike_field = 'area2_spikes'
            guide_field = 'area2_unit_guide'
            ts_field = 'area2_ts'

    elif monkey_name in get_CN_monkeys():
        spike_field = 'cuneate_spikes'
        guide_field = 'cuneate_unit_guide'
        ts_field = 'cuneate_ts'
    else:
        print('Not a valid monkey name !')

    return spike_field, guide_field, ts_field

def retrieve_mat_file(path_to_data_folder, monkey, session_date):
    ''' Function to get .mat (10ms bins) file name from monkey name and monkey session.
    Arguments:
    path_to_data_folder - path where neural data files are stored.
    monkey - name of the monkey of interest, case-sensitive!'''

    if not monkey[0].isupper():
        print('Monkey name must contain first upper case !')
        return

    #All neural data files
    neuraldata_files = os.listdir(path_to_data_folder)

    #Selected monkey file
    monkey_file = '{}_CO_{}'.format(monkey, session_date)
    print('Selected monkey file:', monkey_file)

    #Search file
    file_name = None
    for n_file in neuraldata_files:
        if fnmatch.fnmatch(n_file, monkey_file + '*10msBin.mat'):  # this is case sensitive
            file_name = n_file

    if file_name == None:
        print('Could not find a file corresponding to that name! Check monkey name or session id.')
    return file_name


def get_monkey_name(monkey_file):
    ''' Function to get monkey name from a .mat monkey file name.
    Will have upper case first.'''

    monkey_name = monkey_file[:monkey_file.index("_")]
    return monkey_name


def clean_data_dict(data_dict, monkey_name):
    '''A function to clean what can be directly changed when reading the .mat file.
    Called in read_mat_file().
    Arguments:
    data_dict - dict created when reading the .mat file.
    monkey_name - name of the monkey'''
    # Change fields
    data_dict['monkey'] = [monkey_name for i in range(len(data_dict['monkey']))]

    # Reformat - squeeze - fields
    n_trials = len(data_dict['acc'])
    cols_to_reformat = ['bin_size', 'bumpDir', 'idx_bumpTime', 'idx_endTime', 'idx_goCueTime', 'idx_startTime',
                        'idx_tgtOnTime', 'target_direction', 'trial_id', 'trial_start_time']
    for col in cols_to_reformat:
        field_trials = []
        for trial_idx in range(n_trials):
            field_trials.append(data_dict[col][trial_idx][0][0])
        data_dict[col] = field_trials

    print('Cleaned.')
    return data_dict


def read_mat_file(path_to_data, file_name):
    '''Read a .mat file from monkey neural data and return a dict with entries.
    Checks whether:
    	- File is old or new, and this requires different method to open .mat files.
    	- Data sampled with 1ms or 10ms bins, idem.
    Arguments:
    path_to_data - path to folder containing data
    file_name - name of the .mat file, including the extension'''
    data_dict=dict()

    path_to_file = os.path.join(path_to_data, file_name)
    fields_to_ignore = []
    recent_files = ('Han', 'Snap')  # must be tuples
    recent_files_2 = ('NewButter', 'NewLando')
    older_files = ('Lando', 'S1Lando', 'Butter')
    data_size=None
    # Check file recency - determine method used

    # NEW FILES
    if file_name.startswith(recent_files_2):
        mat_dict = mat73.loadmat(path_to_file, use_attrdict=True)
        data_dict = mat_dict['td']

    # NEWER FILES, ANd/OR, 1ms binned
    elif (file_name.startswith(recent_files) and file_name.endswith('TD.mat')):# or file_name.startswith(recent_files_2):
        print('Loading with HDF reader {}...'.format(file_name))
        # Load
        lib = h5py.File(path_to_file, 'r')
        struct_array = lib['td']
        # Init. empty dict
        data_dict = OrderedDict()
        data_size = struct_array['acc'].shape[0]
        print(struct_array.keys())
        for key in struct_array.keys():
            if key not in fields_to_ignore:
                # Init. array
                temp = []
                for row in range(data_size):
                    # Add each row to list
                    temp.append(struct_array[struct_array[key][row].item()][()])
                # Add to dict
                data_dict[key] = temp

        # Close file
        lib.close()
        # Clean data dict - these are messy files
        monkey_name = file_name[:file_name.index('_')]  # get monkey_name
        data_dict = clean_data_dict(data_dict, monkey_name)

    # OLDER FILES, OR, 10ms binned
    elif file_name.startswith(older_files) or file_name.endswith('TD_10msBin.mat'):
        print('Loading with scipy.loadmat {}...'.format(file_name))
        # Load
        mat = sio.loadmat(path_to_file, struct_as_record=True, squeeze_me=True)
        data_array = np.squeeze(mat['td'])
        data_size = data_array.shape[0]
        # Get fields
        n_fields = len(data_array.dtype.descr)
        field_names = [data_array.dtype.descr[f][0] for f in range(n_fields)]
        # Init. empty dict
        data_dict = OrderedDict.fromkeys(field_names)
        for idx, key in enumerate(data_dict.keys()):
            if key not in fields_to_ignore:
                key_list = []
                for trial_idx in range(data_array.shape[0]):
                    key_list.append(data_array[trial_idx][idx])  # add trial to list
                # Fill in dict
                data_dict[key] = key_list

    # Done
    print('File contains {} trials in total.'.format(data_size))
    # Return dict
    return data_dict


def data_dict_to_df(data_dict):
    ''' A function to convert the read .mat file as a data dict into a pandas DataFrame.
    - data_dict - (dict) input data, read with e.g. utils.read_mat_file()  '''
    return pd.DataFrame(data_dict)


def filter_df_columns(data_df, keep_kinematics=True, keep_spike_data=True):
    ''' Remove chunks of data, specified by types of fields
    keep_kinematics - (bool) keeps kinematics data
    keep_spike_data - (bool) keeps neural data.'''

    # Pre-selected fields
    kinematics_fields = ['pos', 'vel', 'acc', 'force', 'opensim', 'opensim_names', 'opensim_mmsec']

    spike_fields = ['cuneate_spikes', 'cuneate_unit_guide', 'cuneate_ts', 'cuneate_naming',
                    'LeftS1Area2_naming', 'LeftS1Area2_spikes', 'LeftS1Area2_ts', 'LeftS1Area2_unit_guide']

    # Drop some columns:
    for col in data_df.columns:
        if keep_kinematics == False:
            if (col in kinematics_fields):# and (col not in remaining_fields):
                data_df.pop(col)
        if keep_spike_data == False:
            if (col in spike_fields):# and (col not in remaining_fields):
                data_df.pop(col)
    return data_df


def remove_unitless_channels(data_df, monkey):
    '''
    Function to remove, from DataFrame data, columns in '<brain_area>_spikes'
    which correspond to recording array channels with no neurons.
    '''

    #monkey_name = data_df['monkey'][0]
    monkey_name = monkey
    print('Monkey name:', monkey_name)
    spike_field, guide_field, ts_field = from_name_sortfields(monkey_name)
    print('Spike field:', spike_field)
    print('Guide field:', guide_field)
    print('Timestamp field:', ts_field)

    # Find indices - all the same for each trials, hence '0' for first trial

    n_neurons = data_df[spike_field][0].shape[1]
    print('Total #recorded sites:', n_neurons)

    if monkey_name in ['NewLando', 'NewButter']:
        indices_to_remove = np.squeeze(np.argwhere(data_df[guide_field][0][:] == 0.0)[:, 1]) # get indices
    else:
        indices_to_remove = np.squeeze(np.argwhere(data_df[guide_field][0][:] == 0.0)[:, 0])
    print('{}/{} columns to remove in {}...'.format(indices_to_remove.shape[0],
                                                    n_neurons,
                                                    spike_field))
    indices_to_keep = np.asarray([x for x in np.arange(n_neurons) if x not in indices_to_remove])

    # Remove channel indices for each trial
    for trial_idx in range(len(data_df)):
        data_df.at[trial_idx, spike_field] = data_df[spike_field][trial_idx][:, indices_to_keep]
    print('Removed!')
    return data_df


def select_spike_window(data_df, window_size_10ms=400, event='idx_movement_on'):
    '''A function to keep spike data within a window after a specified event-
    Arguments:
    data_df - Dataframe with neural data and a "spikes field"
    window_size_10ms - length of window to cut, based on 10ms bins
    event - event from which to keep neural data, align on this event'''

    monkey_name = data_df['monkey'][0]
    spike_field, guide_field, ts_field = from_name_sortfields(monkey_name)

    # Get spike data
    n_valid_trials = len(data_df)
    spike_data = data_df[spike_field]  # [reward_indices]
    n_neurons = data_df[spike_field][0].shape[1]
    half_window_size = int(window_size_10ms / 2)
    # Init. container
    spike_array = np.ndarray((n_valid_trials, window_size_10ms, n_neurons))

    # Select and align
    selected_frame=np.empty([None])
    for idx, trial_idx in enumerate(data_df.index.values):

        # Get trial spikes + event index
        trial_spikes = spike_data[trial_idx]
        event_idx = data_df[event][trial_idx]

        # For Bump indices- sometimes no bump (NaN)
        if np.isnan(event_idx):
            # Take 50 time bins (0.5 sec) before go cue instead
            event_idx = int(data_df['idx_goCueTime'][trial_idx] - 50)
        # Convert to int
        else:
            event_idx = int(event_idx)

        trial_dur = trial_spikes.shape[0]
        # print('Trial dur.', trial_dur)

        # Window around event fits trial duration, select window
        if (event_idx + half_window_size < trial_dur) and (event_idx - half_window_size > 0):
            selected_frame = trial_spikes[event_idx - half_window_size:event_idx + half_window_size, :]

        # Too short before event
        elif event_idx - half_window_size < 0:
            diff_to_pad = -(event_idx - half_window_size)
            selected_frame = trial_spikes[0:event_idx + half_window_size, :]  # select until beginning
            selected_frame = np.pad(selected_frame,
                                    pad_width=((diff_to_pad, 0), (0, 0)),  # pad before, axis 0
                                    constant_values=np.nan,
                                    mode='constant')

        # Too short after event
        elif event_idx + half_window_size > trial_dur:  # if intended window longer than trial end, then pad data
            diff_to_pad = (event_idx + half_window_size) - trial_dur + 1  # +1 to account for excluded index
            selected_frame = trial_spikes[event_idx - half_window_size:-1, :]  # select until end
            selected_frame = np.pad(selected_frame,
                                    pad_width=((0, diff_to_pad), (0, 0)),  # pad after, axis 0
                                    constant_values=np.nan,
                                    mode='constant')

        # Too short before ANd after event
        elif (event_idx - half_window_size < 0) and (event_idx + half_window_size > trial_dur):
            diff_to_pad_before = abs((event_idx - half_window_size))
            diff_to_pad_after = (event_idx + half_window_size) - trial_dur + 1  # +1 to account for excluded index
            selected_frame = trial_spikes[0:-1, :]  # select all
            selected_frame = np.pad(selected_frame,
                                    pad_width=((diff_to_pad_before, diff_to_pad_after), (0, 0)),  # pad after, axis 0
                                    constant_values=np.nan,
                                    mode='constant')

        # Store selected window for that trial
        spike_array[idx, :, :] = selected_frame

    return spike_array


def fano_factor(aligned_spike_array, T=50):
    ''' Compute Fano factors for all neurons (averaged over trials) from spike array aligned
    at a certain event. See corresponding function.
    Arguments:
    aligned_spike_array- spike array to get counts
    T - time window considered to compute counts (10ms bins) '''
    spike_counts = np.sum(aligned_spike_array[0:T, :], axis=0)
    print(spike_counts.shape)
    var = np.var(spike_counts, axis=0)
    mean = np.mean(spike_counts, axis=0)
    factors = var / mean
    return factors, var, mean


def get_reward_indices(data_df):
    ''' Get reward indices from neural data dataframe.
    Arguments:
    data_df - Dataframe with neural data and a "result field"'''
    ids = data_df[(data_df['result'] == 'R') | (data_df['result'] == 82)].index
    return ids


def keep_reward_trials(data_df):
    '''Filter out trials that were not successful.
    Arguments:
    data_df - Neural data dataframe with a "result field".'''

    reward_ids = get_reward_indices(data_df)
    data_df = data_df.loc[reward_ids]
    print('Keeping rewarded trials: {}'.format(len(data_df)))
    return data_df


def load_monkey_data(path_to_data_folder, monkey=None, session_date=None,
                     keep_kinematics=True, keep_spike_data=True, use_new_muscle=True):
    ''' Function to load data: read, makes into dataframe, filter unwanted columns, remove unitless channels
    and keep rewarded trials only.
    Returns a dataframe.
    Arguments:
    path_to_data_folder - path to where neural data files are stored.
    monkey - monkey name string, starting with upper case!
    keep_kinematics - whether to keep kinematic data.
    keep_spike_data - whether to keep spike data.'''

    file_name = retrieve_mat_file(path_to_data_folder, monkey, session_date)
    data_dict = read_mat_file(path_to_data_folder, file_name)
    data_df = data_dict_to_df(data_dict)
    data_df = filter_df_columns(data_df, keep_kinematics=keep_kinematics, keep_spike_data=keep_spike_data)

    if keep_spike_data:
        data_df = remove_unitless_channels(data_df, monkey)
    data_df = keep_reward_trials(data_df)  # remove few non-R trials

    if use_new_muscle:
        print('Using newly computed muscle data...')
        if monkey == 'S1Lando': monkey='Lando'
        if monkey in ['Butter', 'Lando']:
            monkey = 'New'+monkey
        new_muscle_file = os.path.join(get_neuraldata_path(), monkey + '_' + str(session_date) + '_mod.pkl')
        print('Using new muscle file:', new_muscle_file)
        new_muscle_data = pd.read_pickle(new_muscle_file)
        # Ignore non-matching trial ids for Lando
        if monkey == 'NewLando':
            try:  #remove if not already removed in previous function
                new_muscle_data.pop('trial_71')
                new_muscle_data.pop('trial_412')
                new_muscle_data.pop('trial_725')
            except KeyError as err:
                print(err, '- Mismatched trials already removed!from_name_sortfields')


        for t_idx, t_id in enumerate(new_muscle_data.keys()):
            t_duration = new_muscle_data[t_id]['muscle_len'].shape[1]

            #Make new muscle array
            joint_data = data_df.opensim.iloc[t_idx][:,0:14]

            new_t_muscle = np.concatenate((joint_data.T,
                                    new_muscle_data[t_id]['muscle_len']/1000, #because already converted
                                    new_muscle_data[t_id]['muscle_vel']/1000),
                                     axis=0)

            data_df.iat[t_idx, data_df.columns.get_loc('opensim')] = new_t_muscle.T
        print('Opensim muscle data replaced!')

    return data_df

def is_cobump_task(data_df):
    '''Checks if trial dataset has bumpDir i.e. also contains a passive portion.'''
    cols = data_df.columns
    if 'bumpDir' in cols:
        is_cobump = True
        print('Data contains passive and active task.')
    else:
        is_cobump = False
        print('Data only contains active task.')
    return is_cobump

###---- SPIKE DNN DATASET GENERATION ----###

def generate_spike_active_dataset(data_neur_df, monkey_name, active_start, active_length, align, window=5, latency=0):
    '''Generate active dataset of padded spike count from monkey session dataframe.
    Arguments:
    path_to_neural_dfs - (str) path to folder containing pre-saved neural dataframes.
    monkey_name - (str) name of the monkey, starting with upper case.'''

    spike_field, _, _ = from_name_sortfields(monkey_name)

    #Dataset structure
    duration_to_reach = 400
    pad_before = int(align)
    padded_spike_data = []

    #Pad each trial
    for ind_loop, trial_idx in enumerate(data_neur_df.index):

        if active_start == 'cue':
            start_idx = data_neur_df.idx_goCueTime[trial_idx]
        elif active_start == 'mvt':
            start_idx = data_neur_df.idx_movement_on[trial_idx]

        if active_length != 0:
            end_idx = start_idx + active_length
        elif active_length == 0:
            end_idx = data_neur_df.idx_endTime[trial_idx]

        ### Apply latency
        start_idx = start_idx + latency
        end_idx = end_idx + latency

        trial_dur = end_idx - start_idx
        if trial_dur + pad_before >= duration_to_reach:
            pad_diff = 0
        else:
            pad_diff = duration_to_reach - trial_dur - pad_before

        convolved_spike_data = np.apply_along_axis(lambda m: np.convolve(m, np.ones(window) / window, mode='same'),
                                            axis=0, arr=data_neur_df[spike_field][trial_idx])

        if end_idx > convolved_spike_data.shape[0]:
            return

        pad_tmp_array = np.pad(convolved_spike_data[start_idx:end_idx, :],
                                        pad_width=((pad_before, pad_diff), (0, 0)), 
                                        mode='constant', constant_values=0)

        padded_spike_data.append(pad_tmp_array)

    # Make it an array
    padded_spike_data = np.array(padded_spike_data) #np.array(padded_spike_data)
    return padded_spike_data

def generate_spike_passive_dataset(data_neur_df, monkey_name, align, window=5, latency=0):
    '''Generate passive dataset of padded spike count from monkey session dataframe.
    Arguments:
    path_to_neural_dfs - (str) path to folder containing pre-saved neural dataframes.
    monkey_name - (str) name of the monkey, starting with upper case.'''

    spike_field, _, _ = from_name_sortfields(monkey_name)

    #Dataset structure
    duration_to_reach = 400
    pad_before = int(align)
    padded_spike_data = []

    # Pad each trial containing a passive bump
    bump_ids = data_neur_df.index[~np.isnan(data_neur_df['bumpDir'])].tolist()
    for trial_idx in bump_ids:
        bump_idx = int(data_neur_df.idx_bumpTime[trial_idx])
        end_idx = int(bump_idx+13)

        ### Apply latency
        bump_idx = bump_idx + latency
        end_idx = end_idx + latency

        trial_dur = data_neur_df[spike_field][trial_idx][bump_idx:end_idx, :].shape[0]
        if trial_dur + pad_before >= duration_to_reach:
            pad_diff = 0
        else:
            pad_diff = duration_to_reach - trial_dur - pad_before


        convolved_spike_data = np.apply_along_axis(lambda m: np.convolve(m, np.ones(window) / window, mode='same'),
                                            axis=0, arr=data_neur_df[spike_field][trial_idx])

        padded_spike_data.append(np.pad(convolved_spike_data[bump_idx:end_idx, :],
                                        pad_width=((pad_before, pad_diff), (0, 0)), 
                                        mode='constant', constant_values=0))

    # Make it an array
    padded_spike_data = np.array(padded_spike_data)
    return padded_spike_data

def generate_spike_whole_dataset(data_neur_df, monkey_name):
    '''Generate whole-trial dataset of padded spike count from monkey session dataframe.
    Arguments:
    path_to_neural_dfs - (str) path to folder containing pre-saved neural dataframes.
    monkey_name - (str) name of the monkey, starting with upper case.'''

    spike_field, _, _ = from_name_sortfields(monkey_name)

    #Dataset structure
    padded_spike_data = []

    #Take each trial
    for trial_idx in data_neur_df.index:
        end_idx = int(data_neur_df.idx_endTime[trial_idx])
        diff = int(end_idx - data_neur_df.idx_startTime[trial_idx] - 400)

        # Pad if trial duration (start-end) is shorter than 4 sec
        if diff < 0:
            pad_before = np.abs(diff)
            pad_after = 0
            start_idx = data_neur_df.idx_startTime[trial_idx]  # then take from start

            spike_pad = np.pad(data_neur_df[spike_field][trial_idx][start_idx:end_idx, :],
                             pad_width=((pad_before, pad_after), (0, 0)),
                             mode='edge')

        # Else take 4 sec preceding trial end (to include entire reach + bump)
        else:
            start_idx = int(end_idx - 400)
            spike_pad = data_neur_df[spike_field][trial_idx][start_idx:end_idx, :]

        padded_spike_data.append(spike_pad)

    # Make it an array
    padded_spike_data = np.array(padded_spike_data)
    return padded_spike_data


###---- TUNING MODELS DATASET GENERATION ----###

def fit_cosine_curve(rates_tr, rates_te, dir_tr, dir_te):

    # Init. params
    init_b = np.mean(rates_tr, axis=0)
    rms = 3 * np.std(rates_tr) / (2 ** 0.5) / (2 ** 0.5)
    init_z1, init_z2 = rms, rms
    init_params_sin = [init_b, init_z1, init_z2]

    # Fit
    sin_curve_diff = lambda x: x[0] + x[1] * np.sin(dir_tr) + x[2] * np.cos(dir_tr) - rates_tr
    res = least_squares(sin_curve_diff, init_params_sin)
    est_params = res['x']

    # Test predictions
    rates_pred = est_params[0] + est_params[1] * np.sin(dir_te) + est_params[2] * np.cos(dir_te)
    r2_te = r2_score(rates_te, rates_pred)
    n,p=len(dir_te),3
    r2_te_adj = 1-(1-r2_te)*(n-1)/(n-p-1)
    ev_test_score = explained_variance_score(rates_te, rates_pred)
    # PD
    PD = np.arctan2(est_params[1], est_params[2]) % (2 * np.pi)  # in [0, 2pi]

    return ev_test_score, r2_te_adj, PD

def window_conv(x):
    d=[]
    for i in range(x.shape[1]):
        d.append(np.convolve(x[:,i], np.ones(5)/5, mode='same'))
    return np.asarray(d).T

def collect_tuning_data_latency(data_df, spike_field, active_start, is_short, latency=0):
    '''Collect kinematic and spike data for tuning models from monkey dataframe.
    Arguments:
    monkey_name - (str) Monkey name
    session_date - (int) Dataset session.
    active_start - (str) Which active start event to take data from.'''


    print('Active start', active_start, 'short predictions', is_short)

    # LOAD DATA

    bump_ids = data_df[~np.isnan(data_df.bumpDir)].index

    # COLLECT DATA
    # Mean firing rates
    rates_pas = []
    rates_act_cue_end = []
    rates_act_mvt_end = []
    rates_act_peak = []
    # Instant rates
    rates_t_pas = []
    rates_t_act_cue_end = []
    rates_t_act_mvt_end = []
    rates_t_act_peak = []

    # Mvt features
    dir_pas_trial = []
    dir_pas = []
    pos_pas = []
    pos_pas_norm = []
    vel_pas = []
    vel_pas_norm = []
    acc_pas = []
    acc_pas_norm = []
    force_pas = []
    force_pas_norm = []

    dir_act_cue_end_trial = []
    dir_act_mvt_end_trial = []
    dir_act_peak_trial = []
    dir_act_cue_end = []
    dir_act_mvt_end = []
    dir_act_peak = []
    pos_act = []
    pos_act_norm = []
    vel_act = []
    vel_act_norm = []
    acc_act = []
    acc_act_norm = []
    force_act = []
    force_act_norm = []

    #Joints
    joint_ang_ids = np.arange(0,7)
    j_ang_act_mvt_end = []
    joint_vel_ids = np.arange(7, 14)
    j_vel_act_mvt_end = []


    #Muscles
    muscle_len_ids = np.arange(14, 53) #as in opensim field
    m_len_pas = []
    m_vel_pas = []
    m_len_act_cue_end = []
    m_vel_act_cue_end = []
    m_len_act_mvt_end = []
    m_vel_act_mvt_end = []
    m_len_act_peak = []
    m_vel_act_peak = []

    latency = int(latency)

    for t_id in data_df.index:
        cue_time = int(data_df.idx_goCueTime[t_id])
        peak_time = int(data_df.idx_peak_speed[t_id])
        mvt_time = int(data_df.idx_movement_on[t_id])
        end_time = int(data_df.idx_endTime[t_id])
        if active_start == 'mvt':
            start_time = mvt_time
        elif active_start == 'cue':
            start_time = cue_time

        if is_short == True:
            end_time = int(start_time)+13


        #Passive-including trials
        if t_id in bump_ids:
            bump_time = int(data_df.idx_bumpTime[t_id])

            #Mean rates
            rates_pas.append(np.mean(data_df[spike_field][t_id][bump_time + latency:bump_time + latency + 13, :], axis=0))
            #Mean directions
            vect = data_df.pos[t_id][bump_time + 13, :] - data_df.pos[t_id][bump_time, :]
            dir_from_vect = np.arctan2(vect[1], vect[0]) #+ np.pi  # first y-coord then x-coord!
            dir_pas_trial.append(dir_from_vect)

            #Instant - convolved in 50ms
            rates_t_pas.append(window_conv(data_df[spike_field][t_id])[bump_time+ latency:bump_time+ latency + 13, :])
            #Mvt instant directions
            x_vect = data_df.pos[t_id][bump_time:bump_time + 13, 0] - data_df.pos[t_id][bump_time-1, 0]
            y_vect = data_df.pos[t_id][bump_time:bump_time + 13, 1] - data_df.pos[t_id][bump_time-1, 1]
            dir_t_from_vect = np.arctan2(y_vect, x_vect)
            dir_pas.append(dir_t_from_vect)

            #Mvt features
            pos_vect = data_df.pos[t_id][bump_time:bump_time + 13] - data_df.pos[t_id][bump_time - 1]
            pos_pas.append(pos_vect)
            pos_pas_norm.append(np.linalg.norm(pos_vect, axis=1))
            vel_vect = data_df.vel[t_id][bump_time:bump_time + 13]
            vel_pas.append(vel_vect)
            vel_pas_norm.append(np.linalg.norm(vel_vect, axis=1)) #or speed field
            acc_vect = data_df.acc[t_id][bump_time:bump_time + 13]
            acc_pas.append(acc_vect)
            acc_pas_norm.append(np.linalg.norm(acc_vect, axis=1))
            f_vect = data_df.force[t_id][bump_time:bump_time + 13]
            force_pas.append(f_vect)
            force_pas_norm.append(np.linalg.norm(f_vect, axis=1))

            #Muscle trajectories
            m_len_trial = data_df.opensim[t_id][bump_time:bump_time + 13, muscle_len_ids]
            m_len_pas.append(m_len_trial)
            m_vel_pas.append(np.gradient(m_len_trial, 0.01, axis=0))

        #All trials - Active
        #Mean rates
        rates_act_cue_end.append(np.mean(data_df[spike_field][t_id][cue_time:end_time, :], axis=0))
        rates_act_peak.append(np.mean(data_df[spike_field][t_id][peak_time-25:peak_time+25, :], axis=0))
        rates_act_mvt_end.append(np.mean(data_df[spike_field][t_id][mvt_time:end_time, :], axis=0))
        #Mean directions
        vect = data_df.pos[t_id][end_time, :] - data_df.pos[t_id][cue_time, :]
        dir_from_vect = np.arctan2(vect[1], vect[0]) + np.pi
        dir_act_cue_end_trial.append(dir_from_vect)

        vect = data_df.pos[t_id][end_time, :] - data_df.pos[t_id][mvt_time, :]
        dir_from_vect = np.arctan2(vect[1], vect[0]) + np.pi
        dir_act_mvt_end_trial.append(dir_from_vect)

        vect = data_df.pos[t_id][peak_time+25, :] - data_df.pos[t_id][peak_time-25, :]
        dir_from_vect = np.arctan2(vect[1], vect[0]) + np.pi
        dir_act_peak_trial.append(dir_from_vect)

        #Instant rates - convolved over 50ms windows
        rates_t_act_cue_end.append(window_conv(data_df[spike_field][t_id])[cue_time+ latency:end_time+ latency, :])
        rates_t_act_peak.append(window_conv(data_df[spike_field][t_id])[peak_time-25+ latency:peak_time+25+ latency, :])
        rates_t_act_mvt_end.append(window_conv(data_df[spike_field][t_id])[mvt_time+ latency:end_time+ latency, :])

        # Mvt instant directions
        x_vect = data_df.pos[t_id][cue_time:end_time, 0] - data_df.pos[t_id][cue_time - 1, 0]
        y_vect = data_df.pos[t_id][cue_time:end_time, 1] - data_df.pos[t_id][cue_time - 1, 1]
        dir_t_from_vect = np.arctan2(y_vect, x_vect)
        dir_act_cue_end.append(dir_t_from_vect)

        x_vect = data_df.pos[t_id][mvt_time:end_time, 0] - data_df.pos[t_id][mvt_time - 1, 0]
        y_vect = data_df.pos[t_id][mvt_time:end_time, 1] - data_df.pos[t_id][mvt_time - 1, 1]
        dir_t_from_vect = np.arctan2(y_vect, x_vect)
        dir_act_mvt_end.append(dir_t_from_vect)

        x_vect = data_df.pos[t_id][peak_time-25:peak_time+25, 0] - data_df.pos[t_id][peak_time-25-1, 0]
        y_vect = data_df.pos[t_id][peak_time-25:peak_time+25, 1] - data_df.pos[t_id][peak_time-25-1, 1]
        dir_t_from_vect = np.arctan2(y_vect, x_vect)
        dir_act_peak.append(dir_t_from_vect)


        #Mvt features: all cue/mvt-end
        pos_vect = data_df.pos[t_id][start_time:end_time]
        pos_act.append(pos_vect)
        pos_act_norm.append(np.linalg.norm(pos_vect, axis=1))
        vel_vect = data_df.vel[t_id][start_time:end_time]
        vel_act.append(vel_vect)
        vel_act_norm.append(np.linalg.norm(vel_vect, axis=1))  # or speed field
        acc_vect = data_df.acc[t_id][start_time:end_time]
        acc_act.append(acc_vect)
        acc_act_norm.append(np.linalg.norm(acc_vect, axis=1))
        f_vect = data_df.force[t_id][start_time:end_time]
        force_act.append(f_vect)
        force_act_norm.append(np.linalg.norm(f_vect, axis=1))

        #Joint trajectories
        j_ang_trial = data_df.opensim[t_id][mvt_time:end_time, joint_ang_ids]
        j_ang_act_mvt_end.append(j_ang_trial)
        j_vel_trial = data_df.opensim[t_id][mvt_time:end_time, joint_vel_ids]
        #j_vel_act_mvt_end.append(np.gradient(j_ang_trial, 0.01, axis=0))
        j_vel_act_mvt_end.append(j_vel_trial)

        # Muscle trajectories
        m_len_trial = data_df.opensim[t_id][cue_time:end_time, muscle_len_ids]
        m_len_act_cue_end.append(m_len_trial)
        m_vel_act_cue_end.append(np.gradient(m_len_trial, 0.01, axis=0))

        m_len_trial = data_df.opensim[t_id][mvt_time:end_time, muscle_len_ids]
        m_len_act_mvt_end.append(m_len_trial)
        m_vel_act_mvt_end.append(np.gradient(m_len_trial, 0.01, axis=0))

        m_len_trial = data_df.opensim[t_id][peak_time-25:peak_time+25, muscle_len_ids]
        m_len_act_peak.append(m_len_trial)
        m_vel_act_peak.append(np.gradient(m_len_trial, 0.01, axis=0))

    print('Trial kinematic data collected.')

    # Make as arrays time-averaged variables
    rates_pas = np.asarray(rates_pas)
    dir_pas = np.asarray(dir_pas)
    rates_act_cue_end = np.asarray(rates_act_cue_end)
    rates_act_mvt_end = np.asarray(rates_act_mvt_end)
    rates_act_peak = np.asarray(rates_act_peak)
    dir_act_cue_end = np.asarray(dir_act_cue_end)
    dir_act_mvt_end = np.asarray(dir_act_mvt_end)
    dir_act_peak = np.asarray(dir_act_peak)

    Sets = namedtuple('Sets',
                         ['rates_trial', 'dir_trial', # Mean
                          'rates_t', 'dir', 'pos', 'pos_norm', 'vel', 'vel_norm', 'acc', 'acc_norm', 'force', 'force_norm', #Instantaneous
                          'j_ang', 'j_vel',
                          'm_len', 'm_vel']
                         )

    passive_sets = Sets(rates_pas, dir_pas_trial,#Mean
                        rates_t_pas, dir_pas, pos_pas, pos_pas_norm, vel_pas, vel_pas_norm, acc_pas, acc_pas_norm, force_pas, force_pas_norm, #Instantaneous
                        [],[],
                        m_len_pas, m_vel_pas)

    #From cue time
    if active_start == 'cue':
        active_sets = Sets(rates_act_cue_end, dir_act_cue_end_trial,
                           rates_t_act_cue_end, dir_act_cue_end, pos_act, pos_act_norm, vel_act, vel_act_norm, acc_act, acc_act_norm, force_act, force_act_norm,
                           m_len_act_cue_end, m_vel_act_cue_end)
    #From movement onset
    elif active_start == 'mvt':
        active_sets = Sets(rates_act_mvt_end, dir_act_mvt_end_trial,
                           rates_t_act_mvt_end, dir_act_mvt_end, pos_act, pos_act_norm, vel_act, vel_act_norm, acc_act, acc_act_norm, force_act, force_act_norm,
                           j_ang_act_mvt_end, j_vel_act_mvt_end,
                           m_len_act_mvt_end, m_vel_act_mvt_end)
    else:
        print('The specified active start event is not valid!')

    Datasets = namedtuple('Datasets', ['PassiveSets', 'ActiveSets'])
    session_sets = Datasets(passive_sets, active_sets)

    return session_sets


def collect_tuning_data(data_df, spike_field, active_start, is_short):
    '''Collect kinematic and spike data for tuning models from monkey dataframe.
    Arguments:
    monkey_name - (str) Monkey name
    session_date - (int) Dataset session.
    active_start - (str) Which active start event to take data from.'''


    print('Active start', active_start, 'short predictions', is_short)

    # LOAD DATA

    bump_ids = data_df[~np.isnan(data_df.bumpDir)].index

    # COLLECT DATA
    # Mean firing rates
    rates_pas = []
    rates_act_cue_end = []
    rates_act_mvt_end = []
    rates_act_peak = []
    # Instant rates
    rates_t_pas = []
    rates_t_act_cue_end = []
    rates_t_act_mvt_end = []
    rates_t_act_peak = []

    # Mvt features
    dir_pas_trial = []
    dir_pas = []
    pos_pas = []
    pos_pas_norm = []
    vel_pas = []
    vel_pas_norm = []
    acc_pas = []
    acc_pas_norm = []
    force_pas = []
    force_pas_norm = []

    dir_act_cue_end_trial = []
    dir_act_mvt_end_trial = []
    dir_act_peak_trial = []
    dir_act_cue_end = []
    dir_act_mvt_end = []
    dir_act_peak = []
    pos_act = []
    pos_act_norm = []
    vel_act = []
    vel_act_norm = []
    acc_act = []
    acc_act_norm = []
    force_act = []
    force_act_norm = []

    #Joints
    joint_ang_ids = np.arange(0,7)
    j_ang_act_mvt_end = []
    joint_vel_ids = np.arange(7, 14)
    j_vel_act_mvt_end = []


    #Muscles
    muscle_len_ids = np.arange(14, 53) #as in opensim field
    m_len_pas = []
    m_vel_pas = []
    m_len_act_cue_end = []
    m_vel_act_cue_end = []
    m_len_act_mvt_end = []
    m_vel_act_mvt_end = []
    m_len_act_peak = []
    m_vel_act_peak = []

    for t_id in data_df.index:
        cue_time = int(data_df.idx_goCueTime[t_id])
        peak_time = int(data_df.idx_peak_speed[t_id])
        mvt_time = int(data_df.idx_movement_on[t_id])
        end_time = int(data_df.idx_endTime[t_id])
        if active_start == 'mvt':
            start_time = mvt_time
        elif active_start == 'cue':
            start_time = cue_time

        if is_short == True:
            end_time = int(start_time)+13


        #Passive-including trials
        if t_id in bump_ids:
            bump_time = int(data_df.idx_bumpTime[t_id])

            #Mean rates
            rates_pas.append(np.mean(data_df[spike_field][t_id][bump_time:bump_time + 13, :], axis=0))
            #Mean directions
            vect = data_df.pos[t_id][bump_time + 13, :] - data_df.pos[t_id][bump_time, :]
            dir_from_vect = np.arctan2(vect[1], vect[0]) #+ np.pi  # first y-coord then x-coord!
            dir_pas_trial.append(dir_from_vect)

            #Instant - convolved in 50ms
            rates_t_pas.append(window_conv(data_df[spike_field][t_id])[bump_time:bump_time + 13, :])
            #Mvt instant directions
            x_vect = data_df.pos[t_id][bump_time:bump_time + 13, 0] - data_df.pos[t_id][bump_time-1, 0]
            y_vect = data_df.pos[t_id][bump_time:bump_time + 13, 1] - data_df.pos[t_id][bump_time-1, 1]
            dir_t_from_vect = np.arctan2(y_vect, x_vect)
            dir_pas.append(dir_t_from_vect)

            #Mvt features
            pos_vect = data_df.pos[t_id][bump_time:bump_time + 13] - data_df.pos[t_id][bump_time - 1]
            pos_pas.append(pos_vect)
            pos_pas_norm.append(np.linalg.norm(pos_vect, axis=1))
            vel_vect = data_df.vel[t_id][bump_time:bump_time + 13]
            vel_pas.append(vel_vect)
            vel_pas_norm.append(np.linalg.norm(vel_vect, axis=1)) #or speed field
            acc_vect = data_df.acc[t_id][bump_time:bump_time + 13]
            acc_pas.append(acc_vect)
            acc_pas_norm.append(np.linalg.norm(acc_vect, axis=1))
            f_vect = data_df.force[t_id][bump_time:bump_time + 13]
            force_pas.append(f_vect)
            force_pas_norm.append(np.linalg.norm(f_vect, axis=1))

            #Muscle trajectories
            m_len_trial = data_df.opensim[t_id][bump_time:bump_time + 13, muscle_len_ids]
            m_len_pas.append(m_len_trial)
            m_vel_pas.append(np.gradient(m_len_trial, 0.01, axis=0))

        #All trials - Active
        #Mean rates
        rates_act_cue_end.append(np.mean(data_df[spike_field][t_id][cue_time:end_time, :], axis=0))
        rates_act_peak.append(np.mean(data_df[spike_field][t_id][peak_time-25:peak_time+25, :], axis=0))
        rates_act_mvt_end.append(np.mean(data_df[spike_field][t_id][mvt_time:end_time, :], axis=0))
        #Mean directions
        vect = data_df.pos[t_id][end_time, :] - data_df.pos[t_id][cue_time, :]
        dir_from_vect = np.arctan2(vect[1], vect[0]) + np.pi
        dir_act_cue_end_trial.append(dir_from_vect)

        vect = data_df.pos[t_id][end_time, :] - data_df.pos[t_id][mvt_time, :]
        dir_from_vect = np.arctan2(vect[1], vect[0]) + np.pi
        dir_act_mvt_end_trial.append(dir_from_vect)

        vect = data_df.pos[t_id][peak_time+25, :] - data_df.pos[t_id][peak_time-25, :]
        dir_from_vect = np.arctan2(vect[1], vect[0]) + np.pi
        dir_act_peak_trial.append(dir_from_vect)

        #Instant rates - convolved over 50ms windows
        rates_t_act_cue_end.append(window_conv(data_df[spike_field][t_id])[cue_time:end_time, :])
        rates_t_act_peak.append(window_conv(data_df[spike_field][t_id])[peak_time-25:peak_time+25, :])
        rates_t_act_mvt_end.append(window_conv(data_df[spike_field][t_id])[mvt_time:end_time, :])

        # Mvt instant directions
        x_vect = data_df.pos[t_id][cue_time:end_time, 0] - data_df.pos[t_id][cue_time - 1, 0]
        y_vect = data_df.pos[t_id][cue_time:end_time, 1] - data_df.pos[t_id][cue_time - 1, 1]
        dir_t_from_vect = np.arctan2(y_vect, x_vect)
        dir_act_cue_end.append(dir_t_from_vect)

        x_vect = data_df.pos[t_id][mvt_time:end_time, 0] - data_df.pos[t_id][mvt_time - 1, 0]
        y_vect = data_df.pos[t_id][mvt_time:end_time, 1] - data_df.pos[t_id][mvt_time - 1, 1]
        dir_t_from_vect = np.arctan2(y_vect, x_vect)
        dir_act_mvt_end.append(dir_t_from_vect)

        x_vect = data_df.pos[t_id][peak_time-25:peak_time+25, 0] - data_df.pos[t_id][peak_time-25-1, 0]
        y_vect = data_df.pos[t_id][peak_time-25:peak_time+25, 1] - data_df.pos[t_id][peak_time-25-1, 1]
        dir_t_from_vect = np.arctan2(y_vect, x_vect)
        dir_act_peak.append(dir_t_from_vect)


        #Mvt features: all cue/mvt-end
        pos_vect = data_df.pos[t_id][start_time:end_time]
        pos_act.append(pos_vect)
        pos_act_norm.append(np.linalg.norm(pos_vect, axis=1))
        vel_vect = data_df.vel[t_id][start_time:end_time]
        vel_act.append(vel_vect)
        vel_act_norm.append(np.linalg.norm(vel_vect, axis=1))  # or speed field
        acc_vect = data_df.acc[t_id][start_time:end_time]
        acc_act.append(acc_vect)
        acc_act_norm.append(np.linalg.norm(acc_vect, axis=1))
        f_vect = data_df.force[t_id][start_time:end_time]
        force_act.append(f_vect)
        force_act_norm.append(np.linalg.norm(f_vect, axis=1))

        #Joint trajectories
        j_ang_trial = data_df.opensim[t_id][mvt_time:end_time, joint_ang_ids]
        j_ang_act_mvt_end.append(j_ang_trial)
        j_vel_trial = data_df.opensim[t_id][mvt_time:end_time, joint_vel_ids]
        #j_vel_act_mvt_end.append(np.gradient(j_ang_trial, 0.01, axis=0))
        j_vel_act_mvt_end.append(j_vel_trial)

        # Muscle trajectories
        m_len_trial = data_df.opensim[t_id][cue_time:end_time, muscle_len_ids]
        m_len_act_cue_end.append(m_len_trial)
        m_vel_act_cue_end.append(np.gradient(m_len_trial, 0.01, axis=0))

        m_len_trial = data_df.opensim[t_id][mvt_time:end_time, muscle_len_ids]
        m_len_act_mvt_end.append(m_len_trial)
        m_vel_act_mvt_end.append(np.gradient(m_len_trial, 0.01, axis=0))

        m_len_trial = data_df.opensim[t_id][peak_time-25:peak_time+25, muscle_len_ids]
        m_len_act_peak.append(m_len_trial)
        m_vel_act_peak.append(np.gradient(m_len_trial, 0.01, axis=0))

    print('Trial kinematic data collected.')

    # Make as arrays time-averaged variables
    rates_pas = np.asarray(rates_pas)
    dir_pas = np.asarray(dir_pas)
    rates_act_cue_end = np.asarray(rates_act_cue_end)
    rates_act_mvt_end = np.asarray(rates_act_mvt_end)
    rates_act_peak = np.asarray(rates_act_peak)
    dir_act_cue_end = np.asarray(dir_act_cue_end)
    dir_act_mvt_end = np.asarray(dir_act_mvt_end)
    dir_act_peak = np.asarray(dir_act_peak)

    Sets = namedtuple('Sets',
                         ['rates_trial', 'dir_trial', # Mean
                          'rates_t', 'dir', 'pos', 'pos_norm', 'vel', 'vel_norm', 'acc', 'acc_norm', 'force', 'force_norm', #Instantaneous
                          'j_ang', 'j_vel',
                          'm_len', 'm_vel']
                         )

    passive_sets = Sets(rates_pas, dir_pas_trial,#Mean
                        rates_t_pas, dir_pas, pos_pas, pos_pas_norm, vel_pas, vel_pas_norm, acc_pas, acc_pas_norm, force_pas, force_pas_norm, #Instantaneous
                        [],[],
                        m_len_pas, m_vel_pas)

    #From cue time
    if active_start == 'cue':
        active_sets = Sets(rates_act_cue_end, dir_act_cue_end_trial,
                           rates_t_act_cue_end, dir_act_cue_end, pos_act, pos_act_norm, vel_act, vel_act_norm, acc_act, acc_act_norm, force_act, force_act_norm,
                           m_len_act_cue_end, m_vel_act_cue_end)
    #From movement onset
    elif active_start == 'mvt':
        active_sets = Sets(rates_act_mvt_end, dir_act_mvt_end_trial,
                           rates_t_act_mvt_end, dir_act_mvt_end, pos_act, pos_act_norm, vel_act, vel_act_norm, acc_act, acc_act_norm, force_act, force_act_norm,
                           j_ang_act_mvt_end, j_vel_act_mvt_end,
                           m_len_act_mvt_end, m_vel_act_mvt_end)
    else:
        print('The specified active start event is not valid!')

    Datasets = namedtuple('Datasets', ['PassiveSets', 'ActiveSets'])
    session_sets = Datasets(passive_sets, active_sets)

    return session_sets


###---- NEURAL RELIABILITY TOOLS ----###

def spearmanbrown_correct(rho):
    'Spearman-Brown-corrected trial split correlation score.'
    return (rho * 2) / (1 + rho)