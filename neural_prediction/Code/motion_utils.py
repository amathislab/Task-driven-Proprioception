### IMPORTS
import numpy as np
import pickle
import matplotlib.pyplot as plt
import h5py

###---- GENERAL ----####

def get_motiondata_path():
    return None

def get_DDdata_path():
    return None

def get_neuraldata_path():
    return None

def get_figfolder_path():
    return None

### --- DATA PREPARATION ----####

def get_DD_muscleorder():
    '''Function to get the order of muscle IDs as in DeepDraw pipeline. 
       See Saul et al., 2014 for muscle code references.'''
    
    deepdraw_order = ['CORB', 'DELT1', 'DELT2', 'DELT3', 'INFSP', 'LAT1', 'LAT2', 'LAT3', 'PECM1',
        'PECM2', 'PECM3', 'SUBSC', 'SUPSP', 'TMAJ', 'TMIN', 'ANC', 'BIClong', 'BICshort',
        'BRA', 'BRD', 'ECRL', 'PT', 'TRIlat', 'TRIlong', 'TRImed']
    return deepdraw_order

def get_mapping_len_ids():
    ''' Function to get muscle LENGTH indices to match muscle orders between DeepDraw and opensim field in .mat files.'''
    return [20, 21, 22, 23, 36, 37, 38, 39, 41, 42, 42, 45, 47, 48, 49, 15, 16, 17, 18, 19, 25, 44, 50, 51, 52]

def get_mapping_vel_ids():
    ''' Function to get muscle VELOCITY indices to match muscle orders between DeepDraw and opensim field in .mat files.'''
    return [59, 60, 61, 62, 75, 76, 77, 78, 80, 81, 81, 84, 86, 87, 88, 54, 55, 56, 57, 58, 64, 83, 89, 90, 91]


def scale_trial_vel(trial_vel, ECDFs_m, human_vel_muscle):
    ''' Function to compute, for one trial, scaled monkey velocity based on human statistics.
    Arguments:
    trial_vel - (tx25 array) velocity of one trial from monkey dataset (typicallly data_df) for each muscle
    ECDFs_m - (dict) dictionary containing pre-computed ECDF for each monkey muscle
    human_vel - (Tx25 array) human velocity data set for each muscle'''
    
    scaled_vel = []
    for m_idx, m_code in enumerate(get_DD_muscleorder()):
        #Compute probabilities P(X<=x)
        probs = ECDFs_m[m_code](trial_vel[:,m_idx]) #and convert in mm/sec like human data
        #Sort human velocity i.e. get quantile function of human vel. distribution
        human_vels = human_vel_muscle[:,m_idx]
        sort = sorted(human_vels)
        #Compute values x from quantile function corresponding to Ps, for each data point
        s_vel = []      
        for i in range(len(trial_vel)):
            try:  
                if probs[i] == 1.0:
                    s_vel.append(sort[int(len(human_vels) * probs[i])-1]) #take the one just before
                else:
                    s_vel.append(sort[int(len(human_vels) * probs[i])])
            except IndexError:
                print(probs[i])
                print('Muscle', m_code, 'point', i, len(sort), int(len(human_vels)*probs[i]))
        
        scaled_vel.append(s_vel)
    return np.asarray(scaled_vel).T

def scale_velocity(monkey_data_df):
    '''Function to scale muscle velocity for all trials of a monkey data set (DataFrame).
    Arguments:
    monkey_data_df - monkey DataFrame from ut.read_mat_file() process.'''
    
    #Load human velocity data (subset from test, -15° horizontal plane)
    human_vel_data = np.load(get_motiondata_path()+'sample_human_muscVel_quantiles.npy').T
    #Load ECDFs for monkey
    ECDFs_m = np.load(get_motiondata_path() + 'Snap_monkey_ECDF.pkl', allow_pickle=True)

    all_scaled_trials = []
    for trial_idx in monkey_data_df.index:
        if trial_idx % 100 == 0:
            print('Scaling trial {}'.format(trial_idx))
        #Scale
        scaled_trial = scale_trial_vel(monkey_data_df['opensim_mmsec'][trial_idx][:, get_mapping_vel_ids()],
                                       ECDFs_m,
                                       human_vel_data)
        all_scaled_trials.append(scaled_trial)
    monkey_data_df['scaled_vels'] = all_scaled_trials
    print('Muscle stretch velocity scaled! Added field "scaled_vels" in dataframe.')
    return monkey_data_df

def scale_trial_len(trial_len, ECDFs_m, human_len_muscle):
    ''' Function to compute, for one trial, scaled monkey velocity based on human statistics.
    Arguments:
    trial_len - (tx25 array) length of one trial from monkey dataset (typicallly data_df) for each muscle
    ECDFs_m - (dict) dictionary containing pre-computed ECDF for each monkey muscle
    human_len - (Tx25 array) human length data set for each muscle'''
    
    scaled_len = []
    for m_idx, m_code in enumerate(get_DD_muscleorder()):
        #Compute probabilities P(X<=x)
        probs = ECDFs_m[m_code](trial_len[:,m_idx]) #and convert in mm like human data
        #Sort human length i.e. get quantile function of human len. distribution
        human_lens = human_len_muscle[:,m_idx]
        sort = sorted(human_lens)
        #Compute values x from quantile function corresponding to Ps, for each data point
        s_len = []      
        for i in range(len(trial_len)):
            try:  
                if probs[i] == 1.0:
                    s_len.append(sort[int(len(human_lens) * probs[i])-1]) #take the one just before
                else:
                    s_len.append(sort[int(len(human_lens) * probs[i])])
            except IndexError:
                print(probs[i])
                print('Muscle', m_code, 'point', i, len(sort), int(len(human_lens)*probs[i]))
        
        scaled_len.append(s_len)
    return np.asarray(scaled_len)

def scale_length(monkey_data_df):
    '''Function to scale muscle velocity for all trials of a monkey data set (DataFrame).
    Arguments:
    monkey_data_df - monkey DataFrame from ut.read_mat_file() process.'''
    
    #Load human length quantiles data (subset from test, -15° horizontal plane)
    human_len_data = np.load(get_motiondata_path()+'sample_human_muscLen_quantiles.npy').T
    
    #Load ECDFs for monkey
    ECDFs_m = np.load(get_motiondata_path() + 'Snap_monkey_ECDF_len.pkl', allow_pickle=True)

    all_scaled_trials = []
    for trial_idx in monkey_data_df.index:
        if trial_idx % 100 == 0:
            print('Scaling trial {}'.format(trial_idx))
            
        scaled_trial = scale_trial_len(monkey_data_df['opensim_mmsec'][trial_idx][:, get_mapping_len_ids()],
                                       ECDFs_m,
                                      human_len_data)
        all_scaled_trials.append(scaled_trial.T)
    monkey_data_df['scaled_lens'] = all_scaled_trials
    print('Muscle stretch length scaled! Added field "scaled_lens" in dataframe.')
    return monkey_data_df


def preprocess_monkey_data(data_df, scale_vel = True, scale_len = True):
    '''Function that gathers preprocessing steps on monkey data.
    Arguments:
    data_df - (pd.DataFrame)mMonkey data DataFrame.
    scale_vel - (bool) whether to scale muscle velocity field
    scale_len - (bool) whether to scale msucle length field'''
    
    print('Processing monkey kinematic data:')
    #Add opensim column but in mm/sec - this must happen before scaling
    data_df['opensim_mmsec'] = data_df['opensim'] * 1000
    
    #Scale velocity to human range
    if scale_vel:    
        print('Scaling muscle velocity...')
        data_df = scale_velocity(data_df)
    if scale_len:
        print('Scaling muscle length...')
        data_df = scale_length(data_df)
    
    return data_df


def load_kin_df(path_to_datafolder, monkey):
    ''' Load a pickle file for a monkey dataframe containing scaled velocities and lengths (kin) 
    saved in MotionData.
    Arguments:
    path_to_datafolder - (str) path to folder containing .pkl file.
    monkey - (str) name of the monkey, starting with uppercase!'''
    
    print('Loading dataframe for {}...'.format(monkey))
    file_name = path_to_datafolder + monkey + '_kin_df.pkl'
    with open(file_name, 'rb') as pickle_file:
        data_kin_df = pickle.load(pickle_file)
    
    return data_kin_df

def signpow(a,b): return np.sign(a)*(np.abs(a)**b)
def make_spindle_coords(muscle_vel):
    ''' Make muscle spindle instantaneous firing rates from muscle velocities.
    Model from ...
    Arguments:
    muscle_vel - (array) muscle velocity array.'''
    p_rate = 82 + 4.3*signpow(muscle_vel, 0.6)
    return p_rate

def add_noise(mconf, factor):
    '''Add factor of Gaussian noise proportional to firing rate.
    Arguments:
    mconf - (array) muscle configuration i.e. muscle array of data.
    factor - (float) factor of noise to add.'''
    
    noisy_mconf = mconf + factor*mconf.std(axis=1)[:, None]*np.random.randn(*mconf.shape)
    return noisy_mconf

def generate_kin_active_dataset(kin_df, active_start, active_length, align):
    '''Generate dataset of padded muscle active muscle kinematic input from monkey session dataframe.
    Arguments:
    path_to_kin_df - (str) path to folder containing pre-saved kin dataframes.
    monkey_name - (str) name of the monkey, starting with upper case.
    session_date - (int) date of session (YYYYMMDD).'''

    #Dataset structure
    duration_to_reach = 400
    pad_before = int(align) #all trials are aligned at time point 100
    padded_len_data = []
    padded_vel_data = []
    padded_endeff_pos = []
    padded_endeff_vel = []
    padded_joint_len_data = []
    padded_joint_vel_data = []
    # Indices matching Chan and Moran model: 39
    muscle_len_ids = np.arange(14, 53) #Muscle arrangement in NN #Indices matching Chan and Moran model: 39
    # muscle_vel_ids = np.arange(53, 92)
    joint_len_ids = np.arange(0, 4)


    #Pad each trial
    for trial_idx in kin_df.index:
        if active_start == 'cue':
            start_idx = kin_df.idx_goCueTime[trial_idx]
        elif active_start == 'mvt':
            start_idx =  kin_df.idx_movement_on[trial_idx]
        if active_length != 0:
            end_idx = start_idx + int(active_length)
        elif active_length == 0:
            end_idx = kin_df.idx_endTime[trial_idx]
        trial_dur = end_idx - start_idx

        if trial_dur + pad_before >= duration_to_reach:
            pad_diff = 0
        else:
            pad_diff = duration_to_reach - trial_dur - pad_before

        len_pad = np.pad(1000 * kin_df['opensim'][trial_idx][start_idx:end_idx, muscle_len_ids],
                                      pad_width=((pad_before, pad_diff), (0, 0)),
                                      mode='edge')

        padded_len_data.append(len_pad)
        padded_vel_data.append(np.gradient(len_pad, 0.01, axis=0))

        joint_vel_data = np.gradient(kin_df['opensim'][trial_idx][start_idx:end_idx, joint_len_ids], 0.01, axis=0)
        joint_len_data = np.pad(kin_df['opensim'][trial_idx][start_idx:end_idx, joint_len_ids],
                                      pad_width=((pad_before, pad_diff), (0, 0)),
                                      mode='edge')
        joint_vel_data = np.pad(joint_vel_data,
                                      pad_width=((pad_before, pad_diff), (0, 0)),
                                      mode='edge')

        padded_joint_len_data.append(joint_len_data)
        padded_joint_vel_data.append(joint_vel_data)

        padded_endeff_pos.append(np.pad(kin_df['pos'][trial_idx][start_idx:end_idx, :],
                                        pad_width=((pad_before, pad_diff), (0, 0)),
                                        mode='edge'))
        padded_endeff_vel.append(np.pad(kin_df['vel'][trial_idx][start_idx:end_idx, :],
                                        pad_width=((pad_before, pad_diff), (0, 0)),
                                        mode='edge'))

    # Make it an array
    padded_len_data = np.array(padded_len_data)
    padded_vel_data = np.array(padded_vel_data)
    padded_joint_len_data = np.array(padded_joint_len_data)
    padded_joint_vel_data = np.array(padded_joint_vel_data)
    padded_endeff_pos = np.array(padded_endeff_pos)
    padded_endeff_vel = np.array(padded_endeff_vel)

    padded_len_data_reshaped = np.swapaxes(padded_len_data, 2, 1)
    padded_vel_data_reshaped = np.swapaxes(padded_vel_data, 2, 1)

    padded_joint_len_data_reshaped = np.swapaxes(padded_joint_len_data, 2, 1)
    padded_joint_vel_data_reshaped = np.swapaxes(padded_joint_vel_data, 2, 1)

    #Stack muscle length and vel
    input_kin = np.stack((padded_len_data_reshaped, padded_vel_data_reshaped), axis=3)  #LEN then VEL

    input_joint = np.stack((padded_joint_len_data_reshaped, padded_joint_vel_data_reshaped), axis=3)

    #Stack end-effector coordinates
    endeff_coords = np.stack((padded_endeff_pos, padded_endeff_vel), axis=2) #POS then VEL
    return input_kin, input_joint, endeff_coords

def generate_kin_passive_dataset(kin_df, align):
    '''Generate dataset of padded muscle passive muscle kinematic input from monkey session dataframe.
        Arguments:
        path_to_kin_df - (str) path to folder containing pre-saved kin dataframes.
        monkey_name - (str) name of the monkey, starting with upper case.
        session_date - (int) date of session (YYYYMMDD).'''

    # Dataset structure
    duration_to_reach = 400
    pad_before = int(align)  # all trials are aligned at time point 100
    padded_len_data = []
    padded_vel_data = []
    padded_endeff_pos = []
    padded_endeff_vel = []
    padded_joint_len_data = []
    padded_joint_vel_data = []
    # Indices matching Chan and Moran model: 39
    muscle_len_ids = np.arange(14, 53) #Muscle arrangement in NN #Indices matching Chan and Moran model: 39
    # muscle_vel_ids = np.arange(53, 92)
    joint_len_ids = np.arange(0, 4)

    #Pad each trial containing a passive bump
    bump_ids = kin_df.index[~np.isnan(kin_df['bumpDir'])].tolist()
    for trial_idx in bump_ids:
        bump_idx = int(kin_df.idx_bumpTime[trial_idx])
        end_idx = int(bump_idx + 13)
        trial_dur = end_idx - bump_idx
        if trial_dur + pad_before >= duration_to_reach:
            pad_diff = 0
        else:
            pad_diff = duration_to_reach - trial_dur - pad_before
        len_pad = np.pad(1000 * kin_df['opensim'][trial_idx][bump_idx:end_idx, muscle_len_ids], #in mm/sec (like data gen.)
                                      pad_width=((pad_before, pad_diff), (0, 0)),
                                      mode='edge')
        padded_len_data.append(len_pad)
        padded_vel_data.append(np.gradient(len_pad, 0.01, axis=0))
        padded_endeff_pos.append(np.pad(kin_df['pos'][trial_idx][bump_idx:end_idx, :],
                                      pad_width=((pad_before, pad_diff), (0, 0)),
                                      mode='edge'))
        padded_endeff_vel.append(np.pad(kin_df['vel'][trial_idx][bump_idx:end_idx, :],
                                   pad_width=((pad_before, pad_diff), (0, 0)),
                                   mode='edge'))

        joint_vel_data = np.gradient(kin_df['opensim'][trial_idx][bump_idx:end_idx, joint_len_ids], 0.01, axis=0)
        joint_len_data = np.pad(kin_df['opensim'][trial_idx][bump_idx:end_idx, joint_len_ids],
                                      pad_width=((pad_before, pad_diff), (0, 0)),
                                      mode='edge')
        joint_vel_data = np.pad(joint_vel_data,
                                      pad_width=((pad_before, pad_diff), (0, 0)),
                                      mode='edge')

        padded_joint_len_data.append(joint_len_data)
        padded_joint_vel_data.append(joint_vel_data)


    # Make it an array
    padded_len_data = np.array(padded_len_data)
    padded_vel_data = np.array(padded_vel_data)
    padded_joint_len_data = np.array(padded_joint_len_data)
    padded_joint_vel_data = np.array(padded_joint_vel_data)
    padded_endeff_pos = np.array(padded_endeff_pos)
    padded_endeff_vel = np.array(padded_endeff_vel)

    padded_len_data_reshaped = np.swapaxes(padded_len_data, 2, 1)
    padded_vel_data_reshaped = np.swapaxes(padded_vel_data, 2, 1)

    padded_joint_len_data_reshaped = np.swapaxes(padded_joint_len_data, 2, 1)
    padded_joint_vel_data_reshaped = np.swapaxes(padded_joint_vel_data, 2, 1)

    # Stack muscle length and vel
    input_kin = np.stack((padded_len_data_reshaped, padded_vel_data_reshaped), axis=3)  # LEN then VEL

    input_joint = np.stack((padded_joint_len_data_reshaped, padded_joint_vel_data_reshaped), axis=3)

    #Stack end-effector coordinates
    endeff_coords = np.stack((padded_endeff_pos, padded_endeff_vel), axis=2) #POS then VEL

    return input_kin, input_joint, endeff_coords


def generate_kin_whole_dataset(kin_df):
    '''Generate dataset of padded muscle whole muscle kinematic input from monkey session dataframe.
    Arguments:
    path_to_kin_df - (str) path to folder containing pre-saved kin dataframes.
    monkey_name - (str) name of the monkey, starting with upper case.
    session_date - (int) date of session (YYYYMMDD).'''

    #Dataset structure
    duration_to_reach = 400
    pad_before = 0 #all trials are aligned at time point 100
    padded_len_data = []
    padded_vel_data = []
    padded_endeff_pos = []
    padded_endeff_vel = []
    # Indices matching Chan and Moran model: 39
    muscle_len_ids = np.arange(14, 53) #Muscle arrangement in NN #Indices matching Chan and Moran model: 39
    # muscle_vel_ids = np.arange(53, 92)
    # Indices matching Saul et al: 35 then missing muscles
    muscle_len_ids = [21, 22, 23, 47, 36, 45, 49, 48, 41, 42, 37, 38, 39, 20, 50, 51, 52,
               15, 46, 16, 17, 18, 19, 25, 26, 27, 31, 32, 40, 44, 43, 29, 30, 35, 14]

    #padded_m2_data = []

    #Pad each trial
    for trial_idx in kin_df.index:
        end_idx = int(kin_df.idx_endTime[trial_idx])
        diff = int(end_idx - kin_df.idx_startTime[trial_idx] - 400)

        # Pad if trial duration (start-end) is shorter than 4 sec
        if diff < 0:
            pad_before = np.abs(diff)
            pad_after = 0
            start_idx = kin_df.idx_startTime[trial_idx] #then take from start

            len_pad = np.pad(1000 * kin_df['opensim'][trial_idx][start_idx:end_idx, muscle_len_ids],
                             pad_width=((pad_before, pad_after), (0, 0)),
                             mode='edge')
            endeff_pos_pad = np.pad(kin_df['pos'][trial_idx][start_idx:end_idx, :],
                                        pad_width=((pad_before, pad_after), (0, 0)),
                                        mode='edge')
            endeff_vel_pad = np.pad(kin_df['vel'][trial_idx][start_idx:end_idx, :],
                                        pad_width=((pad_before, pad_after), (0, 0)),
                                        mode='edge')

            #m2_pad = np.pad(kin_df['markers'][trial_idx][start_idx:end_idx, 3:6], #marker 2-xyz coord Snap (18:21), Han(3:6)
            #                            pad_width=((pad_before, pad_after), (0, 0)),
            #                            mode='edge')
        #Else take 4 sec preceding trial end (to include entire reach + bump)
        else:
            start_idx = int(end_idx-400)
            len_pad = 1000 * kin_df['opensim'][trial_idx][start_idx:end_idx, muscle_len_ids]
            endeff_pos_pad = kin_df['pos'][trial_idx][start_idx:end_idx, :]
            endeff_vel_pad = kin_df['vel'][trial_idx][start_idx:end_idx, :]
            #m2_pad = kin_df['markers'][trial_idx][start_idx:end_idx, 3:6]

        #Add each
        padded_len_data.append(len_pad)
        padded_vel_data.append(np.gradient(len_pad, 0.01, axis=0))
        padded_endeff_pos.append(endeff_pos_pad)
        padded_endeff_vel.append(endeff_vel_pad)

        #padded_m2_data.append(m2_pad)


    # Make array
    padded_len_data = np.array(padded_len_data)
    padded_vel_data = np.array(padded_vel_data)
    padded_endeff_pos = np.array(padded_endeff_pos)
    padded_endeff_vel = np.array(padded_endeff_vel)

    #padded_m2_data = np.asarray(padded_m2_data)

    #Change axes
    padded_len_data_reshaped = np.swapaxes(padded_len_data, 2, 1)
    padded_vel_data_reshaped = np.swapaxes(padded_vel_data, 2, 1)

    #Stack muscle length and vel
    input_kin = np.stack((padded_len_data_reshaped, padded_vel_data_reshaped), axis=3)  #LEN then VEL

    #Stack end-effector coordinates
    endeff_coords = np.stack((padded_endeff_pos, padded_endeff_vel), axis=2) #POS then VEL

    return input_kin, endeff_coords, #padded_m2_data