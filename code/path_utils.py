import os

### Change the following path with the folder where you downloaded the data.
ROOT_PATH = '/PATH_TO_DATA'

###################        PCR-DATA-GENERATION and RL-DATA-GENERATION               ###################
# 
# PCR-dataset generation
PATH_TO_STARTPOINT = os.path.join(ROOT_PATH,'PCR-data-generation','start_points')
PATH_TO_SPINDLES = os.path.join(ROOT_PATH,'spindle_datasets')

PATH_TO_SAVE_SPINDLEDATASET =  os.path.join(ROOT_PATH,'spindle_datasets','pcr_dataset')

PATH_TO_UNPROCESSED_DATA =  os.path.join(ROOT_PATH,'spindle_datasets','pcr_dataset','unprocessed_data') # '../data/'  #or PATH_TO_SPINDLES but make sure that volume is connected to the docker
PATH_TO_UNPROCESSED_RL_DATA =  os.path.join(ROOT_PATH,'spindle_datasets','rl_dataset','unprocessed_data')
PATH_TO_CONVERTED_RL_DATA =  os.path.join(ROOT_PATH,'spindle_datasets','rl_dataset','converted_data')

PATH_TO_SAVE_RL_DATASET =  os.path.join(ROOT_PATH,'spindle_datasets','rl_dataset')

###################                     NETWORKS TRAINING                           ###################

### Path to datasets
PATH_TO_DATA = os.path.join(ROOT_PATH, 'spindle_datasets', 'pcr_dataset')
PATH_TO_DATA_RL = os.path.join(ROOT_PATH, 'spindle_datasets', 'rl_dataset')
PATH_TO_DATA_SPIKES = os.path.join(ROOT_PATH,'exp_analysis','beh_exp_datasets','MonkeySpikeRegressDatasets')

### Path to models dataframe (hyperparameters)
PATH_TO_OLD = os.path.join(ROOT_PATH,'models','deepdraw_models')

### Path for saving models (should contains also exp: 4015, 5015, 4045 for initialization)
MODELS_DIR = os.path.join(ROOT_PATH,'models')

### Path for saving datadriven models
PATH_TO_RESULTS_DATADRIVEN = os.path.join(ROOT_PATH,'exp_analysis','results','data_driven')
PATH_TO_DATAFRAME_DATADRIVEN = os.path.join(ROOT_PATH,'exp_analysis','results','dataframes','data_driven')

###################                     EXPERIMENTAL DATA                           ###################

### Path to experimental data
PATH_TO_BEH_EXP = os.path.join(ROOT_PATH,'exp_analysis','beh_exp_datasets') 
PATH_MONKEY_PROCESSED_DATAFRAMES = os.path.join(ROOT_PATH,'exp_analysis','beh_exp_datasets','processed_dataframes')
PATH_MONKEY_PROCESSED_DICT = os.path.join(ROOT_PATH,'exp_analysis','beh_exp_datasets','processed_dict')

### Datasplit used for active data
PATH_TO_DATASPLITS = os.path.join(ROOT_PATH,'exp_analysis','beh_exp_datasets','MonkeySpikeRegressDatasets')

### Path where original matlab files are saved
PATH_TO_MATLAB_DATA = os.path.join(ROOT_PATH,'exp_analysis','beh_exp_datasets','matlab_data') 

### Path where Neural data is stored
PATH_TO_NEURAL_DATA = os.path.join(ROOT_PATH,'exp_analysis','beh_exp_datasets','MonkeyAlignedDatasets_new')
PATH_TO_NEURAL_DATA_NOTALIGN = os.path.join(ROOT_PATH,'exp_analysis','beh_exp_datasets','MonkeyDatasets')

PATH_TO_SPIKE_REGRESS_DATA_PASSIVE = os.path.join(ROOT_PATH,'exp_analysis','beh_exp_datasets','MonkeySpikeRegressDatasets_passive')

### Path for saving predictions from linear models
PATH_TO_SAVE_LINEAR = os.path.join(ROOT_PATH,'exp_analysis','predictions')

### Path for activations and predictions
PATH_TO_ACTIVATIONS = os.path.join(ROOT_PATH,'exp_analysis','activations')
PATH_TO_PREDICTIONS = os.path.join(ROOT_PATH,'exp_analysis','predictions')

PATH_TO_RESULTS = os.path.join(ROOT_PATH,'exp_analysis','results')