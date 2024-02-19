import os
import importlib

import sys
sys.path.append('./../neural_prediction/Code/')
sys.path.append('./../code/')

import predict_utils as pu

from path_utils import PATH_TO_DATAFRAME_DATADRIVEN

### Script to extract the explained variance from the data-driven networks

## All monkey dset: ('Snap',20190829), ('Butter',20180326), ('Lando', 20170917), ('Han',20171122), ('Chips',20170913), ('S1Lando', 20170917)

## Change the monkey name according to the dset above and the corresponding trained exp_id.

def main():
    monkey_dsets = [('S1Lando', 20170917)]

    for EXP in [11015, 11030, 11045]:
        res_df = pu.load_exp_results_datadriven(EXP, 
                                        monkey_dsets, 
                                        normalize=False, 
                                        load_passive=False, 
                                        params_dict=None, 
                                        tuned_ids=False) 
        ### Save        
        if not os.path.exists(PATH_TO_DATAFRAME_DATADRIVEN):
            os.makedirs(PATH_TO_DATAFRAME_DATADRIVEN)
        
        res_df.to_pickle(os.path.join(PATH_TO_DATAFRAME_DATADRIVEN, 'exp_{}_allmonkey_active_datadriven.pkl'.format(EXP)), protocol=4)


if __name__=='__main__':
    main()