# Imports
import sys
import os
import argparse
from multiprocessing import Pool

# Modules
import compute_model_session_predictivity_h5_cv_align

def pool_model_pred(inputs):

    monkey_name, session_date, exp_id, train_iter, model_name, window, latency, active_start, active_length, align, control_dict, shuffle_flag, active_as_passive_flag, path_to_act, path_to_results = inputs


    compute_model_session_predictivity_h5_cv_align.main(monkey_name=monkey_name,
                                                    session_date=session_date,
                                                    exp_id=exp_id,
                                                    train_iter=train_iter,
                                                    model_name=model_name,
                                                    window=window,
                                                    latency=latency,
                                                    # cv_winlat=cv_winlat,
                                                    active_start=active_start,
                                                    active_length=active_length,
                                                    align=align,
                                                    control_dict=control_dict,
                                                    shuffle_flag=shuffle_flag,
                                                    active_as_passive_flag=active_as_passive_flag,
                                                    path_to_act=path_to_act,
                                                    path_to_results=path_to_results)
    return


def main(monkey_name, session_date, exp_id, train_iter=None, active_start='mvt', active_length=0, align=100,
         permut_m=False, permut_t=False, constant_input=False, window=5, latency=0, cv_winlat=False, shuffle_flag=False, active_as_passive_flag=False, path_to_act='..', path_to_results='..'):
    '''
    Script to make predictions of one monkey session dataset,
    for all models with activations from DNN experiment, if predictions not already provided.

    :param monkey_name: Name of monkey, first letter capitalized.
    :param session_date: Monkey session date DDMMYYYY.
    :param exp_id: DNN experiment number, 'exp_<int>'
    :param train_iter: Training iteration i.e. training checkpoint integer. 0 for untrained models.
    :param active_start: Trial event from which wet get behavioural/neural data (mvt, cue).
    :param active_length: Duration of active data from active_start, in multiple of 10ms.
    :param align: Index of trial onset alignment.
    :param permut_m: Whether to permute muscle inputs.
    :param permut_t: Whether to permute time inputs.
    :param constant_input: Whether to feed constant muscle inputs.
    :param window: Window parameter of neural data smoothing, in multiple of 10ms.
    :param latency: Latency paramter of neural data delaying, in multiple of 10ms.
    :param cv_winlat: Whether to cross-validate window/latency parameters.
    :return:
    '''

    PATH_TO_ACTIVATIONS = path_to_act
    print(path_to_act)
    # LIST DNN MODELS WITH COMPUTED ACTIVATIONS
    path_to_exp_act = os.path.join(PATH_TO_ACTIVATIONS, 'experiment_{}'.format(exp_id))
    print('Getting models in:', path_to_exp_act)
    model_list = os.listdir(path_to_exp_act)
    print('(exp_{}) Number of models: {}'.format(exp_id, len(model_list)))

    ## This model doesn't converge for Chips
    if int(exp_id) in [20516]:
        model_list = [i for i in model_list if i not in ['spatiotemporal_r_1_64_5252']]


    # COMPUTE NEURAL PREDICTIVITY FOR EACH MODEL
    print('SELECTION ONE MODEL FOR CV WINLAT')
    print('cv_winlat', cv_winlat)

    control_dict = {'permut_m': permut_m,
                    'permut_t': permut_t,
                    'constant_input': constant_input}

    all_files = [[monkey_name, session_date, exp_id, train_iter, model_name, window, latency, active_start, active_length, align, control_dict, shuffle_flag, active_as_passive_flag, path_to_act, path_to_results] for model_name in model_list]

    p = Pool()
    
    p.map(pool_model_pred, all_files)
    p.close()
    p.join()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predictivity script for one monkey session data, all available models.')

    parser.add_argument('--monkey', type=str, help='Which monkey?', required=True)
    parser.add_argument('--session', type=int, help='Which session data?', required=True)
    parser.add_argument('--exp_id', type=str, help='Which experiment activation folder ?', required=True)
    parser.add_argument('--train_iter', type=int, default=None, help='Which training checkpoint index?', required=False)
    parser.add_argument('--active_start', type=str, default='mvt', help='Which active start index?', required=True)
    parser.add_argument('--active_length', type=int, default=0,
                        help='Length after movement onset (1bin=10ms)? [None if passive/hold].', required=False)
    parser.add_argument('--align', type=int, default=0, help='Index of trial onset alignment.', required=False)
    parser.add_argument('--permut_m', action='store_false', help='Permut muscles control?', required=False)
    parser.add_argument('--permut_t', action='store_false', help='Permut time control?', required=False)
    parser.add_argument('--constant_input', action='store_false', help='Constant input control?', required=False)
    parser.add_argument('--window', type=int, default=5, required=False)
    parser.add_argument('--latency', type=int, default=0, required=False)
    parser.add_argument('--cv_winlat', action='store_true', help='Cross-validate window/latency?', required=False)
    parser.add_argument('--shuffle', action='store_true', help='Flag to shuffle time of activations (e.g. shuffle if true)', default = False)
    parser.add_argument('--act_as_pas', action='store_true', help='Flag to reduce the number of active mov equal to passive (e.g. act_as_pas if true)', default = False)
    parser.add_argument('--path_to_act', type=str, help='Path where to load activations', required=False, default = '..')
    parser.add_argument('--path_to_results', type=str, help='Path where to save predictions', required=False, default = '..')
    args = parser.parse_args()

    main(args.monkey,
         args.session,
         args.exp_id,
         args.train_iter,
         args.active_start,
         args.active_length,
         args.align,
         args.permut_m,
         args.permut_t,
         args.constant_input,
         args.window,
         args.latency,
         args.cv_winlat,
         args.shuffle,
         args.act_as_pas,
         args.path_to_act, 
         args.path_to_results)
