import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy as sp

############################## PLOT UTILS

params = {
   'axes.labelsize': 24,
   'legend.fontsize': 15,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'text.usetex': False,
   'figure.figsize': [7,7],
   'font.size': 20,
    'lines.markersize':10,
   }

plt.rcParams.update(params)

def remove_top_right_frame(ax):
    '''Remove top and right parts of the figure frame. Takes an 'ax' object as input from plt.subplots.'''
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    return

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

############################## MODEL TASK NAME

model_task_dict_name = {'autoencoder': 'Autoencoder',
                    'classification': 'Action recognition',
                    'regression': 'Hand position',
                    'bt': 'Redundancy Reduction',
                    'torque': 'Torque',
                    'regress_ee_vel': 'Hand velocity',
                    'regress_ee_pos_vel': 'Hand pos. & vel.',
                    'regress_ee_pos_vel_acc': 'Hand pos., vel. & acc.',
                    'regress_ee_elbow_pos': 'Limb position',
                    'regress_ee_elbow_vel': 'Limb velocity',
                    'regress_ee_elbow_pos_vel': 'Limb pos. & vel.',
                    'regress_ee_elbow_pos_vel_acc': 'Limb pos., vel. & acc.',
                    'regress_joints_pos': 'Joints position',
                    'regress_joints_vel': 'Joints velocity',
                    'regress_joints_pos_vel': 'Joints pos. & vel.',
                    'regress_joints_pos_vel_acc': 'Joints pos., vel. & acc.'

                    }

############################## COLOR ARCHITECTURE

## Set1 cmap
cmap = matplotlib.cm.get_cmap('Set1')

st_color = cmap(0.3)
t_s_color = cmap(0.1)
s_t_color = cmap(0.2)
lstm_color = cmap(0.4)

## tab10 cmap
cmap = matplotlib.cm.get_cmap('tab10')

st_color = cmap(0.2)
t_s_color = cmap(0.1)
s_t_color = cmap(0)
lstm_color = cmap(0.4)

cmap = matplotlib.cm.get_cmap('tab10')

color_arch = {
    'spatiotemporal': cmap(0),
    'temporal_spatial': cmap(0.1),
    'spatial_temporal': cmap(0.2),
    'lstm': cmap(0.4)
}



########################## TASK PERFORMANCE

def add_arch_type(res_df_tmp):
    arch_type = []
    # model_task_tmp = 'torque'
    # res_df_torque = res_df[res_df.model_task == model_task_tmp]

    for ii in range(len(res_df_tmp)):
        model_name_tmp = res_df_tmp.iloc[ii].model_name
        if 'spatial_temporal' in model_name_tmp:
            arch_type.append('spatial_temporal')
        elif 'temporal_spatial' in model_name_tmp:
            arch_type.append('temporal_spatial')
        elif 'spatiotemporal' in model_name_tmp:
            arch_type.append('spatiotemporal')
        elif 'lstm' in model_name_tmp:
            arch_type.append('lstm')

    res_df_tmp['arch_type'] = arch_type
    return res_df_tmp

def errobar_task_perf(res_df_torque, task, save_flag, PATH_TO_FIG):
    all_arch_type = np.unique(res_df_torque['arch_type'])
    avg_task_perf_arch_type = {}
    std_task_perf_arch_type = {}
    max_all_layer_dict = {}
    for jj in range(len(all_arch_type)):

        avg_task_perf_layer = []
        std_task_perf_layer = []
        arch_type_tmp = all_arch_type[jj]
        df_sub = res_df_torque[res_df_torque.arch_type == arch_type_tmp]
        max_all_layer = np.unique(df_sub['model_max_layer'])
        for ii, layer_tmp in enumerate(max_all_layer):
            
            df_sub_tmp = df_sub[df_sub.model_max_layer == layer_tmp] #['model_max_layer']
            # print(df_sub)
            avg_task_perf = df_sub_tmp['model_test_acc'].mean()
            std_task_perf = df_sub_tmp['model_test_acc'].std()

            avg_task_perf_layer.append(avg_task_perf)
            std_task_perf_layer.append(std_task_perf)
        avg_task_perf_arch_type[arch_type_tmp] = avg_task_perf_layer
        std_task_perf_arch_type[arch_type_tmp] = std_task_perf_layer
        # if arch_type_tmp == 'lstm':
        #     max_all_layer_dict[arch_type_tmp] = max_all_layer - 1
        # else:
        max_all_layer_dict[arch_type_tmp] = max_all_layer


    all_layer_ticks = np.arange(1,13)
    # fig,ax = plt.subplots(1,3,figsize=[20,6])
    fig,ax = plt.subplots(figsize=[6,6])
    for jj,arch_type_tmp in enumerate(all_arch_type):
        max_all_layer = max_all_layer_dict[arch_type_tmp]

        if arch_type_tmp == 'lstm':
            max_all_layer = max_all_layer -1

        ax.scatter(max_all_layer,avg_task_perf_arch_type[arch_type_tmp],s=30, color=color_arch[arch_type_tmp])
        ax.errorbar(max_all_layer, avg_task_perf_arch_type[arch_type_tmp], yerr=std_task_perf_arch_type[arch_type_tmp], color=color_arch[arch_type_tmp])
        
    ax.set_xticks(all_layer_ticks)
    ax.set_xticklabels(all_layer_ticks) #,rotation = 90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('N° of layers')
    if task == 'classification':
        plt.ylabel('Test accuracy')
    elif task == 'bt':
        plt.ylabel('Test Barlow loss')
    else:
        plt.ylabel('Test MSE')
    plt.title(model_task_dict_name[task])
    if save_flag:

        plt.savefig(PATH_TO_FIG + '/task_perf_'+task+'.png', format='png', dpi=600, bbox_inches='tight')
        plt.savefig(PATH_TO_FIG + '/task_perf_'+task+'.pdf', format='pdf', dpi=600, bbox_inches='tight')
        plt.savefig(PATH_TO_FIG + '/task_perf_'+task+'.svg', format='svg', dpi=600, bbox_inches='tight')
    plt.show()
    return

def swarm_task_perf(res_df_torque, task, save_flag, PATH_TO_FIG):

    if task == 'bt':
        res_df_torque = res_df_torque[res_df_torque.model_test_acc<4.5]
    elif task == 'autoencoder':
        res_df_torque = res_df_torque[res_df_torque.model_test_acc<60]
    elif task == 'regress_joints_pos_vel_acc':
        res_df_torque = res_df_torque[res_df_torque.model_test_acc<200]


    all_arch_type = np.unique(res_df_torque['arch_type'])
    all_layer_ticks = np.arange(1,13)

    fig,ax = plt.subplots(figsize=[5,5])

    # ax = sns.swarmplot(x='model_max_layer', y='model_test_acc', 
    #             data=res_df_torque, hue='arch_type', edgecolor='white', linewidth=0.8, size=6, palette = [st_color,t_s_color,s_t_color,lstm_color])
    ax = sns.stripplot(ax=ax,x='model_max_layer', y='model_test_acc', 
                data=res_df_torque, hue='arch_type', edgecolor='white', linewidth=0.8, size=6, jitter=.23, palette = [st_color,t_s_color,s_t_color,lstm_color])
        
    # ax.set_xticks(all_layer_ticks)
    ax.set_xticklabels(all_layer_ticks) #,rotation = 90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend([],[],frameon=False)
    # ax.locator_params(axis='x', nbins=6)

    # if task in ['classification','regression','regress_ee_elbow_pos','bt','regress_joints_pos']:
    #     # plt.xlabel('')
    #     ax.set_xticks([])
    #     ax.set(xlabel=None)
    # else:
    plt.xticks(np.arange(1,13,2), np.arange(2,13,2))
    plt.xlabel('N° of layers')
    
    if task == 'classification':
        plt.ylabel('Test accuracy')
    elif task == 'bt':
        plt.ylabel('Test Barlow loss')
    elif task in ['autoencoder','regression','regress_ee_elbow_pos']:
        plt.ylabel('Test MSE [cm]')
        plt.ylim(0,)
    elif task in ['regress_ee_vel','regress_ee_elbow_vel']:
        plt.ylabel('Test MSE [cm/s]')
        plt.ylim(0,)
    elif task == 'regress_joints_pos':
        plt.ylabel('Test MSE [deg]')
        plt.ylim(0,)
    elif task == 'regress_joints_vel':
        plt.ylabel('Test MSE [deg/s]')
        plt.ylim(0,)
    else:
        plt.ylabel('Test MSE')
    plt.title(model_task_dict_name[task])
    if save_flag:

        plt.savefig(PATH_TO_FIG + '/swarm_task_perf_'+task+'.png', format='png', dpi=600, bbox_inches='tight')
        plt.savefig(PATH_TO_FIG + '/swarm_task_perf_'+task+'.pdf', format='pdf', dpi=600, bbox_inches='tight')
        plt.savefig(PATH_TO_FIG + '/swarm_task_perf_'+task+'.svg', format='svg', dpi=600, bbox_inches='tight')
    plt.show()
    return


#### Correlation figure

def plot_ev_vs_task_singlemonkey_singletask(res_df_snap,model_task, model_type_list, n_max_layer, flag_dict, PATH_TO_FIG):
    area = flag_dict['area']
    mean_flag = flag_dict['mean_flag']
    median_flag = flag_dict['median_flag']
    mean_flag_all = flag_dict['mean_flag_all']
    standardize_flag = flag_dict['standardize_flag']
    # task_diff_flag = flag_dict['task_diff_flag']
    quantile_flag = flag_dict['quantile_flag']
    quantile_value = flag_dict['quantile_value']
    model_layer_flag = flag_dict['model_layer_flag']
    model_layer = flag_dict['model_layer']
    test_flag = flag_dict['test_flag']
    tuned_flag = flag_dict['tuned']
    save_flag = flag_dict['save_flag']
    norm_flag = flag_dict['norm_flag']
    layer_neural_expl_flag = flag_dict['layer_neural_expl_flag']
    neuron_neural_expl_flag = flag_dict['neuron_neural_expl_flag']

    if area == 'CN':
        color = 'crimson'
    else:
        color = 'darkslateblue'

    ev_str = "ev_test" if test_flag else "ev_train"
    ev_train = "ev_train"
    if norm_flag:
        ev_str += '_norm'
        ev_train += '_norm'
    
    monkey_list = np.unique(res_df_snap.monkey)
    ev_task = {}
    min_list = []
    max_list = []
    best_layers_task = {}
    
    fig, ax = plt.subplots(1,1,figsize=[5,7])

    ev_monkey = {}
    best_layers_monkey = {}
    for i in range(len(monkey_list)):
        res_df_sub = res_df_snap[res_df_snap.model_task==model_task]
        res_df_sub = res_df_sub[res_df_sub.monkey==monkey_list[i]]
        if standardize_flag:
            res_df_sub[ev_str] = (res_df_sub[ev_str]-res_df_sub[ev_str].mean())/res_df_sub[ev_str].std()

        ### Here LSTM are removed
        res_df_sub = res_df_sub[res_df_sub.arch_type != 'lstm']
        
        scores = res_df_sub.groupby(['model_name'])['model_test_acc'].mean()

        best_layers_tmp = 0
        if layer_neural_expl_flag:
            res_df_sub = res_df_sub.groupby(['model_name','monkey','model_layer'],as_index=False).mean()
            ev = res_df_sub.sort_values([ev_train],ascending=False).groupby(['model_name','monkey'],as_index=False,sort=False).head(1)
            # best_layers_tmp = ev.model_layer
            ev_depth = ev[ev.model_max_layer == n_max_layer]
            # best_layers_tmp = 100*(np.array(ev_depth.model_layer) / (np.array(ev_depth.model_max_layer)-1))
            best_layers_tmp = np.array(ev_depth.model_layer)
            ev = ev.groupby(['model_name'],as_index=False)[ev_str].mean(ev_str)
            ev = ev[ev_str]

        if neuron_neural_expl_flag:
            ## Select best layer on train set
            ev = res_df_sub.sort_values([ev_train],ascending=False).groupby(['model_name','monkey','neuron_ids'],as_index=False,sort=False).head(1)
            ev = ev.groupby(['model_name','monkey'],as_index=False)[ev_str].mean(ev_str).groupby(['model_name'],as_index=False)[ev_str].mean(ev_str)
            
            ev = ev[ev_str]

        if median_flag:
            ## Select best layer on train set
            ev = res_df_sub.groupby(['model_name','monkey','neuron_ids'])[[ev_str,ev_train]].max(ev_train).groupby(['model_name','monkey'])[ev_str].mean().groupby(['model_name']).median()

        if mean_flag_all:
            ev = res_df_sub.groupby(['model_name','arch_type','neuron_ids'])[ev_str].mean().groupby(['model_name','arch_type']).mean().groupby(['model_name']).mean()

        min_list.append(min(ev))
        max_list.append(max(ev))
        ev_monkey[monkey_list[i]] = (ev,scores)
        best_layers_monkey[monkey_list[i]] = best_layers_tmp

        ev_task[model_task] = ev_monkey
        best_layers_task[model_task] = best_layers_monkey

    all_min = min(min_list)
    all_max = max(max_list)
    if all_min < 0:
        all_min = 0.09 #0.1
    all_min = all_min - 1/10 * (all_max-all_min)
    all_max = all_max + 1/10 * (all_max-all_min)
    color_monkey = np.linspace(0.4,1,len(monkey_list))

    for j in range(len(monkey_list)):
        ev, scores = ev_task[model_task][monkey_list[j]]

        remove_top_right_frame(ax)

        if monkey_list[j] == 'S1Lando':
            monkey_name_tmp = 'S1L'
        else:
            monkey_name_tmp = monkey_list[j][0]

        ax.scatter(y=ev,
                    x=scores,
                    color = lighten_color(color,color_monkey[j]),
                s=15, label='m={}, r={:.3f}, p={:.2e}'.format(monkey_name_tmp,sp.stats.pearsonr(scores,ev)[0],
                                            sp.stats.pearsonr(scores,ev)[1])) #,c=color)
        m, b = np.polyfit(scores, ev, 1)

        ax.plot(scores, m*scores+b, ls='-', lw=2, color = lighten_color(color,color_monkey[j])) #, c=color) 
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.5),
            ncol=1)

    ax.locator_params(axis='y', nbins=4)
            
    ax.set_ylim((all_min,all_max))
    # ax[i].set_ylim((0.1,0.4))
    
    ax.set_ylabel('Test explained variance')
    
    if model_task == 'classification':
        ax.set_xlabel('Test accuracy')
    elif model_task == 'bt':
        ax.set_xlabel('Test Barlow loss')
    else:
        ax.set_xlabel('Test MSE')
    
    ax.set_title(model_task_dict_name[model_task])
    fig.tight_layout()
    
    if save_flag:
        if test_flag:
            suffix_ev = 'test'
        else:
            suffix_ev = 'train'

        suffix = 'sin_monk'

        if mean_flag:
            suffix1 = 'max'
        elif mean_flag_all:
            suffix1 = 'mean'
        else:
            suffix1 = ''
        if standardize_flag:
            suffix2 = 'std'
        else:
            suffix2 = ''
        
        if median_flag:
            suffix3 = 'med'
        else:
            suffix3 = ''
        
        suffix4 = ''
        if layer_neural_expl_flag:
            suffix4 = '_layer'

        plt.savefig(PATH_TO_FIG + 'ev_' + suffix_ev + '_vs_perf_' + suffix + '_' + suffix1 + '_' + suffix2 + '_' + suffix3 + suffix4 + '_' + model_task + '.png', format='png', dpi=600, bbox_inches='tight')
        plt.savefig(PATH_TO_FIG + 'ev_' + suffix_ev + '_vs_perf_' + suffix + '_' + suffix1 + '_' + suffix2 + '_' + suffix3 + suffix4 + '_' + model_task + '.pdf', format='pdf', dpi=600, bbox_inches='tight')
        plt.savefig(PATH_TO_FIG + 'ev_' + suffix_ev + '_vs_perf_' + suffix + '_' + suffix1 + '_' + suffix2 + '_' + suffix3 + suffix4 + '_' + model_task + '.svg', format='svg', dpi=600, bbox_inches='tight')
    plt.show()
    return best_layers_task


def plot_layer_hist_single_task_layer(res_df_snap,best_layer_task,model_task, n_max_layer, flag_dict,PATH_TO_FIG):
    area = flag_dict['area']
    mean_flag = flag_dict['mean_flag']
    median_flag = flag_dict['median_flag']
    mean_flag_all = flag_dict['mean_flag_all']
    standardize_flag = flag_dict['standardize_flag']
    # task_diff_flag = flag_dict['task_diff_flag']
    quantile_flag = flag_dict['quantile_flag']
    quantile_value = flag_dict['quantile_value']
    model_layer_flag = flag_dict['model_layer_flag']
    model_layer = flag_dict['model_layer']
    test_flag = flag_dict['test_flag']
    tuned_flag = flag_dict['tuned']
    save_flag = flag_dict['save_flag']
    norm_flag = flag_dict['norm_flag']
    layer_neural_expl_flag = flag_dict['layer_neural_expl_flag']
    neuron_neural_expl_flag = flag_dict['neuron_neural_expl_flag']

    if area == 'CN':
        color = 'crimson'
    else:
        color = 'darkslateblue'

    ev_str = "ev_test" if test_flag else "ev_train"
    ev_train = "ev_train"
    if norm_flag:
        ev_str += '_norm'
        ev_train += '_norm'
    
    monkey_list = np.unique(res_df_snap.monkey)


    fig, ax = plt.subplots(1,1,figsize=[5,7])
    color_monkey = np.linspace(0.4,1,len(monkey_list))


    for j in range(len(monkey_list)):
        best_layer = best_layer_task[model_task][monkey_list[j]]

        remove_top_right_frame(ax)

        if monkey_list[j] == 'S1Lando':
            monkey_name_tmp = 'S1L'
        else:
            monkey_name_tmp = monkey_list[j][0]

        ax.hist(best_layer,
                    color = lighten_color(color,color_monkey[j]),
                    alpha=0.5,
                label='m={}'.format(monkey_name_tmp)) #,c=color)
        
        if n_max_layer == 2:
            ax.axvline(x=np.median(best_layer)+j*0.02,linestyle='--',color = lighten_color(color,color_monkey[j]))
        elif (n_max_layer > 2) and (n_max_layer < 9):
            ax.axvline(x=np.median(best_layer)+j*0.05,linestyle='--',color = lighten_color(color,color_monkey[j]))
        else:
            ax.axvline(x=np.median(best_layer)+j*0.1,linestyle='--',color = lighten_color(color,color_monkey[j]))
        
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.5),
            ncol=1)

            
    ax.set_ylim((0,42))
    best_layer_tmp = best_layer_task[model_task][monkey_list[0]]
    ax.set_xticks(np.arange(1,np.max(best_layer_tmp)+2,2))
    new_label = [str(int(perc)) for perc in np.arange(2,np.max(best_layer_tmp)+3,2)]
    ax.set_xticklabels(new_label,fontsize=20)


    ax.set_ylabel('Frequency')
    ax.set_xlabel('Model layer')
    
    ax.set_title(model_task_dict_name[model_task])

    fig.tight_layout()
    
    if save_flag:
        if test_flag:
            suffix_ev = 'test'
        else:
            suffix_ev = 'train'

        suffix = 'sin_monk'

        if mean_flag:
            suffix1 = 'max'
        elif mean_flag_all:
            suffix1 = 'mean'
        else:
            suffix1 = ''
        if standardize_flag:
            suffix2 = 'std'
        else:
            suffix2 = ''
        
        if median_flag:
            suffix3 = 'med'
        else:
            suffix3 = ''
        
        suffix4 = ''
        if layer_neural_expl_flag:
            suffix4 = '_layer'

        suffix5 = '_depth' + str(n_max_layer)

        plt.savefig(PATH_TO_FIG + 'num_layer_hist_' + suffix_ev + '_' + suffix + '_' + suffix1 + '_' + suffix2 + '_' + suffix3 + suffix4 + '_' + model_task + suffix5 + '.png', format='png', dpi=600, bbox_inches='tight')
        plt.savefig(PATH_TO_FIG + 'num_layer_hist_' + suffix_ev + '_' + suffix + '_' + suffix1 + '_' + suffix2 + '_' + suffix3 + suffix4 + '_' + model_task + suffix5 + '.pdf', format='pdf', dpi=600, bbox_inches='tight')
        plt.savefig(PATH_TO_FIG + 'num_layer_hist_' + suffix_ev + '_' + suffix + '_' + suffix1 + '_' + suffix2 + '_' + suffix3 + suffix4 + '_' + model_task + suffix5 + '.svg', format='svg', dpi=600, bbox_inches='tight')
    plt.show()
    return