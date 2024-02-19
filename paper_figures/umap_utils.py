import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

params = {
   'axes.labelsize': 20,
   'legend.fontsize': 15,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'text.usetex': False,
   'figure.figsize': [7,7],
   'font.size': 20,
    'lines.markersize':10,
   }

plt.rcParams.update(params)


model_task_dict_name = {'classification': 'Action recognition',
                    'regression': 'Hand localization',
                    'bt': 'Redundancy Reduction',
                    'torque': 'Torque',
                    'regress_ee_vel': 'Hand movements',
                    'regress_ee_pos_vel': 'Hand loc. & mov.',
                    'regress_ee_pos_vel_acc': 'Hand loc., mov. & acc.',
                    'regress_ee_elbow_pos': 'Limb localization',
                    'regress_ee_elbow_vel': 'Limb movements',
                    'regress_ee_elbow_pos_vel': 'Limb loc. & mov.',
                    'regress_ee_elbow_pos_vel_acc': 'Limb loc., mov. & acc.',
                    'regress_joints_pos': 'Joints angles',
                    'regress_joints_vel': 'Joints velocity',
                    'regress_joints_pos_vel': 'Joints ang. & vel.',
                    'regress_joints_pos_vel_acc': 'Joints ang., vel. & acc.'

                    }

model_task_dict_label = {'autoencoder': 'AUTO',
                        'linear': 'LIN',
                        'random': 'RAN',
                        'classification': 'AR',
                        'regression': 'HL',
                        'bt': 'RR',
                        'torque': 'T',
                        'regress_ee_vel': 'HM',
                        'regress_ee_pos_vel': 'HL & HM',
                        'regress_ee_pos_vel_acc': 'HL & HM & HA',
                        'regress_ee_elbow_pos': 'LL',
                        'regress_ee_elbow_vel': 'LM',
                        'regress_ee_elbow_pos_vel': 'LL & LM',
                        'regress_ee_elbow_pos_vel_acc': 'LL & LM & LA',
                        'regress_joints_pos': 'JL',
                        'regress_joints_vel': 'JM',
                        'regress_joints_pos_vel': 'JL & JM',
                        'regress_joints_pos_vel_acc': 'JL & JM & JA'


                        }
                        
task_colormap_dict = {
    'linear': plt.cm.Greys(0.25),
    'random': plt.cm.Greys(0.5),

    'classification': plt.cm.tab20c(0.025),

    'bt': plt.cm.tab20b(0.725),
    'autoencoder': plt.cm.tab20b(0.775),

    'torque': plt.cm.tab20b(0.575),

    'regression': plt.cm.tab20c(0.225),
    'regress_ee_vel': plt.cm.tab20c(0.275),
    'regress_ee_pos_vel': plt.cm.tab20c(0.325),
    'regress_ee_pos_vel_acc': plt.cm.tab20c(0.375),
    
    
    'regress_ee_elbow_pos': plt.cm.tab20c(0.625),
    'regress_ee_elbow_vel': plt.cm.tab20c(0.675),
    'regress_ee_elbow_pos_vel': plt.cm.tab20c(0.725),
    'regress_ee_elbow_pos_vel_acc': plt.cm.tab20c(0.775),

    'regress_joints_pos': plt.cm.tab20c(0.425),
    'regress_joints_vel': plt.cm.tab20c(0.475),
    'regress_joints_pos_vel': plt.cm.tab20c(0.525),
    'regress_joints_pos_vel_acc': plt.cm.tab20c(0.575)
    # "boo": plt.cm.Purples
}

##################### UTILS FUNCTIONS

def color_list_tasks(model_task_list):
    """Return list of colors related to each task

    Args:
        task_order (list of str): List of task

    Returns:
        list of str: Color related to each task
    """
    color_order = [task_colormap_dict[task] for task in model_task_list]
    return color_order

def add_dummy_var_df(res_df):
    """
    Add dummy variable for model test performance
    """
    model_task_list = np.unique(res_df.model_task)
    layer_type_list = np.unique(res_df.layer_type)
    all_test_name_list = []
    for model_task in model_task_list:
        idx_tmp = np.where(res_df.model_task == model_task)[0]
        # idx_tmp = res_df.index[res_df.model_task == model_task].tolist()
        model_test_acc_tmp = np.array(res_df.model_test_acc)
        # model_test_acc_tmp[~np.array(idx_tmp)] = 0

        model_test_acc_new = np.zeros_like(model_test_acc_tmp)
        model_test_acc_new[idx_tmp] = model_test_acc_tmp[idx_tmp]
        # model_test_acc_new[np.array(idx_tmp)] = model_test_acc_tmp[np.array(idx_tmp)]

        name = 'model_test_' + model_task
        # res_df_tmp[name] = model_test_acc_tmp
        res_df[name] = model_test_acc_new
        all_test_name_list.append(name)
    
    all_layer_type_list = []
    for layer_type in layer_type_list:
        idx_tmp = np.where(res_df.layer_type == layer_type)[0]
        # idx_tmp = res_df.index[res_df.model_task == model_task].tolist()
        layer_type_tmp = np.array(res_df.layer_type)
        # model_test_acc_tmp[~np.array(idx_tmp)] = 0

        layer_type_tmp_new = np.zeros_like(layer_type_tmp)
        layer_type_tmp_new[idx_tmp] = 1 #layer_type_tmp[idx_tmp]
        # model_test_acc_new[np.array(idx_tmp)] = model_test_acc_tmp[np.array(idx_tmp)]

        name = 'layer_type_' + layer_type
        # res_df_tmp[name] = model_test_acc_tmp
        res_df[name] = layer_type_tmp_new
        all_layer_type_list.append(name)
    return res_df, all_test_name_list, all_layer_type_list


def get_neural_explainability(res_df_sub, all_layer_type_list):
    res_df_sub = res_df_sub.groupby(['arch_type','model_task','model_name','monkey']+all_layer_type_list+['model_layer'],as_index=False).mean()
    ev = res_df_sub.sort_values(['ev_train'],ascending=False).groupby(['arch_type','model_task','model_name','monkey'],as_index=False,sort=False).head(1)
    return ev


def get_common_models(ev_tmp,model_task_list):
    # Take common models
    model_lists=[]
    for m_name in np.unique(ev_tmp.monkey):
        for task_name in model_task_list:
            res_df_tmp = ev_tmp[ev_tmp.monkey==m_name]
            model_lists.append(np.unique(res_df_tmp[res_df_tmp.model_task==task_name].model_name))
            # print(len(np.unique(res_df_tmp[res_df_tmp.model_task==task_name].model_name)))
    # print(len(model_lists))
    min_model_list = list(set.intersection(*map(set,model_lists)))
    # print(len(min_model_list))
    ev_tmp = ev_tmp[ev_tmp.model_name.isin(min_model_list)]
    return ev_tmp

def refactor_ev_matrix(ev_all_arch,test_ev_list,batch_size,standardize_flag=True):

    monkey_list = ['Snap','Butter','Lando','Han', 'Chips', 'S1Lando']

    all_ev = pd.DataFrame()
    all_ev_before = pd.DataFrame()
    for ii,monkey_tmp in enumerate(monkey_list):

        all_ev_tmp_orig = ev_all_arch[(ev_all_arch.monkey == monkey_tmp)] 
        all_ev_tmp_orig = all_ev_tmp_orig.groupby(['model_task','arch_type','model_name','model_layer'],as_index=False).mean()
        all_ev_tmp = all_ev_tmp_orig[test_ev_list]

        ### Remove random EV - EV DIFF = Task-driven EV - Random EV
        all_ev_tmp = all_ev_tmp_orig[test_ev_list].sub(all_ev_tmp_orig['ev_test_random'], axis=0)
        
        if standardize_flag:
            all_ev_tmp = all_ev_tmp.sub(all_ev_tmp.mean(1), axis=0).div(all_ev_tmp.std(1), axis=0)

        all_ev_tmp = all_ev_tmp[:292]               ### Hard fixed number, divisibile and that take into account all models
        all_ev_tmp_orig = all_ev_tmp_orig[:292]

        all_ev = pd.concat([all_ev,all_ev_tmp])
        all_ev_before = pd.concat([all_ev_before,all_ev_tmp_orig])


    ### Shuffle the order of the models in the group
    ind_shuffled = np.arange(len(all_ev))
    rng = np.random.RandomState(12)
    rng.shuffle(ind_shuffled)
    ind_shuffled_list = [ind_shuffled[start_idx*batch_size:(start_idx+1)*batch_size] for start_idx in range(len(all_ev)//batch_size)]

    new_shuffled_ev = [] # pd.DataFrame()
    for ii in range(len(ind_shuffled_list)):
        tmp_tmp_ev = all_ev.iloc[ind_shuffled_list[ii]]
        new_shuffled_ev.append(np.array(tmp_tmp_ev).T)
    new_shuffled_ev = np.vstack(new_shuffled_ev)

    all_ev_refactor = new_shuffled_ev

    n_batches = all_ev_refactor.shape[0]//16

    return all_ev_refactor, n_batches


################# PLOTTING UMAP

from matplotlib.colors import ListedColormap
import matplotlib
def plot_embedding_task_all(embedding,n_batches,task_list,standardize_flag,save_flag,PATH_TO_FIG,suffix_emb='umap'):
    
    c=color_list_tasks(task_list)
    c1 = []
    for _ in range(n_batches):
        # c1.extend([c[ii]]*17)
        c1.extend(c)

    fig = plt.figure(figsize=[6,6])
    
    # new_dict_col_list = dict(zip(model_task_list,np.arange(0,len(model_task_list))))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],s=50,c=c1)

    if suffix_emb == 'umap':
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
    elif suffix_emb == 'pca':
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')

    fig.tight_layout()

    if save_flag:
        suffix = ''
        suffix1 = ''
        if standardize_flag:
            suffix1 = '_standardize'
        plt.savefig(PATH_TO_FIG + suffix_emb + '_task_all_' + suffix + suffix1 + '.png', format='png', dpi=600, bbox_inches='tight')
        plt.savefig(PATH_TO_FIG + suffix_emb + '_task_all_' + suffix + suffix1 + '.pdf', format='pdf', dpi=600, bbox_inches='tight')
        plt.savefig(PATH_TO_FIG + suffix_emb + '_task_all_' + suffix + suffix1 + '.svg', format='svg', dpi=600, bbox_inches='tight')
        
    return

def plot_embedding_task_all_ev(embedding,all_ev,standardize_flag,save_flag,PATH_TO_FIG,suffix_emb='umap'):
    

    fig = plt.figure(figsize=[8,6])

    ev_mean_all = np.array(all_ev.mean(axis=1))
    ev_color  = (ev_mean_all - ev_mean_all.min()) / (ev_mean_all.max() - ev_mean_all.min())

    my_cmap = ListedColormap(sns.color_palette("flare"))
    norm = mpl.colors.Normalize(vmin=ev_mean_all.min(), vmax=ev_mean_all.max())

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1], s =50 ,c=[sns.color_palette("flare", as_cmap=True)(x) for x in ev_color])
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=my_cmap))

    if suffix_emb == 'umap':
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
    elif suffix_emb == 'pca':
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')

    fig.tight_layout()

    if save_flag:
        suffix = ''
        suffix1 = ''
        if standardize_flag:
            suffix1 = '_standardize'
        plt.savefig(PATH_TO_FIG + suffix_emb + '_task_ev_all_' + suffix + suffix1 + '.png', format='png', dpi=600, bbox_inches='tight')
        plt.savefig(PATH_TO_FIG + suffix_emb + '_task_ev_all_' + suffix + suffix1 + '.pdf', format='pdf', dpi=600, bbox_inches='tight')
        plt.savefig(PATH_TO_FIG + suffix_emb + '_task_ev_all_' + suffix + suffix1 + '.svg', format='svg', dpi=600, bbox_inches='tight')
        
    return