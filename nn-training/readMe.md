**This folder contains code to train neural network models of proprioception.**

Task-driven neural networks training:
* `train_all_regression_task.sh` is the script to train DNNs on the regression tasks (hand localization, limb localization, ...) and `train_all_regression_task.py` contains the corresponding code. To train on a different regression task, change the task, exp_id, the target keys in the bash script and add a folder in deepdraw_models with the name of the task and containing the corresponding hyperparameters.
* `train_all_autoencoder_task.sh` is the script to train DNNs on the autoencoder task and `train_all_autoencoder_task.py` contains the corresponding code. 
* `train_all_torque_task.sh` is the script to train DNNs on the joint torque regression task and `train_all_torque_task.py` contains the corresponding code. 
* `train_all_barlow_task.sh` is the script to train DNNs on the redundancy reduction task and `train_all_barlow_task.py` contains the corresponding code. 

Data-driven neural networks training:
* `train_datadriven_networks.sh` is the script to train DNNs directly to predict the neural activity (data-driven) and `train_datadriven_networks.py` contains the corresponding code. 

In order to train networks, please make sure that you use the provided docker container and that your host system has CUDA and CudNN specifications met. Additionally, check the global paths set in `code/path_utils.py`. 

In order to train networks: 
* please make sure that you use the provided docker container and that your host system has CUDA and CudNN specifications met
* check the global paths set in `code/path_utils.py`
* The `deepdraw_models` folder should contain a folder with the name of the task where the hyperparameters for each conv type are specified

Each folder in `experiment_XXXX` contain configurations and weights corresponding to multiple networks for the same task or model type. To look at the configurations (hyperparameter settings) of the network, browse the `config.yaml` file inside each sub-folder. 