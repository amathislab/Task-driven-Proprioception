**This folder contains the code to use pre-trained neural networks to predict experimental monkey neural data:**

The folder contains the code to predict neural data using neural network models trained on a task.
The Code contains the following files, to execute in this order:
  - `1_run_sessions_75pca_h5_align.sh` & `1_run_sessions_75pca_h5_align_passive.sh` bash script to generate network's activations giving as input the test behavioral muscle spindles inputs to the frozen neural networks (run in the docker). The PCs of the activations are stored. The same can be applied for untrained models by adding the `_untrained` suffix. It uses the following scripts `generate_session_activations_active.py` & `generate_session_activations_passive.py`.
  - `2_run_predictions_75pca_h5_align.sh` & `2_run_predictions_75pca_h5_align_passive.sh` bash script to predict neural activity using the previously generated network's activations (use conda environment _DeepProprio_). The same can be applied for untrained models by adding the `_untrained` suffix. It uses the following scripts `compute_session_predictivity_h5_cv_pool_align.py`, `compute_model_session_predictivity_h5_cv_align.py` & `compute_session_predictivity_h5_cv_pool_align_passive.py`, `compute_model_session_predictivity_h5_cv_align_passive.py`
  - `3_fit_linear_models_cv.sh` & `3_fit_linear_models_cv_passive.sh` bash script to predict neural activity with linear models using task-related variables(use conda environment _DeepProprio_).

To run the same scripts but using passive data, add '_passive' as suffix to the script name (e.g. `run_sessions_75pca_h5_align.sh` --> `run_sessions_75pca_h5_align_passive.sh`)
