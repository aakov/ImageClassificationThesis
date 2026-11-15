# Code and Results for the Master's Thesis

Comparison of the effectiveness of transformer models in image classification.

## Contents
- train_deit*.py, train_swin*.py, and deit_initiaor.py scripts: model training and batch execution of experiment runs.
- deit_tfhub.py, swin_tfhub.py: model construction and adaptation; tfhub_cache.py: model weight cache management.
- experiment_config.py, data_loader.py, data_generator.py, experiment_utils.py: configuration, data, and shared operations.
- evaluate_models.py: evaluation of saved models on the test or validation set.
- smoke_test_pipeline.py: checks of data and configuration before training.
- outputs/experiments/runs/: original logs, configurations, and test results for the 17 main runs, as well as configurations and head-training logs for two additional runs comparing eager and graph execution modes.
- docs/derived_results_v3.json: run selection and an earlier analysis of the results.
- analiza_wynikow/generate_*.py: scripts for analyzing results and generating tables and plots.
- analiza_wynikow/v5_assets, v6_assets, v7_assets: derived summaries used by successive scripts. These are processed results, not direct training records.

## Reproducing the Analysis Without Training
Run from this package's directory:

    python odtworz_analize.py

Requires NumPy, pandas, and Matplotlib. The command does not run models or download weights. It checks the source results and generates tables and plots in the analiza_wynikow/v5_assets, v6_assets, and v7_assets folders. It overwrites the analysis output files; it does not modify the logs in outputs/experiments/runs. The v5/v6/v7 versions represent stages of thesis preparation, not three separate series of experiments.

## Environment and Running the Models
requirements.txt lists package versions recorded from the local tf environment on September 19, 2026; it is not an archived snapshot of the environment used for each training run. The thesis work used Python 3.10.13, TensorFlow/Keras 2.10, a single RTX 4060 Ti 16 GB GPU, CUDA, and cuDNN. Installing the Python packages alone does not configure GPU drivers or libraries.

Example commands (run from the package directory after preparing the environment and data):

    python smoke_test_pipeline.py --experiment stanford_deit
    python train_deit.py --backbone deit_tiny
    python train_deit_oxfordpets.py --backbone deit_tiny
    python train_swin.py --dataset stanford --backbone swin_tiny
    python evaluate_models.py --model PATH_TO_MODEL.keras --dataset stanford --backbone swin_tiny --split test

The train_* commands start training. Model construction may download weights from TF Hub. Re-evaluation requires a saved model and the images; these large files are not included.

Datasets: Stanford Cars and Oxford-IIIT Pet. The path structure is defined in experiment_config.py and data_loader.py. Stanford Cars requires the images, cars_meta.mat, cars_train_annos.mat, and the labeled test annotations in cars_test_annos_withlabels.mat. Oxford-IIIT Pet requires the images and breed annotations. The data locations must be adjusted in both of the specified modules. The saved split_manifest.csv files document the original splits, while config.json files record the settings for individual runs; paths within these files refer to the original computer.

## Source Results and Timing Correction
The files in runs have been preserved without changes. evaluation/official_test_metrics.json contains the final metrics, and official_test_predictions.csv contains the labels and predicted classes. Per-class metrics and confusion matrices are also included. The head/fine logs contain epoch timing measurements, while training_history contains the training and validation history.

In the analysis of run 20260830_062653_oxfordpets_deit_deit_small_distilled, the duration of the sixth fine-tuning epoch, 21100.035 s, was replaced with the median duration of the remaining epochs, 35.861 s. The raw logs in the package still contain the original measurement. The other times in the tables are sums of epoch durations. The code for this correction is in generate_v5_assets.py and generate_v5_performance.py.
