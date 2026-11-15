from dataclasses import dataclass


SEED = 42
PROTOCOL_VERSION = "v3"

RESULTS_CSV = "outputs/experiments/results_v3.csv"
RUNS_DIR = "outputs/experiments/runs"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    images_dir: str
    val_size: float
    split_strategy: str
    train_augment: bool
    val_augment: bool
    uses_bbox_crop: bool
    annotations_dir: str = ""
    official_test_dir: str = ""
    official_test_annotations: str = ""
    official_test_has_labels: bool = False


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    family: str
    input_size: tuple
    weights: str
    preprocessing: str
    preprocess_fn: object = None


@dataclass(frozen=True)
class TrainingConfig:
    name: str
    dataset: str
    default_backbone: str
    batch_size_head: int
    batch_size_fine: int
    epochs_head: int
    epochs_fine: int
    lr_head: float
    lr_fine: float
    run_eagerly: bool
    seed: int = SEED
    early_stopping_patience: int = 5
    gpu_policy: str = "float32"
    gpu_sample_interval: int = 20
    # TensorFlow 2.10 on Windows/CUDA 11.2 has no deterministic GPU kernel for
    # UnsortedSegmentSum used by the sparse cross-entropy gradient.  The seed
    # still fixes data splitting, shuffling and initialization.
    deterministic_ops: bool = False
    skip_fine_tune_broadcast_error: bool = False
    resume_from_run: bool = False
    evaluate_after_training: bool = True


DATASETS = {
    "stanford": DatasetConfig(
        name="stanford",
        images_dir="stanford-cars-dataset/versions/1/cars_train/cars_train",
        val_size=0.2,
        split_strategy="stratified_80_20_from_official_train",
        train_augment=True,
        val_augment=False,
        uses_bbox_crop=True,
        official_test_dir="stanford-cars-dataset/versions/1/cars_test/cars_test",
        official_test_annotations=(
            "stanford-cars-dataset/versions/1/car_devkit/devkit/"
            "cars_test_annos_withlabels.mat"
        ),
        official_test_has_labels=True,
    ),
    "oxfordpets": DatasetConfig(
        name="oxfordpets",
        images_dir="oxford-iiit-pet/images/images",
        val_size=0.2,
        split_strategy="stratified_80_20_from_official_trainval_then_official_test",
        train_augment=True,
        val_augment=False,
        uses_bbox_crop=False,
        annotations_dir="oxford-iiit-pet/annotations",
        official_test_dir="oxford-iiit-pet/images/images",
        official_test_annotations="oxford-iiit-pet/annotations/test.txt",
        official_test_has_labels=True,
    ),
}


DEIT_BACKBONE_NAMES = (
    "deit_tiny",
    "deit_small",
    "deit_base",
    "deit_tiny_distilled",
    "deit_small_distilled",
    "deit_base_distilled",
)

SWIN_BACKBONE_NAMES = (
    "swin_tiny",
    "swin_small",
    "swin_base",
)

SWIN_MAIN_BACKBONE_NAMES = (
    "swin_tiny",
    "swin_base",
)


BACKBONES = {
    name: BackboneConfig(
        name=name,
        family="deit",
        input_size=(224, 224),
        weights="tfhub_imagenet",
        preprocessing="decode_resize_then_imagenet_mean_std_inside_model",
    )
    for name in DEIT_BACKBONE_NAMES
}
BACKBONES.update(
    {
        "swin_tiny": BackboneConfig(
            name="swin_tiny",
            family="swin",
            input_size=(224, 224),
            weights="tfhub_imagenet1k",
            preprocessing=(
                "decode_resize_then_imagenet_mean_std_to_"
                "global_average_pooled_features"
            ),
        ),
        "swin_small": BackboneConfig(
            name="swin_small",
            family="swin",
            input_size=(224, 224),
            weights="tfhub_imagenet1k",
            preprocessing=(
                "decode_resize_then_imagenet_mean_std_to_"
                "global_average_pooled_features"
            ),
        ),
        "swin_base": BackboneConfig(
            name="swin_base",
            family="swin",
            input_size=(224, 224),
            weights="tfhub_imagenet22k_then_imagenet1k",
            preprocessing=(
                "decode_resize_then_imagenet_mean_std_to_"
                "global_average_pooled_features"
            ),
        ),
    }
)


TRAINING = {
    "stanford_deit": TrainingConfig(
        name="stanford_deit",
        dataset="stanford",
        default_backbone="deit_tiny",
        batch_size_head=16,
        batch_size_fine=8,
        epochs_head=10,
        epochs_fine=10,
        lr_head=1e-3,
        lr_fine=1e-5,
        run_eagerly=False,
    ),
    "oxfordpets_deit": TrainingConfig(
        name="oxfordpets_deit",
        dataset="oxfordpets",
        default_backbone="deit_base_distilled",
        batch_size_head=16,
        batch_size_fine=8,
        epochs_head=10,
        epochs_fine=10,
        lr_head=1e-3,
        lr_fine=1e-5,
        run_eagerly=False,
    ),
    "stanford_swin": TrainingConfig(
        name="stanford_swin",
        dataset="stanford",
        default_backbone="swin_tiny",
        batch_size_head=16,
        batch_size_fine=8,
        epochs_head=10,
        epochs_fine=10,
        lr_head=1e-3,
        lr_fine=1e-5,
        run_eagerly=False,
    ),
    "oxfordpets_swin": TrainingConfig(
        name="oxfordpets_swin",
        dataset="oxfordpets",
        default_backbone="swin_tiny",
        batch_size_head=16,
        batch_size_fine=8,
        epochs_head=10,
        epochs_fine=10,
        lr_head=1e-3,
        lr_fine=1e-5,
        run_eagerly=False,
    ),
    "stanford_swin_head": TrainingConfig(
        name="stanford_swin_head",
        dataset="stanford",
        default_backbone="swin_tiny",
        batch_size_head=16,
        batch_size_fine=8,
        epochs_head=10,
        epochs_fine=0,
        lr_head=1e-3,
        lr_fine=1e-5,
        run_eagerly=False,
    ),
    "stanford_swin_finetune": TrainingConfig(
        name="stanford_swin_finetune",
        dataset="stanford",
        default_backbone="swin_tiny",
        batch_size_head=16,
        batch_size_fine=8,
        epochs_head=0,
        epochs_fine=10,
        lr_head=1e-3,
        lr_fine=1e-5,
        run_eagerly=False,
        deterministic_ops=False,
        skip_fine_tune_broadcast_error=False,
    ),
}


RESULT_FIELDNAMES = [
    "protocol_version",
    "run_id",
    "dataset",
    "backbone",
    "status",
    "error_message",
    "seed",
    "num_classes",
    "train_size",
    "val_size",
    "test_size",
    "official_test_size",
    "official_test_has_labels",
    "eval_split",
    "split_strategy",
    "eval_accuracy",
    "best_val_acc",
    "test_acc",
    "top5_acc",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "total_params",
    "trainable_params_head",
    "trainable_params_fine",
    "epochs_completed_head",
    "epochs_completed_fine",
    "selected_stage",
    "epochs_head",
    "epochs_fine",
    "batch_size",
    "batch_size_head",
    "batch_size_fine",
    "lr_head",
    "lr_fine",
    "preprocessing",
    "train_augment",
    "val_augment",
    "train_time_sec",
    "weights",
    "split_file",
    "gpu_current_mb",
    "gpu_peak_mb",
    "gpu_memory_source",
    "gpu_device_baseline_mb",
    "gpu_device_peak_mb",
    "gpu_device_delta_peak_mb",
    "gpu_device_total_mb",
    "gpu_utilization",
]


def get_dataset_config(name):
    return DATASETS[name]


def get_backbone_config(name):
    return BACKBONES[name.lower()]


def get_training_config(name):
    return TRAINING[name]
