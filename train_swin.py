import argparse
import gc
import os
import subprocess
import sys
from dataclasses import replace

import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_generator import create_dataset, create_simple_dataset
from data_loader import load_oxford_pets, load_stanford_cars
from experiment_config import (
    RESULT_FIELDNAMES,
    RESULTS_CSV,
    SWIN_BACKBONE_NAMES,
    SWIN_MAIN_BACKBONE_NAMES,
    get_backbone_config,
    get_dataset_config,
    get_training_config,
)
from experiment_utils import (
    GPUStatsCallback,
    append_results_csv,
    average_gpu_utilization,
    build_result_row,
    checkpoint_exists,
    count_trainable_params,
    create_run_context,
    evaluate_classification_model,
    get_gpu_memory_mb,
    get_gpu_utilization,
    get_model_config,
    plot_training_results,
    reset_gpu_memory_stats,
    restore_best_experiment_checkpoint,
    save_model_profile,
    save_model_safe,
    save_split_manifest,
    save_training_history,
    set_global_seed,
    setup_gpu,
    summarize_gpu_device_usage,
    time_block,
)
from swin_tfhub import build_swin_tfhub_model


TRAINING_CONFIGS = {
    "stanford": "stanford_swin",
    "oxfordpets": "oxfordpets_swin",
}


def load_records(dataset_config):
    if dataset_config.name == "stanford":
        return load_stanford_cars()
    if dataset_config.name == "oxfordpets":
        return load_oxford_pets(dataset_config.images_dir, dataset_config.annotations_dir)
    raise ValueError(f"Unsupported dataset for Swin: {dataset_config.name}")


def build_dataset(records, dataset_config, model_config, batch_size, augment, seed=None, official_test=False):
    if dataset_config.uses_bbox_crop:
        images_dir = dataset_config.official_test_dir if official_test else dataset_config.images_dir
        return create_dataset(
            records,
            images_dir,
            batch_size=batch_size,
            shuffle=seed is not None,
            input_size=model_config["input_size"],
            preprocess_fn=model_config["preprocess_fn"],
            augment=augment,
            seed=seed,
        )
    return create_simple_dataset(
        records,
        batch_size=batch_size,
        shuffle=seed is not None,
        input_size=model_config["input_size"],
        preprocess_fn=model_config["preprocess_fn"],
        augment=augment,
        seed=seed,
    )


def append_failure(
    training_config,
    dataset_config,
    backbone_config,
    run_context,
    split_file,
    num_classes,
    train_size,
    val_size,
    test_size,
    exc,
    total_params="",
    trainable_params_head="",
    trainable_params_fine="",
    gpu_callbacks=(),
):
    current_mb, peak_mb = get_gpu_memory_mb()
    device_gpu = summarize_gpu_device_usage(*gpu_callbacks)
    append_results_csv(
        RESULTS_CSV,
        build_result_row(
            dataset_config,
            training_config,
            backbone_config,
            num_classes,
            train_size,
            val_size,
            official_test_size=test_size,
            status="failed",
            error_message=str(exc).replace("\n", " ")[:500],
            run_id=run_context.run_id,
            split_file=split_file,
            total_params=total_params,
            trainable_params_head=trainable_params_head,
            trainable_params_fine=trainable_params_fine,
            gpu_current_mb=current_mb,
            gpu_peak_mb=peak_mb,
            gpu_device_baseline_mb=device_gpu["baseline_mb"],
            gpu_device_peak_mb=device_gpu["peak_mb"],
            gpu_device_delta_peak_mb=device_gpu["delta_peak_mb"],
            gpu_device_total_mb=device_gpu["total_mb"],
            gpu_utilization=get_gpu_utilization(),
        ),
        RESULT_FIELDNAMES,
    )


def main(dataset_name="stanford", backbone=None):
    training_config = get_training_config(TRAINING_CONFIGS[dataset_name])
    backbone = backbone or training_config.default_backbone
    if backbone not in SWIN_BACKBONE_NAMES:
        raise ValueError(
            f"Unsupported Swin backbone: {backbone}. "
            f"Available: {', '.join(SWIN_BACKBONE_NAMES)}"
        )
    training_config = replace(training_config, default_backbone=backbone)
    dataset_config = get_dataset_config(training_config.dataset)
    backbone_config = get_backbone_config(backbone)

    set_global_seed(training_config.seed, deterministic=training_config.deterministic_ops)
    run_context = create_run_context(training_config, dataset_config, backbone_config)
    setup_gpu(policy=training_config.gpu_policy)

    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU is required for transformer training, but none was detected.")

    class_names, development_records, test_records = load_records(dataset_config)
    train_split, val_split = train_test_split(
        development_records,
        test_size=dataset_config.val_size,
        random_state=training_config.seed,
        stratify=[record.label for record in development_records],
    )
    split_file = save_split_manifest(run_context, train_split, val_split, test_records)

    print(f"\n{'=' * 60}")
    print(f"Training Swin: dataset={dataset_name}, backbone={backbone}")
    print(f"{'=' * 60}")

    timer = time_block()
    reset_gpu_memory_stats()

    checkpoint_dir = run_context.checkpoints_dir
    backup_head_dir = os.path.join(run_context.backups_dir, "head")
    backup_fine_dir = os.path.join(run_context.backups_dir, "fine")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(backup_head_dir, exist_ok=True)
    os.makedirs(backup_fine_dir, exist_ok=True)

    head_best_weights = os.path.join(checkpoint_dir, "head_best.ckpt")
    fine_best_weights = os.path.join(checkpoint_dir, "fine_best.ckpt")
    head_log = os.path.join(run_context.logs_dir, f"{backbone}_head_metrics.csv")
    fine_log = os.path.join(run_context.logs_dir, f"{backbone}_fine_metrics.csv")

    model_config = get_model_config(backbone)
    train_ds = build_dataset(
        train_split,
        dataset_config,
        model_config,
        training_config.batch_size_head,
        augment=dataset_config.train_augment,
        seed=training_config.seed,
    )
    val_ds = build_dataset(
        val_split,
        dataset_config,
        model_config,
        training_config.batch_size_head,
        augment=False,
    )
    test_ds = build_dataset(
        test_records,
        dataset_config,
        model_config,
        training_config.batch_size_fine,
        augment=False,
        official_test=True,
    )

    try:
        model, base_model_layer = build_swin_tfhub_model(
            backbone,
            len(class_names),
            model_config["input_size"] + (3,),
        )
    except Exception as exc:
        append_failure(
            training_config,
            dataset_config,
            backbone_config,
            run_context,
            split_file,
            len(class_names),
            len(train_split),
            len(val_split),
            len(test_records),
            exc,
        )
        raise
    total_params = model.count_params()
    trainable_params_head = count_trainable_params(model)
    expected_head_params = (base_model_layer.feature_dim + 1) * len(class_names)
    if trainable_params_head != expected_head_params:
        raise RuntimeError(
            f"Unexpected Swin target-head parameter count: "
            f"{trainable_params_head}; expected {expected_head_params}."
        )
    trainable_params_fine = trainable_params_head

    if training_config.resume_from_run and checkpoint_exists(head_best_weights):
        model.load_weights(head_best_weights)
        print(f"Loaded existing head weights from {head_best_weights}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training_config.lr_head),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=training_config.run_eagerly,
    )
    head_gpu_callback = GPUStatsCallback(
        log_file=head_log,
        sample_interval=training_config.gpu_sample_interval,
    )
    head_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=training_config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=head_best_weights,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.BackupAndRestore(backup_dir=backup_head_dir),
        head_gpu_callback,
    ]

    try:
        history_head = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=training_config.epochs_head,
            callbacks=head_callbacks,
        )
    except Exception as exc:
        save_model_safe(model, run_context.models_dir, f"{backbone}_head_interrupted")
        append_failure(
            training_config,
            dataset_config,
            backbone_config,
            run_context,
            split_file,
            len(class_names),
            len(train_split),
            len(val_split),
            len(test_records),
            exc,
            total_params,
            trainable_params_head,
            trainable_params_fine,
            gpu_callbacks=(head_gpu_callback,),
        )
        raise

    base_model_layer.trainable = True
    trainable_params_fine = count_trainable_params(model)
    if trainable_params_fine != total_params:
        raise RuntimeError(
            f"Swin backbone was not fully unfrozen: {trainable_params_fine} "
            f"of {total_params} parameters are trainable."
        )
    train_ds = build_dataset(
        train_split,
        dataset_config,
        model_config,
        training_config.batch_size_fine,
        augment=dataset_config.train_augment,
        seed=training_config.seed,
    )
    val_ds = build_dataset(
        val_split,
        dataset_config,
        model_config,
        training_config.batch_size_fine,
        augment=False,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training_config.lr_fine),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=training_config.run_eagerly,
    )
    if training_config.resume_from_run and checkpoint_exists(fine_best_weights):
        model.load_weights(fine_best_weights)
        print(f"Loaded existing fine-tune weights from {fine_best_weights}")

    fine_gpu_callback = GPUStatsCallback(
        log_file=fine_log,
        sample_interval=training_config.gpu_sample_interval,
    )
    fine_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=training_config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=fine_best_weights,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.BackupAndRestore(backup_dir=backup_fine_dir),
        fine_gpu_callback,
    ]

    try:
        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=training_config.epochs_fine,
            callbacks=fine_callbacks,
        )
    except Exception as exc:
        save_model_safe(model, run_context.models_dir, f"{backbone}_fine_interrupted")
        append_failure(
            training_config,
            dataset_config,
            backbone_config,
            run_context,
            split_file,
            len(class_names),
            len(train_split),
            len(val_split),
            len(test_records),
            exc,
            total_params,
            trainable_params_head,
            trainable_params_fine,
            gpu_callbacks=(head_gpu_callback, fine_gpu_callback),
        )
        raise

    duration = timer()
    current_mb, peak_mb = get_gpu_memory_mb()
    utilization = average_gpu_utilization(head_gpu_callback, fine_gpu_callback)
    device_gpu = summarize_gpu_device_usage(head_gpu_callback, fine_gpu_callback)

    try:
        plot_training_results(
            history_head,
            history_fine,
            f"{backbone}_{dataset_name}",
            duration=duration,
            gpu_peak=device_gpu["peak_mb"],
            output_dir=run_context.plots_dir,
            metrics_log_path=fine_log,
        )
    except Exception as exc:
        print(f"Plotting failed: {exc}")

    selected_stage, best_val_acc, _ = restore_best_experiment_checkpoint(
        model,
        [
            ("head", history_head, head_best_weights),
            ("fine", history_fine, fine_best_weights),
        ],
    )

    epochs_completed_head = len(history_head.history.get("loss", []))
    epochs_completed_fine = len(history_fine.history.get("loss", []))
    save_training_history(
        history_head,
        history_fine,
        run_context.logs_dir,
        f"{backbone}_{dataset_name}",
    )
    save_model_profile(
        model,
        run_context.root_dir,
        backbone,
        total_params,
        trainable_params_head,
        trainable_params_fine,
    )
    save_model_safe(model, run_context.models_dir, f"{backbone}_{dataset_name}_final")

    validation_metrics = {}
    test_metrics = {}
    test_evaluation_error = ""
    if training_config.evaluate_after_training:
        try:
            validation_metrics = evaluate_classification_model(
                model,
                val_ds,
                class_names,
                run_context.eval_dir,
                prefix="validation",
            )
        except Exception as exc:
            print(f"Validation evaluation failed: {exc}")
        try:
            test_metrics = evaluate_classification_model(
                model,
                test_ds,
                class_names,
                run_context.eval_dir,
                prefix="official_test",
            )
        except Exception as exc:
            print(f"Official test evaluation failed: {exc}")
            test_evaluation_error = str(exc).replace("\n", " ")[:500]

    result_status = "failed" if training_config.evaluate_after_training and not test_metrics else "completed"
    row = build_result_row(
        dataset_config,
        training_config,
        backbone_config,
        len(class_names),
        len(train_split),
        len(val_split),
        test_size=len(test_records),
        official_test_size=len(test_records),
        eval_split="official_test",
        eval_accuracy="" if not test_metrics else f"{test_metrics['accuracy']:.4f}",
        best_val_acc=f"{best_val_acc:.4f}",
        test_acc="" if not test_metrics else f"{test_metrics['accuracy']:.4f}",
        top5_acc="" if not test_metrics else f"{test_metrics['top5_acc']:.4f}",
        macro_precision="" if not test_metrics else f"{test_metrics['macro_precision']:.4f}",
        macro_recall="" if not test_metrics else f"{test_metrics['macro_recall']:.4f}",
        macro_f1="" if not test_metrics else f"{test_metrics['macro_f1']:.4f}",
        total_params=total_params,
        trainable_params_head=trainable_params_head,
        trainable_params_fine=trainable_params_fine,
        epochs_completed_head=epochs_completed_head,
        epochs_completed_fine=epochs_completed_fine,
        selected_stage=selected_stage,
        train_time_sec=f"{duration:.1f}",
        status=result_status,
        error_message=test_evaluation_error,
        run_id=run_context.run_id,
        split_file=split_file,
        gpu_current_mb=current_mb,
        gpu_peak_mb=peak_mb,
        gpu_device_baseline_mb=device_gpu["baseline_mb"],
        gpu_device_peak_mb=device_gpu["peak_mb"],
        gpu_device_delta_peak_mb=device_gpu["delta_peak_mb"],
        gpu_device_total_mb=device_gpu["total_mb"],
        gpu_utilization=utilization,
    )
    append_results_csv(RESULTS_CSV, row, RESULT_FIELDNAMES)

    del model
    del base_model_layer
    tf.keras.backend.clear_session()
    gc.collect()

    if result_status == "failed":
        raise RuntimeError(f"Official test evaluation failed: {test_evaluation_error}")
    print(
        f"Swin completed: dataset={dataset_name}, "
        f"best_val_acc={best_val_acc:.4f}, test_acc={test_metrics['accuracy']:.4f}"
    )


def cli():
    parser = argparse.ArgumentParser(
        description="Run Swin-Tiny and Swin-Base experiments using protocol v3."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dataset", choices=tuple(TRAINING_CONFIGS), default="stanford")
    selection.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run Stanford Cars and Oxford-IIIT Pet sequentially.",
    )
    parser.add_argument(
        "--backbone",
        choices=("all",) + SWIN_MAIN_BACKBONE_NAMES,
        default="all",
        help="Backbone scope. The default runs Swin-Tiny and Swin-Base.",
    )
    args = parser.parse_args()

    datasets = tuple(TRAINING_CONFIGS) if args.all_datasets else (args.dataset,)
    backbones = (
        SWIN_MAIN_BACKBONE_NAMES if args.backbone == "all" else (args.backbone,)
    )
    planned_runs = [
        (dataset_name, backbone_name)
        for dataset_name in datasets
        for backbone_name in backbones
    ]

    print("Planned Swin experiments:")
    for dataset_name, backbone_name in planned_runs:
        print(f"  dataset={dataset_name}, backbone={backbone_name}")

    if len(planned_runs) == 1:
        dataset_name, backbone_name = planned_runs[0]
        main(dataset_name, backbone_name)
        return

    failures = []
    for dataset_name, backbone_name in planned_runs:
        completed = subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--dataset",
                dataset_name,
                "--backbone",
                backbone_name,
            ],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=False,
        )
        if completed.returncode != 0:
            failures.append((dataset_name, backbone_name, completed.returncode))
            print(
                f"Swin experiment failed for {dataset_name}/{backbone_name}: "
                f"exit_code={completed.returncode}"
            )

    if failures:
        names = ", ".join(
            f"{dataset_name}/{backbone_name}"
            for dataset_name, backbone_name, _ in failures
        )
        raise RuntimeError(f"Failed Swin experiments: {names}")


if __name__ == "__main__":
    cli()
