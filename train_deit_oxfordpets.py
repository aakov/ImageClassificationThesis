import os
import gc
import argparse
import subprocess
import sys

import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_generator import create_simple_dataset
from data_loader import load_oxford_pets
from experiment_config import (
    DEIT_BACKBONE_NAMES,
    RESULT_FIELDNAMES,
    RESULTS_CSV,
    get_backbone_config,
    get_dataset_config,
    get_training_config,
)
from experiment_utils import (
    append_results_csv,
    ensure_dir,
    time_block,
    reset_gpu_memory_stats,
    get_gpu_memory_mb,
    get_model_config,
    setup_gpu,
    GPUStatsCallback,
    get_gpu_utilization,
    save_model_safe,
    plot_training_results,
    set_global_seed,
    build_result_row,
    create_run_context,
    evaluate_classification_model,
    average_gpu_utilization,
    save_split_manifest,
    save_training_history,
    count_trainable_params,
    save_model_profile,
    summarize_gpu_device_usage,
    checkpoint_exists,
    restore_best_experiment_checkpoint,
)
from deit_tfhub import build_deit_tfhub_model

TRAINING_CONFIG = get_training_config("oxfordpets_deit")
DATASET_CONFIG = get_dataset_config(TRAINING_CONFIG.dataset)
# Options: deit_tiny, deit_small, deit_base, deit_tiny_distilled, deit_small_distilled, deit_base_distilled
BACKBONE = TRAINING_CONFIG.default_backbone


def init_metrics_log(log_path):
    ensure_dir(os.path.dirname(log_path))
    if os.path.exists(log_path):
        return
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,loss,accuracy,val_loss,val_accuracy,"
            "tf_allocator_current_mb,tf_allocator_peak_mb,"
            "gpu_device_used_mb,gpu_device_peak_mb,gpu_device_total_mb,"
            "gpu_device_delta_peak_mb,gpu_util_avg,"
            "epoch_time_sec,total_time_sec\n"
        )


def main(backbone=BACKBONE):
    backbone_config = get_backbone_config(backbone)
    set_global_seed(TRAINING_CONFIG.seed, deterministic=TRAINING_CONFIG.deterministic_ops)
    run_context = create_run_context(TRAINING_CONFIG, DATASET_CONFIG, backbone_config)
    setup_gpu(policy=TRAINING_CONFIG.gpu_policy)

    if not tf.config.list_physical_devices("GPU"):
        print("⚠️ WARNING: GPU not detected. Training on CPU will be extremely slow.")

    class_names, trainval_records, test_records = load_oxford_pets(
        DATASET_CONFIG.images_dir,
        DATASET_CONFIG.annotations_dir,
    )

    labels = [rec.label for rec in trainval_records]
    train_split, val_split = train_test_split(
        trainval_records,
        test_size=DATASET_CONFIG.val_size,
        random_state=TRAINING_CONFIG.seed,
        stratify=labels,
    )
    split_file = save_split_manifest(run_context, train_split, val_split, test_records)

    print(f"\n{'=' * 60}")
    print(f"🚀 Training DeiT Transformer on OxfordPets: {backbone}")
    print(f"Classes: {len(class_names)}")
    print(f"{'=' * 60}")

    timer = time_block()
    reset_gpu_memory_stats()

    run_dir = run_context.models_dir
    checkpoint_dir = run_context.checkpoints_dir
    backup_head_dir = os.path.join(run_context.backups_dir, "head")
    backup_fine_dir = os.path.join(run_context.backups_dir, "fine")
    ensure_dir(checkpoint_dir)
    ensure_dir(backup_head_dir)
    ensure_dir(backup_fine_dir)

    head_best_weights = os.path.join(checkpoint_dir, "head_best.ckpt")
    fine_best_weights = os.path.join(checkpoint_dir, "fine_best.ckpt")

    config = get_model_config(backbone)
    train_ds = create_simple_dataset(
        train_split,
        batch_size=TRAINING_CONFIG.batch_size_head,
        input_size=config["input_size"],
        preprocess_fn=config["preprocess_fn"],
        augment=DATASET_CONFIG.train_augment,
        seed=TRAINING_CONFIG.seed,
    )
    val_ds = create_simple_dataset(
        val_split,
        batch_size=TRAINING_CONFIG.batch_size_head,
        shuffle=False,
        input_size=config["input_size"],
        preprocess_fn=config["preprocess_fn"],
        augment=DATASET_CONFIG.val_augment,
    )
    test_ds = create_simple_dataset(
        test_records,
        batch_size=TRAINING_CONFIG.batch_size_fine,
        shuffle=False,
        input_size=config["input_size"],
        preprocess_fn=config["preprocess_fn"],
        augment=False,
    )

    try:
        model, base_model_layer = build_deit_tfhub_model(
            backbone,
            len(class_names),
            config["input_size"] + (3,),
        )
    except Exception as exc:
        curr_mb, peak_mb = get_gpu_memory_mb()
        append_results_csv(
            RESULTS_CSV,
            build_result_row(
                DATASET_CONFIG,
                TRAINING_CONFIG,
                backbone_config,
                len(class_names),
                len(train_split),
                len(val_split),
                test_size=len(test_records),
                official_test_size=len(test_records),
                eval_split="official_test",
                status="failed",
                error_message=str(exc).replace("\n", " ")[:500],
                run_id=run_context.run_id,
                split_file=split_file,
                gpu_current_mb=curr_mb,
                gpu_peak_mb=peak_mb,
                gpu_utilization=get_gpu_utilization(),
            ),
            RESULT_FIELDNAMES,
        )
        raise
    total_params = model.count_params()
    trainable_params_head = count_trainable_params(model)
    trainable_params_fine = trainable_params_head

    if TRAINING_CONFIG.resume_from_run and checkpoint_exists(head_best_weights):
        try:
            model.load_weights(head_best_weights)
            print(f"Loaded existing head weights from {head_best_weights}")
        except Exception as exc:
            print(f"Could not load existing head weights: {exc}")

    print(f"\n--- Stage 1: Training head ({backbone}) ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TRAINING_CONFIG.lr_head),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=TRAINING_CONFIG.run_eagerly
    )

    head_log = os.path.join(run_context.logs_dir, f"{backbone}_oxfordpets_head_metrics.csv")
    init_metrics_log(head_log)
    head_gpu_callback = GPUStatsCallback(
        sample_interval=TRAINING_CONFIG.gpu_sample_interval,
        log_file=head_log,
    )

    head_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=TRAINING_CONFIG.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=head_best_weights,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.BackupAndRestore(backup_dir=backup_head_dir),
        head_gpu_callback,
    ]

    try:
        history_head = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=TRAINING_CONFIG.epochs_head,
            callbacks=head_callbacks
        )
    except Exception as exc:
        print(f"Training interrupted: {exc}")
        save_model_safe(model, run_dir, f"{backbone}_oxfordpets_head_interrupted")
        curr_mb, peak_mb = get_gpu_memory_mb()
        device_gpu = summarize_gpu_device_usage(head_gpu_callback)
        append_results_csv(
            RESULTS_CSV,
            build_result_row(
                DATASET_CONFIG,
                TRAINING_CONFIG,
                backbone_config,
                len(class_names),
                len(train_split),
                len(val_split),
                official_test_size=len(test_records),
                status="failed",
                error_message=str(exc).replace("\n", " ")[:500],
                run_id=run_context.run_id,
                split_file=split_file,
                gpu_current_mb=curr_mb,
                gpu_peak_mb=peak_mb,
                gpu_device_baseline_mb=device_gpu["baseline_mb"],
                gpu_device_peak_mb=device_gpu["peak_mb"],
                gpu_device_delta_peak_mb=device_gpu["delta_peak_mb"],
                gpu_device_total_mb=device_gpu["total_mb"],
                gpu_utilization=get_gpu_utilization(),
            ),
            RESULT_FIELDNAMES,
        )
        raise

    history_fine = history_head
    fine_log = None
    fine_gpu_callback = None
    if base_model_layer is not None and TRAINING_CONFIG.epochs_fine > 0:
        print(f"\n--- Stage 2: Fine-tuning ({backbone}) ---")
        base_model_layer.trainable = True
        trainable_params_fine = count_trainable_params(model)

        train_ds = create_simple_dataset(
            train_split,
            batch_size=TRAINING_CONFIG.batch_size_fine,
            input_size=config["input_size"],
            preprocess_fn=config["preprocess_fn"],
            augment=DATASET_CONFIG.train_augment,
            seed=TRAINING_CONFIG.seed,
        )
        val_ds = create_simple_dataset(
            val_split,
            batch_size=TRAINING_CONFIG.batch_size_fine,
            shuffle=False,
            input_size=config["input_size"],
            preprocess_fn=config["preprocess_fn"],
            augment=DATASET_CONFIG.val_augment,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=TRAINING_CONFIG.lr_fine),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            run_eagerly=TRAINING_CONFIG.run_eagerly
        )

        if TRAINING_CONFIG.resume_from_run and checkpoint_exists(fine_best_weights):
            try:
                model.load_weights(fine_best_weights)
                print(f"Loaded existing fine-tune weights from {fine_best_weights}")
            except Exception as exc:
                print(f"Could not load fine-tune weights: {exc}")

        fine_log = os.path.join(run_context.logs_dir, f"{backbone}_oxfordpets_fine_metrics.csv")
        init_metrics_log(fine_log)
        fine_gpu_callback = GPUStatsCallback(
            sample_interval=TRAINING_CONFIG.gpu_sample_interval,
            log_file=fine_log,
        )

        fine_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=TRAINING_CONFIG.early_stopping_patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=fine_best_weights,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                save_weights_only=True
            ),
            tf.keras.callbacks.BackupAndRestore(backup_dir=backup_fine_dir),
            fine_gpu_callback,
        ]

        try:
            history_fine = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=TRAINING_CONFIG.epochs_fine,
                callbacks=fine_callbacks
            )
        except Exception as exc:
            print(f"Fine-tuning interrupted: {exc}")
            save_model_safe(model, run_dir, f"{backbone}_oxfordpets_fine_interrupted")
            curr_mb, peak_mb = get_gpu_memory_mb()
            device_gpu = summarize_gpu_device_usage(head_gpu_callback, fine_gpu_callback)
            append_results_csv(
                RESULTS_CSV,
                build_result_row(
                    DATASET_CONFIG,
                    TRAINING_CONFIG,
                    backbone_config,
                    len(class_names),
                    len(train_split),
                    len(val_split),
                    official_test_size=len(test_records),
                    status="failed",
                    error_message=str(exc).replace("\n", " ")[:500],
                    run_id=run_context.run_id,
                    split_file=split_file,
                    gpu_current_mb=curr_mb,
                    gpu_peak_mb=peak_mb,
                    gpu_device_baseline_mb=device_gpu["baseline_mb"],
                    gpu_device_peak_mb=device_gpu["peak_mb"],
                    gpu_device_delta_peak_mb=device_gpu["delta_peak_mb"],
                    gpu_device_total_mb=device_gpu["total_mb"],
                    gpu_utilization=get_gpu_utilization(),
                ),
                RESULT_FIELDNAMES,
            )
            raise

    duration = timer()
    curr_mb, peak_mb = get_gpu_memory_mb()
    util = average_gpu_utilization(head_gpu_callback, fine_gpu_callback)
    device_gpu = summarize_gpu_device_usage(head_gpu_callback, fine_gpu_callback)
    print(f"Total training time for {backbone} (OxfordPets): {duration:.1f}s")

    metrics_log_path = fine_log or head_log
    try:
        plot_training_results(
            history_head,
            history_fine,
            f"{backbone}_oxfordpets",
            duration=duration,
            gpu_peak=device_gpu["peak_mb"],
            output_dir=run_context.plots_dir,
            metrics_log_path=metrics_log_path
        )
    except Exception as exc:
        print(f"Plotting failed: {exc}")

    checkpoint_candidates = [("head", history_head, head_best_weights)]
    if history_fine is not history_head:
        checkpoint_candidates.append(("fine", history_fine, fine_best_weights))
    selected_stage, best_val_acc, _ = restore_best_experiment_checkpoint(
        model,
        checkpoint_candidates,
    )

    epochs_completed_head = len(history_head.history.get("loss", []))
    epochs_completed_fine = (
        len(history_fine.history.get("loss", [])) if history_fine is not history_head else 0
    )
    save_training_history(history_head, history_fine, run_context.logs_dir, backbone)
    save_model_profile(
        model,
        run_context.root_dir,
        backbone,
        total_params,
        trainable_params_head,
        trainable_params_fine,
    )

    save_model_safe(model, run_dir, f"{backbone}_oxfordpets_final")

    validation_metrics = {}
    test_metrics = {}
    test_evaluation_error = ""
    if TRAINING_CONFIG.evaluate_after_training:
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

    result_status = (
        "failed"
        if TRAINING_CONFIG.evaluate_after_training and not test_metrics
        else "completed"
    )
    row = build_result_row(
        DATASET_CONFIG,
        TRAINING_CONFIG,
        backbone_config,
        len(class_names),
        len(train_split),
        len(val_split),
        official_test_size=len(test_records),
        test_size=len(test_records),
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
        gpu_current_mb=curr_mb,
        gpu_peak_mb=peak_mb,
        gpu_device_baseline_mb=device_gpu["baseline_mb"],
        gpu_device_peak_mb=device_gpu["peak_mb"],
        gpu_device_delta_peak_mb=device_gpu["delta_peak_mb"],
        gpu_device_total_mb=device_gpu["total_mb"],
        gpu_utilization=util,
    )
    append_results_csv(RESULTS_CSV, row, RESULT_FIELDNAMES)
    print(f"✅ Backbone {backbone} training finished. Best Val Acc: {best_val_acc:.4f}")

    del model
    if base_model_layer:
        del base_model_layer
    tf.keras.backend.clear_session()
    gc.collect()

    if result_status == "failed":
        raise RuntimeError(f"Official test evaluation failed: {test_evaluation_error}")

    print("\n🏁 OxfordPets DeiT training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeiT on Oxford-IIIT Pet using protocol v3.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--backbone", choices=DEIT_BACKBONE_NAMES, default=BACKBONE)
    selection.add_argument("--all", action="store_true", help="Run all configured DeiT variants sequentially.")
    args = parser.parse_args()

    if not args.all:
        main(args.backbone)
        raise SystemExit(0)

    failures = []
    for selected_backbone in DEIT_BACKBONE_NAMES:
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--backbone", selected_backbone],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=False,
        )
        if completed.returncode != 0:
            failures.append((selected_backbone, completed.returncode))
            print(
                f"Experiment failed for {selected_backbone}: "
                f"exit_code={completed.returncode}"
            )

    if failures:
        names = ", ".join(name for name, _ in failures)
        raise RuntimeError(f"Failed DeiT experiments: {names}")

