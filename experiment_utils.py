import csv
import json
import os
import random
import time
import subprocess
from dataclasses import asdict
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications import resnet50, efficientnet, mobilenet_v2, inception_v3, resnet, convnext
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from experiment_config import BACKBONES, PROTOCOL_VERSION, RUNS_DIR


def plot_training_results(history_head, history_fine, backbone, duration=None, gpu_peak=None, output_dir="outputs/experiments/plots", metrics_log_path=None):
    """
    Generates combined plots for head training and fine-tuning stages.
    """
    ensure_dir(output_dir)
    sns.set_theme(style="whitegrid")

    # Combine histories
    h1 = history_head.history
    h2 = history_fine.history if history_fine and history_fine != history_head else None

    epochs_h1 = len(h1['accuracy'])
    epochs_h2 = len(h2['accuracy']) if h2 else 0
    total_epochs = epochs_h1 + epochs_h2
    x_range = np.arange(1, total_epochs + 1)

    plt.figure(figsize=(15, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    acc = list(h1['accuracy'])
    val_acc = list(h1['val_accuracy'])
    if h2:
        acc = acc + list(h2['accuracy'])
        val_acc = val_acc + list(h2['val_accuracy'])

    plt.plot(x_range, acc, label='Train Accuracy', marker='o', markersize=4)
    plt.plot(x_range, val_acc, label='Val Accuracy', marker='s', markersize=4)

    if h2:
        plt.axvline(x=epochs_h1, color='r', linestyle='--', label='Fine-tuning Start')

    plt.title(f'Accuracy - {backbone}', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Loss
    plt.subplot(1, 2, 2)
    loss = list(h1['loss'])
    val_loss = list(h1['val_loss'])
    if h2:
        loss = loss + list(h2['loss'])
        val_loss = val_loss + list(h2['val_loss'])

    plt.plot(x_range, loss, label='Train Loss', marker='o', markersize=4)
    plt.plot(x_range, val_loss, label='Val Loss', marker='s', markersize=4)

    if h2:
        plt.axvline(x=epochs_h1, color='r', linestyle='--', label='Fine-tuning Start')

    plt.title(f'Loss - {backbone}', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{backbone}_training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📊 Training curves saved to {plot_path}")

    # Plot summary bar chart for training time and peak GPU usage
    plot_session_summary(backbone, total_epochs, duration=duration, gpu_peak=gpu_peak, output_dir=output_dir)

    if metrics_log_path and os.path.exists(metrics_log_path):
        plot_epoch_timing(metrics_log_path, backbone, output_dir=output_dir)


def save_training_history(history_head, history_fine, output_dir, backbone):
    ensure_dir(output_dir)
    history_path = os.path.join(output_dir, f"{backbone}_training_history.csv")
    stages = [("head", history_head)]
    if history_fine is not None and history_fine is not history_head:
        stages.append(("fine", history_fine))

    global_epoch = 0
    with open(history_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "stage",
            "stage_epoch",
            "global_epoch",
            "loss",
            "accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate",
        ])
        for stage_name, history in stages:
            values = history.history
            epoch_count = max((len(items) for items in values.values()), default=0)
            learning_rates = values.get("learning_rate", values.get("lr", []))
            for stage_epoch in range(epoch_count):
                global_epoch += 1

                def metric(name):
                    items = values.get(name, [])
                    return "" if stage_epoch >= len(items) else float(items[stage_epoch])

                learning_rate = ""
                if stage_epoch < len(learning_rates):
                    learning_rate = float(learning_rates[stage_epoch])
                writer.writerow([
                    stage_name,
                    stage_epoch + 1,
                    global_epoch,
                    metric("loss"),
                    metric("accuracy"),
                    metric("val_loss"),
                    metric("val_accuracy"),
                    learning_rate,
                ])
    return history_path


def count_trainable_params(model):
    return int(sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights))


def save_model_profile(
    model,
    output_dir,
    backbone,
    total_params,
    trainable_params_head,
    trainable_params_fine,
):
    ensure_dir(output_dir)
    profile_path = os.path.join(output_dir, f"{backbone}_model_profile.json")
    profile = {
        "model_name": model.name,
        "backbone": backbone,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_params": int(total_params),
        "trainable_params_head": int(trainable_params_head),
        "trainable_params_fine": int(trainable_params_fine),
    }
    with open(profile_path, "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2, ensure_ascii=False)
    return profile_path


def plot_epoch_timing(metrics_log_path, backbone, output_dir="outputs/experiments/plots"):
    epochs = []
    epoch_time = []
    total_time = []

    with open(metrics_log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row.get("epoch", 0)))
                epoch_time.append(float(row.get("epoch_time_sec", 0)))
                total_time.append(float(row.get("total_time_sec", 0)))
            except (TypeError, ValueError):
                continue

    if not epochs:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, epoch_time, label="Epoch Time (s)", marker='o', markersize=4)
    plt.plot(epochs, total_time, label="Total Time (s)", marker='s', markersize=4)
    plt.title(f"Timing - {backbone}", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Seconds", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    timing_path = os.path.join(output_dir, f"{backbone}_timing.png")
    plt.tight_layout()
    plt.savefig(timing_path, dpi=300)
    plt.close()
    print(f"📊 Timing plot saved to {timing_path}")


def plot_session_summary(backbone, total_epochs, duration=None, gpu_peak=None, output_dir="outputs/experiments/plots"):
    """
    Optional summary for non-epoch data like total time or peak memory if provided.
    """
    if duration is None and gpu_peak is None:
        return
        
    plt.figure(figsize=(10, 5))
    categories = []
    values = []
    
    if duration:
        categories.append('Time (s)')
        values.append(duration)
    if gpu_peak:
        categories.append('Device peak VRAM (MB)')
        values.append(gpu_peak)
        
    if categories:
        sns.barplot(x=categories, y=values, hue=categories, palette="viridis", legend=False)
        plt.title(f"Training Summary: {backbone} ({total_epochs} epochs)")
        plt.ylabel("Value")
        summary_path = os.path.join(output_dir, f"{backbone}_summary.png")
        plt.savefig(summary_path, dpi=300)
        plt.close()
        print(f"📊 Session summary saved to {summary_path}")


class RunContext:
    def __init__(self, training_config, dataset_config, backbone_config, run_id=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or f"{timestamp}_{training_config.name}_{backbone_config.name}"
        self.root_dir = os.path.join(RUNS_DIR, self.run_id)
        self.models_dir = os.path.join(self.root_dir, "models")
        self.checkpoints_dir = os.path.join(self.root_dir, "checkpoints")
        self.backups_dir = os.path.join(self.root_dir, "backups")
        self.logs_dir = os.path.join(self.root_dir, "logs")
        self.plots_dir = os.path.join(self.root_dir, "plots")
        self.eval_dir = os.path.join(self.root_dir, "evaluation")

        for path in (
            self.root_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.backups_dir,
            self.logs_dir,
            self.plots_dir,
            self.eval_dir,
        ):
            ensure_dir(path)

        self.write_config(training_config, dataset_config, backbone_config)

    def write_config(self, training_config, dataset_config, backbone_config):
        config_path = os.path.join(self.root_dir, "config.json")
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "run_id": self.run_id,
            "training": asdict(training_config),
            "dataset": asdict(dataset_config),
            "backbone": asdict(backbone_config),
        }
        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)


def create_run_context(training_config, dataset_config, backbone_config, run_id=None):
    return RunContext(training_config, dataset_config, backbone_config, run_id=run_id)


def save_split_manifest(run_context, train_records, val_records, test_records):
    manifest_path = os.path.join(run_context.root_dir, "split_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "sample", "label"])
        for split_name, records in (
            ("train", train_records),
            ("validation", val_records),
            ("official_test", test_records),
        ):
            for record in records:
                sample = getattr(record, "filepath", None) or getattr(record, "filename", "")
                writer.writerow([split_name, sample, record.label])
    return os.path.relpath(manifest_path).replace(os.sep, "/")


def get_model_config(backbone):
    backbone = backbone.lower()
    config = {}

    if backbone in BACKBONES:
        backbone_config = BACKBONES[backbone]
        return {
            "input_size": backbone_config.input_size,
            "preprocess_fn": backbone_config.preprocess_fn,
            "weights": backbone_config.weights,
            "preprocessing": backbone_config.preprocessing,
            "family": backbone_config.family,
        }

    if backbone in ("resnet50", "mobilenetv2"):
        config["input_size"] = (224, 224)
    elif backbone == "resnet101":
        config["input_size"] = (224, 224)
    elif backbone == "efficientnetb0":
        config["input_size"] = (224, 224)
    elif backbone == "efficientnetb1":
        config["input_size"] = (240, 240)
    elif "deit" in backbone:
        config["input_size"] = (224, 224)
    elif backbone == "vit_custom":
        config["input_size"] = (144, 144)
    else:
        # Default fallback
        config["input_size"] = (224, 224)

    if "resnet50" in backbone:
        config["preprocess_fn"] = resnet50.preprocess_input
    elif "resnet101" in backbone:
        config["preprocess_fn"] = resnet.preprocess_input
    elif "efficientnet" in backbone:
        config["preprocess_fn"] = efficientnet.preprocess_input
    elif "mobilenetv2" in backbone:
        config["preprocess_fn"] = mobilenet_v2.preprocess_input
    elif "inceptionv3" in backbone:
        config["preprocess_fn"] = inception_v3.preprocess_input
    elif "deit" in backbone:
        # DeiT models expect ImageNet normalization
        # When using layers.Rescaling(1.0/255.0) in build_deit_tfhub_model,
        # we need to further normalize with mean/std
        # However, many TF Hub ports handle this or expect [0, 1]
        # We will use simple [0, 1] rescale in model, so preprocess_fn here is None
        config["preprocess_fn"] = None
    elif "vit_custom" in backbone:
        config["preprocess_fn"] = lambda x: (x / 127.5) - 1.0
    else:
        config["preprocess_fn"] = None

    return config


def set_global_seed(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if not deterministic:
        return
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def setup_gpu(policy='mixed_float16'):
    # Disable XLA entirely to avoid 'ptxas.exe' requirement
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'
    os.environ['TF_XLA_AUTO_JIT'] = '0'
    os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
    
    # Disable XLA at the API level
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    
    # Point XLA to the directory containing libdevice if found in conda pkgs
    # This helps avoid 'Can't find libdevice directory' warnings/errors
    # For Windows, it might need to be the parent of where nvvm/libdevice is expected
    cuda_path = "C:\\Users\\aakov\\miniconda3\\pkgs\\cudatoolkit-11.2.2-h7d7167e_13\\Library"
    if os.path.exists(cuda_path):
        os.environ['XLA_FLAGS'] += f' --xla_gpu_cuda_data_dir={cuda_path}'
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    # Try to set mixed precision
    if policy:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy(policy)
            print(f"Mixed precision policy set to: {policy}")
        except Exception as e:
            print(f"Failed to set mixed precision policy: {e}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def checkpoint_exists(checkpoint_prefix):
    return os.path.exists(checkpoint_prefix) or os.path.exists(f"{checkpoint_prefix}.index")


def restore_best_experiment_checkpoint(model, candidates):
    """Load the best validation checkpoint across one or more training stages."""
    scored_candidates = []
    for priority, (stage, history, checkpoint_prefix) in enumerate(candidates):
        if history is None:
            continue
        val_accuracy = history.history.get("val_accuracy", [])
        if not val_accuracy:
            raise ValueError(f"Missing val_accuracy history for stage: {stage}")
        best_epoch_index = int(np.argmax(val_accuracy))
        scored_candidates.append(
            (
                float(val_accuracy[best_epoch_index]),
                priority,
                stage,
                best_epoch_index + 1,
                checkpoint_prefix,
            )
        )

    if not scored_candidates:
        raise ValueError("No completed training stage was provided for checkpoint selection.")

    best_val_acc, _, selected_stage, best_epoch, checkpoint_prefix = max(
        scored_candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    if not checkpoint_exists(checkpoint_prefix):
        raise FileNotFoundError(
            f"Best {selected_stage} checkpoint does not exist: {checkpoint_prefix}"
        )

    model.load_weights(checkpoint_prefix)
    print(
        f"Selected {selected_stage}-stage checkpoint from epoch {best_epoch} "
        f"using val_accuracy={best_val_acc:.6f}."
    )
    return selected_stage, best_val_acc, best_epoch


def append_results_csv(csv_path, row, fieldnames):
    ensure_dir(os.path.dirname(csv_path))
    file_exists = os.path.isfile(csv_path)

    if file_exists:
        with open(csv_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            existing_header = next(reader, None)

        if existing_header:
            merged_header = list(fieldnames)
            for key in existing_header:
                if key not in merged_header:
                    merged_header.append(key)

            if merged_header != existing_header:
                with open(csv_path, "r", newline="", encoding="utf-8") as file:
                    dict_reader = csv.DictReader(file)
                    existing_rows = list(dict_reader)

                with open(csv_path, "w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=merged_header)
                    writer.writeheader()
                    for existing_row in existing_rows:
                        writer.writerow(existing_row)
                    writer.writerow(row)
                return

            fieldnames = existing_header

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_result_row(
    dataset_config,
    training_config,
    backbone_config,
    num_classes,
    train_size,
    val_size,
    test_size=0,
    official_test_size=0,
    eval_split="validation",
    eval_accuracy="",
    best_val_acc="",
    test_acc="",
    top5_acc="",
    macro_precision="",
    macro_recall="",
    macro_f1="",
    epochs_head=None,
    epochs_fine=None,
    train_time_sec="",
    status="completed",
    error_message="",
    run_id="",
    gpu_current_mb=None,
    gpu_peak_mb=None,
    gpu_utilization=None,
    split_file="",
    total_params="",
    trainable_params_head="",
    trainable_params_fine="",
    epochs_completed_head="",
    epochs_completed_fine="",
    selected_stage="",
    gpu_device_baseline_mb=None,
    gpu_device_peak_mb=None,
    gpu_device_delta_peak_mb=None,
    gpu_device_total_mb=None,
):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "dataset": dataset_config.name,
        "backbone": backbone_config.name,
        "status": status,
        "error_message": error_message,
        "seed": training_config.seed,
        "num_classes": num_classes,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "official_test_size": official_test_size,
        "official_test_has_labels": dataset_config.official_test_has_labels,
        "eval_split": eval_split,
        "split_strategy": dataset_config.split_strategy,
        "eval_accuracy": eval_accuracy,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "top5_acc": top5_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "total_params": total_params,
        "trainable_params_head": trainable_params_head,
        "trainable_params_fine": trainable_params_fine,
        "epochs_completed_head": epochs_completed_head,
        "epochs_completed_fine": epochs_completed_fine,
        "selected_stage": selected_stage,
        "epochs_head": training_config.epochs_head if epochs_head is None else epochs_head,
        "epochs_fine": training_config.epochs_fine if epochs_fine is None else epochs_fine,
        "batch_size": (
            training_config.batch_size_head
            if training_config.epochs_fine == 0
            else training_config.batch_size_fine
            if training_config.epochs_head == 0
            else f"{training_config.batch_size_head}/{training_config.batch_size_fine}"
        ),
        "batch_size_head": training_config.batch_size_head,
        "batch_size_fine": training_config.batch_size_fine,
        "lr_head": training_config.lr_head,
        "lr_fine": training_config.lr_fine,
        "preprocessing": backbone_config.preprocessing,
        "train_augment": dataset_config.train_augment,
        "val_augment": dataset_config.val_augment,
        "train_time_sec": train_time_sec,
        "weights": backbone_config.weights,
        "split_file": split_file,
        "gpu_current_mb": gpu_current_mb,
        "gpu_peak_mb": gpu_peak_mb,
        "gpu_memory_source": "tensorflow_allocator_and_nvidia_smi_device",
        "gpu_device_baseline_mb": gpu_device_baseline_mb,
        "gpu_device_peak_mb": gpu_device_peak_mb,
        "gpu_device_delta_peak_mb": gpu_device_delta_peak_mb,
        "gpu_device_total_mb": gpu_device_total_mb,
        "gpu_utilization": gpu_utilization,
    }


def time_block():
    start = time.time()

    def end():
        return time.time() - start

    return end


def top_k_accuracy(predictions, true_labels, k=5):
    correct = 0
    for index, label in enumerate(true_labels):
        top_k = predictions[index].argsort()[-k:]
        if label in top_k:
            correct += 1
    return correct / max(len(true_labels), 1)


def evaluate_classification_model(model, dataset, class_names, output_dir, prefix="validation", max_cm_plot_classes=50):
    ensure_dir(output_dir)

    predictions = []
    true_labels = []
    for batch_images, batch_labels in dataset:
        batch_predictions = model.predict(batch_images, verbose=0)
        predictions.append(batch_predictions)
        true_labels.append(batch_labels.numpy())

    if not predictions:
        return {
            "accuracy": "",
            "top5_acc": "",
            "macro_precision": "",
            "macro_recall": "",
            "macro_f1": "",
        }

    y_score = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(true_labels, axis=0).astype(int)
    y_pred = np.argmax(y_score, axis=1)
    labels = np.arange(len(class_names))

    accuracy = float(np.mean(y_pred == y_true))
    top5 = top_k_accuracy(y_score, y_true, k=min(5, len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    predictions_path = os.path.join(output_dir, f"{prefix}_predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "true_label", "true_class", "pred_label", "pred_class", "confidence"])
        for index, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            true_name = class_names[true_label] if true_label < len(class_names) else ""
            pred_name = class_names[pred_label] if pred_label < len(class_names) else ""
            writer.writerow([
                index,
                int(true_label),
                true_name,
                int(pred_label),
                pred_name,
                f"{float(y_score[index, pred_label]):.6f}",
            ])

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.csv")
    with open(cm_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class"] + list(class_names))
        for class_index, row in enumerate(cm):
            writer.writerow([class_names[class_index]] + row.tolist())

    row_sums = cm.sum(axis=1, keepdims=True)
    normalized_cm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )
    normalized_cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix_normalized.csv")
    with open(normalized_cm_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class"] + list(class_names))
        for class_index, row in enumerate(normalized_cm):
            writer.writerow([class_names[class_index]] + [f"{value:.6f}" for value in row])

    per_class_path = os.path.join(output_dir, f"{prefix}_per_class_metrics.csv")
    with open(per_class_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "class_index",
            "class_name",
            "support",
            "correct",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ])
        for class_index, class_name in enumerate(class_names):
            mask = y_true == class_index
            support = int(np.sum(mask))
            correct = int(np.sum(y_pred[mask] == class_index)) if support else 0
            class_accuracy = correct / support if support else 0.0
            writer.writerow([
                class_index,
                class_name,
                int(per_support[class_index]),
                correct,
                f"{class_accuracy:.6f}",
                f"{float(per_precision[class_index]):.6f}",
                f"{float(per_recall[class_index]):.6f}",
                f"{float(per_f1[class_index]):.6f}",
            ])

    if len(class_names) <= max_cm_plot_classes:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, cmap="Blues", cbar=True)
        plt.title(f"Confusion matrix - {prefix}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        cm_plot_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
        plt.savefig(cm_plot_path, dpi=300)
        plt.close()

    metrics = {
        "accuracy": accuracy,
        "top5_acc": top5,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "samples": int(len(y_true)),
        "classes": int(len(class_names)),
    }
    metrics_path = os.path.join(output_dir, f"{prefix}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    return metrics


def reset_gpu_memory_stats(device="GPU:0"):
    try:
        tf.config.experimental.reset_memory_stats(device)
    except Exception:
        pass


def get_gpu_memory_mb(device="GPU:0"):
    try:
        info = tf.config.experimental.get_memory_info(device)
    except Exception:
        return None, None

    current_mb = info.get("current", 0) / (1024 * 1024)
    peak_mb = info.get("peak", 0) / (1024 * 1024)
    return round(current_mb, 1), round(peak_mb, 1)


def get_gpu_device_stats():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        values = [value.strip() for value in output.splitlines()[0].split(",")]
        return {
            "used_mb": float(values[0]),
            "total_mb": float(values[1]),
            "utilization": int(float(values[2])),
            "temperature_c": int(float(values[3])),
        }
    except Exception:
        return None


def get_gpu_utilization():
    stats = get_gpu_device_stats()
    return None if stats is None else stats["utilization"]


class GPUStatsCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file=None, sample_interval=20):
        super().__init__()
        self.log_file = log_file
        self.sample_interval = sample_interval
        self.start_time = None
        self.epoch_start_time = None
        self.util_samples = []
        self.all_util_samples = []
        self.device_used_samples = []
        self.device_epoch_used_samples = []
        self.device_baseline_mb = None
        self.device_peak_mb = None
        self.device_total_mb = None

    def _sample_device(self, record_utilization=True):
        stats = get_gpu_device_stats()
        if stats is None:
            return None

        used_mb = stats["used_mb"]
        self.device_total_mb = stats["total_mb"]
        self.device_used_samples.append(used_mb)
        self.device_epoch_used_samples.append(used_mb)
        self.device_peak_mb = used_mb if self.device_peak_mb is None else max(self.device_peak_mb, used_mb)
        if record_utilization:
            utilization = stats["utilization"]
            self.util_samples.append(utilization)
            self.all_util_samples.append(utilization)
        return stats

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.all_util_samples = []
        self.device_used_samples = []
        self.device_epoch_used_samples = []
        self.device_peak_mb = None
        initial_stats = self._sample_device(record_utilization=False)
        if initial_stats is not None:
            self.device_baseline_mb = initial_stats["used_mb"]
        if self.log_file:
            ensure_dir(os.path.dirname(self.log_file))
            if not os.path.exists(self.log_file):
                with open(self.log_file, "w", encoding="utf-8") as f:
                    f.write(
                        "epoch,loss,accuracy,val_loss,val_accuracy,"
                        "tf_allocator_current_mb,tf_allocator_peak_mb,"
                        "gpu_device_used_mb,gpu_device_peak_mb,gpu_device_total_mb,"
                        "gpu_device_delta_peak_mb,gpu_util_avg,epoch_time_sec,total_time_sec\n"
                    )

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.util_samples = []
        self.device_epoch_used_samples = []

    def on_train_batch_end(self, batch, logs=None):
        if self.sample_interval and (batch % self.sample_interval == 0):
            self._sample_device()

    def on_epoch_end(self, epoch, logs=None):
        curr_mb, peak_mb = get_gpu_memory_mb()
        total_time = time.time() - self.start_time
        epoch_time = time.time() - self.epoch_start_time
        current_device_stats = self._sample_device()
        if self.util_samples:
            util = round(sum(self.util_samples) / len(self.util_samples))
        else:
            util = None

        device_used_mb = None if current_device_stats is None else current_device_stats["used_mb"]
        device_delta_peak_mb = self.device_delta_peak_mb
        stats = f" - tf_allocator: {curr_mb}MB current / {peak_mb}MB peak"
        if device_used_mb is not None:
            stats += (
                f" - device_vram: {device_used_mb:.0f}MB / {self.device_total_mb:.0f}MB"
                f" (run peak {self.device_peak_mb:.0f}MB, delta {device_delta_peak_mb:.0f}MB)"
            )
        if util is not None:
            stats += f" - gpu_util_avg: {util}%"
        stats += f" - epoch_time: {epoch_time:.1f}s - total_time: {total_time:.1f}s"

        print(f"\n[GPU Stats] Epoch {epoch + 1}{stats}")

        if self.log_file:
            loss = logs.get("loss") if logs else None
            acc = logs.get("accuracy") if logs else None
            val_loss = logs.get("val_loss") if logs else None
            val_acc = logs.get("val_accuracy") if logs else None
            with open(self.log_file, "a", encoding="utf-8") as f:
                def format_metric(value):
                    return "" if value is None else f"{float(value):.6f}"

                f.write(
                    f"{epoch + 1},"
                    f"{format_metric(loss)},"
                    f"{format_metric(acc)},"
                    f"{format_metric(val_loss)},"
                    f"{format_metric(val_acc)},"
                    f"{'' if curr_mb is None else curr_mb},"
                    f"{'' if peak_mb is None else peak_mb},"
                    f"{'' if device_used_mb is None else round(device_used_mb, 1)},"
                    f"{'' if self.device_peak_mb is None else round(self.device_peak_mb, 1)},"
                    f"{'' if self.device_total_mb is None else round(self.device_total_mb, 1)},"
                    f"{'' if device_delta_peak_mb is None else round(device_delta_peak_mb, 1)},"
                    f"{'' if util is None else util},"
                    f"{epoch_time:.3f},{total_time:.3f}\n"
                )

    @property
    def average_utilization(self):
        if not self.all_util_samples:
            return None
        return round(sum(self.all_util_samples) / len(self.all_util_samples))

    @property
    def device_delta_peak_mb(self):
        if self.device_baseline_mb is None or self.device_peak_mb is None:
            return None
        return max(0.0, self.device_peak_mb - self.device_baseline_mb)


def average_gpu_utilization(*callbacks):
    samples = []
    for callback in callbacks:
        if callback is not None:
            samples.extend(callback.all_util_samples)
    if not samples:
        return None
    return round(sum(samples) / len(samples))


def summarize_gpu_device_usage(*callbacks):
    callbacks = [callback for callback in callbacks if callback is not None]
    baselines = [
        callback.device_baseline_mb
        for callback in callbacks
        if callback.device_baseline_mb is not None
    ]
    peaks = [
        callback.device_peak_mb
        for callback in callbacks
        if callback.device_peak_mb is not None
    ]
    totals = [
        callback.device_total_mb
        for callback in callbacks
        if callback.device_total_mb is not None
    ]
    baseline_mb = baselines[0] if baselines else None
    peak_mb = max(peaks) if peaks else None
    delta_peak_mb = None
    if baseline_mb is not None and peak_mb is not None:
        delta_peak_mb = max(0.0, peak_mb - baseline_mb)
    return {
        "baseline_mb": baseline_mb,
        "peak_mb": peak_mb,
        "delta_peak_mb": delta_peak_mb,
        "total_mb": totals[-1] if totals else None,
    }


class SimpleLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        parts = []
        for key in ("loss", "accuracy", "val_loss", "val_accuracy"):
            if key in logs:
                try:
                    parts.append(f"{key}={float(logs[key]):.4f}")
                except (TypeError, ValueError):
                    continue
        if parts:
            print(f"Epoch {epoch + 1}: " + " | ".join(parts))


def save_model_safe(model, run_dir, base_name):
    """Save a full .keras model, falling back to weights if serialization fails."""
    ensure_dir(run_dir)

    model_path = os.path.join(run_dir, f"{base_name}.keras")
    try:
        model.save(model_path)
        print(f"Model saved to {model_path}")
        return model_path
    except Exception as exc:
        weights_path = os.path.join(run_dir, f"{base_name}.weights.h5")
        print(f"Model serialization failed: {exc}")
        if os.path.exists(model_path):
            os.remove(model_path)
        print(f"Saving weights instead: {weights_path}")
        model.save_weights(weights_path)
        return weights_path
