import argparse
import json
import os
from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_generator import create_dataset, create_simple_dataset
from data_loader import load_oxford_pets, load_stanford_cars
from deit_tfhub import HubWrapper as DeiTHubWrapper
from experiment_config import BACKBONES, DATASETS, RUNS_DIR, SEED
from experiment_utils import (
    ensure_dir,
    evaluate_classification_model,
    get_model_config,
    set_global_seed,
    setup_gpu,
)
from swin_tfhub import SwinFeatureBackbone


def _load_run_config(model_path):
    path = Path(model_path).resolve()
    for parent in [path.parent, *path.parents]:
        config_path = parent / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as file:
                return json.load(file)
    return {}


def _infer_backbone(model_path, run_config):
    backbone = run_config.get("backbone", {}).get("name")
    if backbone:
        return backbone

    normalized = str(model_path).lower()
    for name in sorted(BACKBONES, key=len, reverse=True):
        if name in normalized:
            return name
    return None


def _infer_dataset(model_path, run_config):
    dataset = run_config.get("dataset", {}).get("name")
    if dataset:
        return dataset

    normalized = str(model_path).lower()
    for name in DATASETS:
        if name in normalized:
            return name
    return None


def _custom_objects_for_backbone(backbone):
    family = BACKBONES[backbone].family
    if family == "deit":
        return {"HubWrapper": DeiTHubWrapper}
    if family == "swin":
        return {
            "SwinFeatureBackbone": SwinFeatureBackbone,
            "ImageRecognition>SwinFeatureBackbone": SwinFeatureBackbone,
        }
    return {}


def _load_model(model_path, backbone):
    custom_objects = _custom_objects_for_backbone(backbone)
    return tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
    )


def _stanford_dataset(split_name, input_size, preprocess_fn, batch_size, seed):
    dataset_config = DATASETS["stanford"]
    class_names, train_images, test_images = load_stanford_cars()
    if split_name == "validation":
        labels = [image.label for image in train_images]
        _, records = train_test_split(
            train_images,
            test_size=dataset_config.val_size,
            random_state=seed,
            stratify=labels,
        )
        images_dir = dataset_config.images_dir
    else:
        records = test_images
        images_dir = dataset_config.official_test_dir

    dataset = create_dataset(
        records,
        images_dir,
        batch_size=batch_size,
        shuffle=False,
        input_size=input_size,
        preprocess_fn=preprocess_fn,
        augment=False,
    )
    return dataset, class_names


def _oxfordpets_dataset(split_name, input_size, preprocess_fn, batch_size, seed):
    dataset_config = DATASETS["oxfordpets"]
    class_names, trainval_records, test_records = load_oxford_pets(
        dataset_config.images_dir,
        dataset_config.annotations_dir,
    )
    if split_name == "validation":
        labels = [record.label for record in trainval_records]
        _, records = train_test_split(
            trainval_records,
            test_size=dataset_config.val_size,
            random_state=seed,
            stratify=labels,
        )
    else:
        records = test_records

    dataset = create_simple_dataset(
        records,
        batch_size=batch_size,
        shuffle=False,
        input_size=input_size,
        preprocess_fn=preprocess_fn,
        augment=False,
    )
    return dataset, class_names


def _evaluation_dataset(dataset_name, split_name, input_size, preprocess_fn, batch_size, seed):
    if dataset_name == "stanford":
        return _stanford_dataset(split_name, input_size, preprocess_fn, batch_size, seed)
    if dataset_name == "oxfordpets":
        return _oxfordpets_dataset(split_name, input_size, preprocess_fn, batch_size, seed)
    raise ValueError(f"Unsupported active dataset: {dataset_name}")


def _default_output_dir(model_path):
    stem = Path(model_path).stem
    safe_stem = "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)
    return os.path.join(RUNS_DIR, "manual_evaluation", safe_stem)


def evaluate_model(model_path, backbone, dataset_name, split_name, batch_size, output_dir):
    run_config = _load_run_config(model_path)
    backbone = backbone or _infer_backbone(model_path, run_config)
    dataset_name = dataset_name or _infer_dataset(model_path, run_config)

    if backbone not in BACKBONES:
        raise ValueError(
            "Could not determine active transformer backbone. "
            f"Use --backbone with one of: {', '.join(BACKBONES)}"
        )
    if dataset_name not in DATASETS:
        raise ValueError(
            "Could not determine active dataset. "
            f"Use --dataset with one of: {', '.join(DATASETS)}"
        )

    config = get_model_config(backbone)
    training_config = run_config.get("training", {})
    seed = int(training_config.get("seed", SEED))
    set_global_seed(seed)
    setup_gpu(policy=training_config.get("gpu_policy", "float32"))
    output_dir = output_dir or _default_output_dir(model_path)
    ensure_dir(output_dir)

    print(f"Evaluating model: {model_path}")
    print(f"Backbone: {backbone}")
    print(f"Dataset split: {dataset_name} {split_name}")
    print(f"Output directory: {output_dir}")

    model = _load_model(model_path, backbone)
    dataset, class_names = _evaluation_dataset(
        dataset_name,
        split_name,
        config["input_size"],
        config["preprocess_fn"],
        batch_size,
        seed,
    )
    prefix = "official_test" if split_name == "test" else "validation"
    metrics = evaluate_classification_model(
        model,
        dataset,
        class_names,
        output_dir,
        prefix=prefix,
    )

    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained v3 transformer model on its matching validation or official test split. "
            "Legacy train_transformers/Patches/PatchEncoder models are not part of this protocol."
        )
    )
    parser.add_argument("--model", required=True, help="Path to a saved .keras model")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), help="Backbone name; inferred from run config when possible")
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="Dataset name; inferred from run config when possible")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", help="Directory for predictions, confusion matrix and per-class metrics")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        backbone=args.backbone,
        dataset_name=args.dataset,
        split_name=args.split,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
