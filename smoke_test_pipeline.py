import argparse
import os

import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from data_generator import create_dataset, create_simple_dataset
from data_loader import load_oxford_pets, load_stanford_cars
from deit_tfhub import build_deit_tfhub_model
from experiment_config import get_backbone_config, get_dataset_config, get_training_config
from experiment_utils import get_model_config, set_global_seed, setup_gpu
from swin_tfhub import build_swin_tfhub_model


def build_split(training_config, dataset_config):
    if dataset_config.name == "stanford":
        class_names, records, test_records = load_stanford_cars()
        train_split, val_split = train_test_split(
            records,
            test_size=dataset_config.val_size,
            random_state=training_config.seed,
            stratify=[record.label for record in records],
        )
        return class_names, train_split, val_split, test_records

    if dataset_config.name == "oxfordpets":
        class_names, records, test_records = load_oxford_pets(
            dataset_config.images_dir,
            dataset_config.annotations_dir,
        )
        train_split, val_split = train_test_split(
            records,
            test_size=dataset_config.val_size,
            random_state=training_config.seed,
            stratify=[record.label for record in records],
        )
        return class_names, train_split, val_split, test_records

    raise ValueError(f"Unsupported dataset: {dataset_config.name}")


def build_dataset(
    training_config,
    dataset_config,
    records,
    backbone_config,
    augment=False,
    fine_tune=False,
    official_test=False,
):
    model_config = get_model_config(backbone_config.name)
    batch_size = training_config.batch_size_fine if fine_tune else training_config.batch_size_head
    sample = records[:batch_size]
    if dataset_config.name == "stanford":
        images_dir = dataset_config.official_test_dir if official_test else dataset_config.images_dir
        return create_dataset(
            sample,
            images_dir,
            batch_size=batch_size,
            shuffle=False,
            input_size=model_config["input_size"],
            preprocess_fn=model_config["preprocess_fn"],
            augment=augment,
        )

    return create_simple_dataset(
        sample,
        batch_size=batch_size,
        shuffle=False,
        input_size=model_config["input_size"],
        preprocess_fn=model_config["preprocess_fn"],
        augment=augment,
    )


def validate_labels(records, num_classes, split_name):
    labels = [record.label for record in records]
    if not labels or any(label is None for label in labels):
        raise ValueError(f"Missing labels in {split_name} split.")
    invalid = [label for label in labels if label < 0 or label >= num_classes]
    if invalid:
        raise ValueError(f"Out-of-range labels in {split_name} split: {invalid[:5]}")


def validate_image_files(dataset_config, records, official_test=False):
    if dataset_config.name == "stanford":
        images_dir = dataset_config.official_test_dir if official_test else dataset_config.images_dir
        for record in records:
            path = os.path.join(images_dir, record.filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing Stanford Cars image: {path}")
            with Image.open(path) as image:
                width, height = image.size
            x1, y1, x2, y2 = record.bbox
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(
                    f"Bounding box {record.bbox} is outside {width}x{height}: {path}"
                )
        return

    for record in records:
        if not os.path.isfile(record.filepath):
            raise FileNotFoundError(f"Missing Oxford-IIIT Pet image: {record.filepath}")
        with Image.open(record.filepath) as image:
            image.verify()


def build_model(backbone_config, num_classes):
    input_shape = backbone_config.input_size + (3,)
    if backbone_config.family == "deit":
        return build_deit_tfhub_model(backbone_config.name, num_classes, input_shape)
    if backbone_config.family == "swin":
        return build_swin_tfhub_model(backbone_config.name, num_classes, input_shape)
    raise ValueError(f"Unsupported backbone family: {backbone_config.family}")


def main():
    parser = argparse.ArgumentParser(description="Smoke test active experiment pipeline without training.")
    parser.add_argument("--experiment", default="stanford_deit")
    parser.add_argument("--backbone", default=None)
    parser.add_argument(
        "--forward-pass",
        action="store_true",
        help="Build the TF Hub model and run one batch through it. May require cached/downloadable TF Hub weights.",
    )
    parser.add_argument(
        "--fine-tune-step",
        action="store_true",
        help="Run one optimizer step with the pretrained base layer trainable. This is a diagnostic, not a full training run.",
    )
    parser.add_argument(
        "--full-data-check",
        action="store_true",
        help="Open every annotated image and validate Stanford Cars bounding boxes.",
    )
    args = parser.parse_args()

    training_config = get_training_config(args.experiment)
    dataset_config = get_dataset_config(training_config.dataset)
    backbone_name = args.backbone or training_config.default_backbone
    backbone_config = get_backbone_config(backbone_name)

    set_global_seed(training_config.seed, deterministic=training_config.deterministic_ops)
    setup_gpu(policy=training_config.gpu_policy)

    class_names, train_split, val_split, test_split = build_split(training_config, dataset_config)
    validate_labels(train_split, len(class_names), "train")
    validate_labels(val_split, len(class_names), "validation")
    validate_labels(test_split, len(class_names), "official test")
    if args.full_data_check:
        validate_image_files(dataset_config, train_split + val_split)
        validate_image_files(dataset_config, test_split, official_test=True)
        print("full_data_check ok")

    records = train_split if args.fine_tune_step else val_split
    dataset = build_dataset(
        training_config,
        dataset_config,
        records,
        backbone_config,
        augment=args.fine_tune_step and dataset_config.train_augment,
        fine_tune=args.fine_tune_step,
    )
    batch_images, batch_labels = next(iter(dataset))
    test_dataset = build_dataset(
        training_config,
        dataset_config,
        test_split,
        backbone_config,
        augment=False,
        fine_tune=args.fine_tune_step,
        official_test=True,
    )
    test_images, test_labels = next(iter(test_dataset))

    print(f"experiment={training_config.name}")
    print(f"dataset={dataset_config.name}")
    print(f"backbone={backbone_config.name}")
    print(
        f"classes={len(class_names)} train={len(train_split)} "
        f"val={len(val_split)} official_test={len(test_split)}"
    )
    print(f"batch_images_shape={batch_images.shape}")
    print(f"batch_labels_shape={batch_labels.shape}")
    print(f"test_batch_images_shape={test_images.shape}")
    print(f"test_batch_labels_shape={test_labels.shape}")

    if args.forward_pass or args.fine_tune_step:
        model, base_model_layer = build_model(backbone_config, len(class_names))
        outputs = model(batch_images, training=False)
        print(f"forward_output_shape={outputs.shape}")

    if args.fine_tune_step:
        if base_model_layer is None:
            raise RuntimeError("Fine-tune diagnostic requires a pretrained base layer.")
        base_model_layer.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=training_config.lr_fine),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            run_eagerly=training_config.run_eagerly,
        )
        model.fit(dataset.take(1), epochs=1, steps_per_epoch=1, verbose=2)
        print("fine_tune_step ok")

    print("smoke test ok")


if __name__ == "__main__":
    main()
