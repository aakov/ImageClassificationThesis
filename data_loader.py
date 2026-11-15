import os

import scipy.io
from PIL import Image

class CarImage:
    def __init__(self, filename, label, bbox):
        self.filename = filename
        self.filepath = filename
        self.label = label  # Integer label (index of class)
        self.bbox = bbox  # (x1, y1, x2, y2)

    def __repr__(self):
        return f"CarImage(filename={self.filename}, label={self.label}, bbox={self.bbox})"

    def load_image(self):
        return Image.open(self.filepath).convert('RGB')

    def crop_to_bbox(self):
        img = self.load_image()
        x1, y1, x2, y2 = self.bbox
        return img.crop((x1, y1, x2, y2))


class PetImage:
    def __init__(self, filepath, label, breed, species, breed_id):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.label = label
        self.breed = breed
        self.species = species
        self.breed_id = breed_id

# File paths
META_PATH = "stanford-cars-dataset/versions/1/car_devkit/devkit/cars_meta.mat"
TRAIN_ANNOS_PATH = "stanford-cars-dataset/versions/1/car_devkit/devkit/cars_train_annos.mat"
TEST_ANNOS_PATH = "stanford-cars-dataset/versions/1/car_devkit/devkit/cars_test_annos.mat"
TEST_ANNOS_WITH_LABELS_PATH = (
    "stanford-cars-dataset/versions/1/car_devkit/devkit/"
    "cars_test_annos_withlabels.mat"
)


def _load_annotations(mat_path):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Stanford Cars annotation file not found: {mat_path}")
    return scipy.io.loadmat(mat_path)["annotations"][0]


def _matlab_bbox_to_zero_based(ann):
    x1 = int(ann["bbox_x1"][0][0])
    y1 = int(ann["bbox_y1"][0][0])
    x2 = int(ann["bbox_x2"][0][0])
    y2 = int(ann["bbox_y2"][0][0])
    if min(x1, y1) < 1 or x2 < x1 or y2 < y1:
        raise ValueError(f"Invalid MATLAB bounding box: {(x1, y1, x2, y2)}")

    return x1 - 1, y1 - 1, x2, y2


def load_class_names():
    meta = scipy.io.loadmat(META_PATH)
    return [c[0] for c in meta['class_names'][0]]


def load_train_images(mat_path):
    annotations = _load_annotations(mat_path)
    images = []

    for ann in annotations:
        filename = ann['fname'][0]
        label = int(ann['class'][0][0]) - 1  # 0-based indexing
        bbox = _matlab_bbox_to_zero_based(ann)
        images.append(CarImage(filename, label, bbox))

    return images


def load_test_images(mat_path, require_labels=False):
    annotations = _load_annotations(mat_path)
    fields = annotations.dtype.names or ()
    has_labels = "class" in fields
    if require_labels and not has_labels:
        raise ValueError(f"Stanford Cars test annotations have no class labels: {mat_path}")

    images = []

    for ann in annotations:
        filename = ann['fname'][0]
        label = int(ann['class'][0][0]) - 1 if has_labels else None
        bbox = _matlab_bbox_to_zero_based(ann)
        images.append(CarImage(filename, label, bbox))

    return images

_STANFORD_CACHE = None


def load_stanford_cars():
    global _STANFORD_CACHE
    if _STANFORD_CACHE is None:
        _STANFORD_CACHE = (
            load_class_names(),
            load_train_images(TRAIN_ANNOS_PATH),
            load_test_images(TEST_ANNOS_WITH_LABELS_PATH, require_labels=True),
        )
    return _STANFORD_CACHE


def _load_oxford_pets_annotation_file(images_dir, annotation_path):
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(
            f"Missing official Oxford-IIIT Pet annotations: {annotation_path}. "
            "The final protocol requires annotations/trainval.txt and annotations/test.txt."
        )

    records = []
    with open(annotation_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"Invalid Oxford-IIIT Pet annotation at {annotation_path}:{line_number}"
                )

            image_id, class_id, species, breed_id = parts[:4]
            label = int(class_id) - 1
            filepath = os.path.join(images_dir, f"{image_id}.jpg")
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"Oxford-IIIT Pet image not found: {filepath}")

            breed = image_id.rsplit("_", 1)[0]
            records.append(
                PetImage(
                    filepath=filepath,
                    label=label,
                    breed=breed,
                    species=int(species),
                    breed_id=int(breed_id),
                )
            )

    return records


def load_oxford_pets(images_dir, annotations_dir):
    trainval_path = os.path.join(annotations_dir, "trainval.txt")
    test_path = os.path.join(annotations_dir, "test.txt")
    trainval_records = _load_oxford_pets_annotation_file(images_dir, trainval_path)
    test_records = _load_oxford_pets_annotation_file(images_dir, test_path)

    class_name_by_label = {}
    for record in trainval_records + test_records:
        existing_name = class_name_by_label.setdefault(record.label, record.breed)
        if existing_name != record.breed:
            raise ValueError(
                f"Oxford-IIIT Pet class {record.label + 1} maps to both "
                f"{existing_name!r} and {record.breed!r}."
            )

    labels = sorted(class_name_by_label)
    if labels != list(range(len(labels))):
        raise ValueError(f"Oxford-IIIT Pet class IDs are not contiguous: {labels}")

    class_names = [class_name_by_label[label] for label in labels]
    return class_names, trainval_records, test_records


def __getattr__(name):
    if name in {"class_names", "train_images", "test_images"}:
        loaded_class_names, loaded_train_images, loaded_test_images = load_stanford_cars()
        values = {
            "class_names": loaded_class_names,
            "train_images": loaded_train_images,
            "test_images": loaded_test_images,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
