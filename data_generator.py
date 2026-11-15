import os
import tensorflow as tf


def preprocess_image_tf(image_bytes, bbox, input_size=(224, 224), preprocess_fn=None, augment=False):
    # Decode JPEG
    image = tf.image.decode_jpeg(image_bytes, channels=3)

    # Crop to bounding box
    x1, y1, x2, y2 = tf.unstack(tf.cast(bbox, tf.int32))
    image = tf.image.crop_to_bounding_box(image, y1, x1, y2 - y1, x2 - x1)

    # Resize
    image = tf.image.resize(image, [input_size[0], input_size[1]])

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2 * 255.0)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.clip_by_value(image, 0.0, 255.0)

    # Model-specific preprocessing
    if preprocess_fn is not None:
        image = preprocess_fn(image)

    return image


def create_dataset(
    car_images,
    images_dir,
    batch_size,
    shuffle=True,
    input_size=(224, 224),
    preprocess_fn=None,
    augment=False,
    seed=None,
):
    # Prepare file paths & labels as tensors
    filepaths = [os.path.join(images_dir, ci.filename) for ci in car_images]
    bboxes = [ci.bbox for ci in car_images]
    labels = [ci.label for ci in car_images]

    ds = tf.data.Dataset.from_tensor_slices((filepaths, bboxes, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=batch_size * 10, seed=seed, reshuffle_each_iteration=True)

    # Load & preprocess
    def load_and_preprocess(path, bbox, label):
        image_bytes = tf.io.read_file(path)
        image = preprocess_image_tf(
            image_bytes,
            bbox,
            input_size=input_size,
            preprocess_fn=preprocess_fn,
            augment=augment,
        )
        return image, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch & prefetch
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def preprocess_image_simple(image_bytes, input_size=(224, 224), preprocess_fn=None, augment=False):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [input_size[0], input_size[1]])

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2 * 255.0)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.clip_by_value(image, 0.0, 255.0)

    if preprocess_fn is not None:
        image = preprocess_fn(image)

    return image


def create_simple_dataset(
    image_records,
    batch_size,
    shuffle=True,
    input_size=(224, 224),
    preprocess_fn=None,
    augment=False,
    seed=None,
):
    filepaths = [rec.filepath for rec in image_records]
    labels = [rec.label for rec in image_records]

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=batch_size * 10, seed=seed, reshuffle_each_iteration=True)

    def load_and_preprocess(path, label):
        image_bytes = tf.io.read_file(path)
        image = preprocess_image_simple(image_bytes, input_size=input_size, preprocess_fn=preprocess_fn, augment=augment)
        return image, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
