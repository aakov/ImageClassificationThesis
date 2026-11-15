import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

from tfhub_cache import validate_cached_tfhub_module

TFHUB_MODELS = {
    "swin_tiny": "https://tfhub.dev/sayakpaul/swin_tiny_patch4_window7_224/1",
    "swin_small": "https://tfhub.dev/sayakpaul/swin_small_patch4_window7_224/1",
    "swin_base": "https://tfhub.dev/sayakpaul/swin_base_patch4_window7_224/1",
}


@tf.keras.utils.register_keras_serializable(package="ImageRecognition")
class SwinFeatureBackbone(layers.Layer):
    """Swin trunk ending at the pooled representation before its ImageNet head."""

    def __init__(self, backbone, source_model=None, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        source_model = source_model or _load_source_model(backbone)
        self.projection = source_model.projection
        self.absolute_pos_embed = getattr(source_model, "absolute_pos_embed", None)
        self.position_dropout = source_model.pos_drop
        self.swin_layers = list(source_model.swin_layers)
        self.layer_norm = source_model.norm
        self.feature_dim = int(source_model.head.kernel.shape[0])

    def call(self, inputs, training=None):

        features = self.projection(inputs, training=False)
        if self.absolute_pos_embed is not None:
            features = features + self.absolute_pos_embed
        features = self.position_dropout(features, training=False)
        for swin_layer in self.swin_layers:
            features = swin_layer(features, training=False)
        features = self.layer_norm(features)
        return tf.reduce_mean(features, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"backbone": self.backbone})
        return config


def _load_source_model(backbone):
    if backbone not in TFHUB_MODELS:
        available = ", ".join(TFHUB_MODELS)
        raise ValueError(
            f"Unsupported TF Hub Swin backbone: {backbone}. Available: {available}"
        )

    handle = TFHUB_MODELS[backbone]
    validate_cached_tfhub_module(handle)
    module_path = hub.resolve(handle)
    return tf.keras.models.load_model(module_path, compile=False)


def build_swin_tfhub_model(backbone, num_classes, input_shape):
    source_model = _load_source_model(backbone)

    inputs = layers.Input(shape=input_shape)
    # The converted ImageNet weights expect standard ImageNet channel statistics.
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.Normalization(
        axis=-1,
        mean=[0.485, 0.456, 0.406],
        variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2],
        name="imagenet_normalization",
    )(x)

    base_model = SwinFeatureBackbone(
        backbone,
        source_model=source_model,
        trainable=False,
        name=f"{backbone}_feature_backbone",
    )
    x = base_model(x)
    x = layers.Dropout(0.2, name="target_head_dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="target_head",
    )(x)

    model = tf.keras.Model(inputs, outputs, name=f"{backbone}_transfer_model")
    return model, base_model


def verify_swin_feature_cut(backbone):
    source_model = _load_source_model(backbone)
    feature_backbone = SwinFeatureBackbone(
        backbone,
        source_model=source_model,
        trainable=False,
    )
    sample = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
    source_logits = source_model(sample, training=False)
    features = feature_backbone(sample, training=False)
    reconstructed_logits = source_model.head(features)
    max_abs_diff = float(
        tf.reduce_max(tf.abs(source_logits - reconstructed_logits)).numpy()
    )
    return int(features.shape[-1]), max_abs_diff

