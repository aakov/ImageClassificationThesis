import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

from tfhub_cache import validate_cached_tfhub_module

# Models by sayakpaul (Ported from original PyTorch checkpoints)
TFHUB_MODELS = {
    "deit_tiny": "https://tfhub.dev/sayakpaul/deit_tiny_patch16_224/1",
    "deit_small": "https://tfhub.dev/sayakpaul/deit_small_patch16_224/1",
    "deit_base": "https://tfhub.dev/sayakpaul/deit_base_patch16_224/1",
    "deit_tiny_distilled": "https://tfhub.dev/sayakpaul/deit_tiny_distilled_patch16_224/1",
    "deit_small_distilled": "https://tfhub.dev/sayakpaul/deit_small_distilled_patch16_224/1",
    "deit_base_distilled": "https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_224/1",
}


class HubWrapper(layers.Layer):
    """
    Wraps a TF Hub layer to handle the training argument correctly
    and allow for better integration with Keras Models.
    """
    def __init__(self, handle, name_prefix, **kwargs):
        super().__init__(**kwargs)
        self.handle = handle
        self.name_prefix = name_prefix
        # hub.KerasLayer can sometimes behave unexpectedly with the training argument
        self.hub_layer = hub.KerasLayer(handle, trainable=kwargs.get("trainable", False), name=f"{name_prefix}_hub_inner")

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        
        # DeiT models from Sayak Paul return a list/tuple or dict
        # We need to ensure we handle the output correctly
        try:
            return self.hub_layer(inputs, training=training)
        except Exception:
            # Fallback if training=True is not supported by the specific SavedModel version
            return self.hub_layer(inputs, training=False)

    def get_config(self):
        config = super().get_config()
        config.update({"handle": self.handle, "name_prefix": self.name_prefix})
        return config


def build_deit_tfhub_model(backbone, num_classes, input_shape=(224, 224, 3)):

    if backbone not in TFHUB_MODELS:
        raise ValueError(f"Unsupported TF Hub DeiT backbone: {backbone}. Available: {list(TFHUB_MODELS.keys())}")

    validate_cached_tfhub_module(TFHUB_MODELS[backbone])

    inputs = layers.Input(shape=input_shape)
    
    # The converted ImageNet weights expect standard ImageNet channel statistics.
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.Normalization(
        axis=-1,
        mean=[0.485, 0.456, 0.406],
        variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2],
        name="imagenet_normalization",
    )(x)

    base_model = HubWrapper(TFHUB_MODELS[backbone], name_prefix=backbone, trainable=False, name=f"{backbone}_hub")
    x = base_model(x)
    
    # Handle different output formats from TF Hub
    if isinstance(x, (list, tuple)):
        # Usually [logits] or [cls_token, distillation_token]
        x = x[0]
    elif isinstance(x, dict):
        x = x.get("logits", next(iter(x.values())))

    # Dropout for regularization
    x = layers.Dropout(0.2)(x)
    
    # Classification head
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="head")(x)

    model = tf.keras.Model(inputs, outputs, name=f"deit_model_{backbone}")
    return model, base_model
