"""
Model factories used by Days 3-5.

* ``build_baseline_cnn``  — Day 3 custom CNN (4 conv blocks + GAP + Dense head).
* ``build_mobilenetv2``   — Day 4 transfer-learning model.
* ``build_vgg16``         — Day 5 transfer-learning model.

All three return a compiled ``keras.Model`` and accept the same keyword API so
the training loop in ``src.training`` is model-agnostic.
"""
from __future__ import annotations

from typing import Literal

from src.config import IMG_CHANNELS, IMG_SIZE, NUM_CLASSES


InputSpec = tuple[int, int, int]


def _input_shape(img_size: int = IMG_SIZE, channels: int = IMG_CHANNELS) -> InputSpec:
    return (img_size, img_size, channels)


def _augmentation_layers(seed: int = 42):
    """Return the canonical light-augmentation block used by every training model.

    These layers are added at the head of every model and naturally short-circuit
    to identity at inference time (``training=False``), so the same .keras file
    serves training, evaluation, and Grad-CAM. Doing augmentation inside the
    model graph (instead of inside the PyDataset) avoids a CPU<->GPU round-trip
    on every batch and gives a ~10x speedup on the RTX 5080.
    """
    import keras
    return keras.Sequential([
        keras.layers.RandomRotation(factor=0.03, seed=seed),
        keras.layers.RandomTranslation(0.05, 0.05, seed=seed),
        keras.layers.RandomZoom(0.05, seed=seed),
    ], name="augment")


def build_baseline_cnn(num_classes: int = NUM_CLASSES,
                       input_shape: InputSpec | None = None,
                       dropout: float = 0.3,
                       learning_rate: float = 1e-3,
                       augment: bool = True):
    """Day 3 custom CNN trained from scratch.

    Architecture (matches Day 3, Step 3.1 of the project plan):
        Conv(32) -> BN -> ReLU -> MaxPool
        Conv(64) -> BN -> ReLU -> MaxPool
        Conv(128) -> BN -> ReLU -> MaxPool
        Conv(256) -> BN -> ReLU -> MaxPool
        GlobalAveragePooling
        Dense(128) -> Dropout
        Dense(num_classes, softmax)
    """
    import keras
    from keras import layers, Model

    inp = layers.Input(shape=input_shape or _input_shape(), name="image")

    x = _augmentation_layers()(inp) if augment else inp
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, kernel_size=3, padding="same", use_bias=False,
                          kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool2D(pool_size=2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax", name="probs")(x)

    model = Model(inputs=inp, outputs=out, name="baseline_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _backbone_head(backbone, num_classes: int, dropout: float, head: Literal["gap", "flatten"]):
    """Add the same Dense head on top of any frozen-or-not backbone."""
    import keras
    from keras import layers, Model

    inp = backbone.input
    x = backbone.output
    if head == "gap":
        x = layers.GlobalAveragePooling2D(name="gap")(x)
    else:
        x = layers.Flatten(name="flat")(x)
        x = layers.Dense(256, activation="relu",
                         kernel_initializer="he_normal", name="dense_256")(x)
        x = layers.Dropout(dropout, name="dropout_dense")(x)
    if head == "gap":
        x = layers.Dropout(dropout, name="dropout_gap")(x)
    out = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return Model(inputs=inp, outputs=out, name=backbone.name + "_classifier")


def build_mobilenetv2(num_classes: int = NUM_CLASSES,
                      input_shape: InputSpec | None = None,
                      dropout: float = 0.3,
                      learning_rate: float = 1e-3,
                      freeze_backbone: bool = True,
                      augment: bool = True):
    """Day 4 MobileNetV2 transfer-learning model.

    The model expects inputs in [0, 1] (matches our pipeline) and applies
    MobileNetV2 preprocessing (``[0,1] -> [-1,1]``) inside the graph using a
    ``Rescaling`` layer rather than a ``Lambda(preprocess_input)``. The Lambda
    pattern serializes the function reference, which breaks ``model.save`` /
    ``load_model`` across kernel restarts; ``Rescaling`` is just a layer with
    standard ops and round-trips cleanly.
    """
    import keras
    from keras import layers, Model
    from keras.applications import MobileNetV2

    shape = input_shape or _input_shape()

    raw_input = layers.Input(shape=shape, name="image")
    pre = _augmentation_layers()(raw_input) if augment else raw_input
    # MobileNetV2's preprocess_input is exactly: x_pixels/127.5 - 1, i.e. [0,255] -> [-1,1].
    # Our pipeline already gives [0,1], so the equivalent is x*2 - 1.
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_preprocess")(pre)
    backbone = MobileNetV2(include_top=False, weights="imagenet",
                           input_tensor=x, input_shape=shape)
    backbone.trainable = not freeze_backbone

    h = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    h = layers.Dropout(dropout, name="dropout")(h)
    out = layers.Dense(num_classes, activation="softmax", name="probs")(h)

    model = Model(inputs=raw_input, outputs=out, name="mobilenetv2_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


_HEAD_LAYER_NAMES = ("image", "mobilenet_preprocess",
                     "vgg16_to_pixels", "vgg16_rgb_to_bgr", "vgg16_mean_sub",
                     "gap", "dropout", "dropout_gap", "dropout_dense",
                     "dense_256", "probs")


def unfreeze_top_layers(model, n_layers: int, learning_rate: float = 1e-5,
                        backbone_substring: str = "mobilenetv2",
                        head_layer_names: tuple[str, ...] = _HEAD_LAYER_NAMES):
    """Unfreeze the last ``n_layers`` of the backbone for fine-tuning.

    Two layouts are supported:

    1. **Embedded submodel layout.** Some TF / Keras versions wrap the
       pretrained model as a single layer with sub-layers (``hasattr(layer,
       "layers")``). We locate it by ``backbone_substring`` and unfreeze its
       last ``n_layers``.
    2. **Inlined layout (TF 2.16+).** When ``MobileNetV2(input_tensor=x, ...)``
       is constructed, the backbone's layers get added directly into the
       parent model. There is no submodel to recurse into. We treat every
       top-level layer whose name is **not** in ``head_layer_names`` as a
       backbone layer, then unfreeze the last ``n_layers`` of those.

    Re-compiles the model with a fresh Adam optimizer at ``learning_rate``.
    Raises ``RuntimeError`` if neither path finds layers to unfreeze (better
    than silently doing nothing).
    """
    import keras

    # Path 1 — embedded submodel.
    def _find_backbone(m):
        for layer in m.layers:
            if backbone_substring.lower() in layer.name.lower() and hasattr(layer, "layers"):
                return layer
            if hasattr(layer, "layers") and layer.layers:
                inner = _find_backbone(layer)
                if inner is not None:
                    return inner
        return None

    backbone = _find_backbone(model)

    if backbone is not None:
        for l in backbone.layers:
            l.trainable = False
        for l in backbone.layers[-n_layers:]:
            l.trainable = True
        backbone.trainable = True
    else:
        # Path 2 — inlined backbone. Strip the head layers from the tail.
        backbone_layers = [l for l in model.layers if l.name not in head_layer_names]
        if not backbone_layers:
            raise RuntimeError(
                f"unfreeze_top_layers: could not identify backbone layers in "
                f"model '{model.name}'. head_layer_names={head_layer_names}. "
                f"Top-level layers: {[l.name for l in model.layers]}"
            )
        for l in backbone_layers:
            l.trainable = False
        to_unfreeze = backbone_layers[-n_layers:]
        for l in to_unfreeze:
            l.trainable = True
        if not to_unfreeze:
            raise RuntimeError(
                f"unfreeze_top_layers: requested n_layers={n_layers} but the "
                f"backbone-layer slice is empty after filtering."
            )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_efficientnet_b0(num_classes: int = NUM_CLASSES,
                          input_shape: InputSpec | None = None,
                          dropout: float = 0.3,
                          learning_rate: float = 1e-3,
                          weight_decay: float = 1e-4,
                          freeze_backbone: bool = True,
                          augment: bool = True,
                          loss: str = "categorical_crossentropy",
                          label_smoothing: float = 0.0):
    """Phase D EfficientNet-B0 with the refined recipe.

    Defaults to **224x224 input** to match the pretrained backbone's native
    resolution (Tan & Le 2019, arXiv:1905.11946). The factory accepts:

    * ``loss``: ``"categorical_crossentropy"`` (default; supports one-hot
      targets and ``label_smoothing``) or ``"focal_categorical"`` (focal loss
      from Lin et al. ICCV 2017, arXiv:1708.02002 — useful when the dataset
      is severely imbalanced; off by default since Alzheimer MRI here is
      only mildly imbalanced).
    * ``label_smoothing``: epsilon for softening targets. Set to 0.0 when
      using Mixup in the dataset to avoid double-smoothing (Phase C3 of
      ``refine.md``).
    * ``weight_decay``: AdamW decoupled weight decay
      (Loshchilov & Hutter ICLR 2019, arXiv:1711.05101).

    Preprocessing follows the pattern set in ``build_mobilenetv2`` and
    ``build_vgg16``: a layer-only chain so save/load round-trips cleanly.
    EfficientNet's ``preprocess_input`` is the identity (model expects
    ``[0, 255]``), so we just rescale.
    """
    import keras
    from keras import layers, Model
    from keras.applications import EfficientNetB0

    shape = input_shape or (224, 224, IMG_CHANNELS)
    raw_input = layers.Input(shape=shape, name="image")
    pre = _augmentation_layers()(raw_input) if augment else raw_input
    # EfficientNet expects pixel values in [0, 255]; we feed [0, 1] so rescale.
    x = layers.Rescaling(scale=255.0, name="effnet_to_pixels")(pre)
    backbone = EfficientNetB0(include_top=False, weights="imagenet",
                              input_tensor=x, input_shape=shape)
    backbone.trainable = not freeze_backbone

    h = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    h = layers.Dropout(dropout, name="dropout")(h)
    out = layers.Dense(num_classes, activation="softmax", name="probs")(h)

    model = Model(inputs=raw_input, outputs=out, name="efficientnet_b0_classifier")

    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate,
                                       weight_decay=weight_decay)

    if loss == "categorical_crossentropy":
        loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    elif loss == "focal_categorical":
        loss_fn = keras.losses.CategoricalFocalCrossentropy(
            alpha=0.25, gamma=2.0, label_smoothing=label_smoothing)
    elif loss == "sparse_categorical_crossentropy":
        loss_fn = "sparse_categorical_crossentropy"
    else:
        raise ValueError(f"Unsupported loss '{loss}'")

    metrics = (["accuracy"]
               if loss == "sparse_categorical_crossentropy"
               else [keras.metrics.CategoricalAccuracy(name="accuracy")])
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    return model


def build_vgg16(num_classes: int = NUM_CLASSES,
                input_shape: InputSpec | None = None,
                dropout: float = 0.5,
                learning_rate: float = 1e-3,
                freeze_backbone: bool = True,
                augment: bool = True):
    """Day 5 VGG16 transfer-learning model.

    Head is GlobalAveragePooling -> Dense(256) -> Dropout -> Dense(num_classes).
    The plan allows either Flatten or GAP; we choose GAP because Flatten on a
    128x128 input produces a 4x4x512 = 8192-dim feature vector, and connecting
    that to Dense(256) blows the model up to ~2 M head-only parameters which
    overfits fast on this dataset. GAP keeps the head at ~131 K params.

    Higher dropout (0.5) than MobileNetV2's head because VGG16's feature maps
    are richer and the model has been observed to overfit faster.

    Preprocessing: VGG16 expects per-channel mean subtraction in **BGR** order
    on a [0, 255]-scale input. We reproduce that with two standard layers
    rather than ``Lambda(preprocess_input)`` so the saved model round-trips
    cleanly via ``model.save`` / ``load_model``.
    """
    import keras
    from keras import layers, Model
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preproc

    shape = input_shape or _input_shape()
    raw_input = layers.Input(shape=shape, name="image")
    pre = _augmentation_layers()(raw_input) if augment else raw_input

    # [0, 1] -> [0, 255]
    x = layers.Rescaling(scale=255.0, name="vgg16_to_pixels")(pre)
    # RGB -> BGR + ImageNet mean subtraction. Calling preprocess_input directly
    # on a symbolic tensor bakes the underlying ops (reverse / subtract) into
    # the graph as regular nodes — no Lambda wrapper, so model.save / load
    # round-trip without needing to import this function on the loading side.
    x = vgg_preproc(x)

    backbone = VGG16(include_top=False, weights="imagenet",
                     input_tensor=x, input_shape=shape)
    backbone.trainable = not freeze_backbone

    h = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    h = layers.Dense(256, activation="relu", kernel_initializer="he_normal",
                     name="dense_256")(h)
    h = layers.Dropout(dropout, name="dropout")(h)
    out = layers.Dense(num_classes, activation="softmax", name="probs")(h)

    model = Model(inputs=raw_input, outputs=out, name="vgg16_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
