"""
model.py — 3D U-Net with residual blocks and optional SE attention.

All parameters are passed explicitly through function arguments.
Default values come from config.py (imported lazily, never at module level).
"""

import tensorflow as tf
from tensorflow import keras as K


# ══════════════════════════════════════════════════════════════════════════════
#  Losses & Metrics
# ══════════════════════════════════════════════════════════════════════════════

def dice_coef(target, prediction, axis=(1, 2, 3, 4), smooth=1e-6,
              threshold=None):
    """Dice coefficient. soft by default, hard when threshold is set."""
    if threshold is not None:
        prediction = tf.cast(prediction > threshold, prediction.dtype)
    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union        = tf.reduce_sum(target + prediction, axis=axis)
    dice         = (2.0 * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)


def dice_loss(target, prediction, axis=(1, 2, 3), smooth=1e-4):
    """Soft Dice loss using -log(Dice)."""
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator   = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    return -tf.math.log(2.0 * numerator) + tf.math.log(denominator)


# ══════════════════════════════════════════════════════════════════════════════
#  Building blocks
# ══════════════════════════════════════════════════════════════════════════════

def ResidualBlock(x, name: str, filters: int, params: dict,
                  dropout_rate: float = 0.20):
    """Two-conv residual unit with BN + ReLU + optional SpatialDropout3D."""
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = K.layers.Conv3D(
            filters, kernel_size=(1, 1, 1), padding="same",
            kernel_initializer="he_uniform", name=f"{name}_proj")(shortcut)
        shortcut = K.layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    out = K.layers.Conv3D(filters=filters, **params, name=f"{name}_conv1")(x)
    out = K.layers.BatchNormalization(name=f"{name}_bn1")(out)
    out = K.layers.Activation("relu", name=f"{name}_relu1")(out)
    if dropout_rate > 0:
        out = K.layers.SpatialDropout3D(dropout_rate, name=f"{name}_drop1")(out)

    out = K.layers.Conv3D(filters=filters, **params, name=f"{name}_conv2")(out)
    out = K.layers.BatchNormalization(name=f"{name}_bn2")(out)
    if dropout_rate > 0:
        out = K.layers.SpatialDropout3D(dropout_rate, name=f"{name}_drop2")(out)

    out = K.layers.Add(name=f"{name}_add")([shortcut, out])
    out = K.layers.Activation("relu", name=name)(out)
    return out


def SEGate(x, name: str, r: int = 8):
    """Squeeze-and-Excitation attention for 3-D tensors."""
    ch = x.shape[-1]
    z = K.layers.GlobalAveragePooling3D(name=f"{name}_gap")(x)
    z = K.layers.Dense(ch // r, activation="relu", name=f"{name}_fc1")(z)
    z = K.layers.Dense(ch, activation="sigmoid", name=f"{name}_fc2")(z)
    z = K.layers.Reshape((1, 1, 1, ch), name=f"{name}_reshape")(z)
    return K.layers.Multiply(name=f"{name}_scale")([x, z])


# ══════════════════════════════════════════════════════════════════════════════
#  U-Net 3D
# ══════════════════════════════════════════════════════════════════════════════

def unet_3d(input_dim,
            filters=16,
            number_output_classes=1,
            use_upsampling=False,
            dropout_rate=0.20,
            model_name="unet3d"):
    """
    U-Net-3D with residual blocks and Dropout.
    All parameters are explicit — no global imports.
    """

    def RB(x, name, f):
        return ResidualBlock(x, name, f, params, dropout_rate)

    inputs = K.layers.Input(shape=input_dim, name="MRImages")

    params = dict(kernel_size=(3, 3, 3), activation=None,
                  padding="same", kernel_initializer="he_uniform")
    params_trans = dict(kernel_size=(2, 2, 2), strides=(2, 2, 2),
                        padding="same", kernel_initializer="he_uniform")

    # ── Encoder ──────────────────────────────────────────────────────────
    encodeA = RB(inputs, "encodeA", filters)
    poolA   = K.layers.MaxPooling3D(pool_size=(2, 2, 2), name="poolA")(encodeA)

    encodeB = RB(poolA, "encodeB", filters * 2)
    poolB   = K.layers.MaxPooling3D(pool_size=(2, 2, 2), name="poolB")(encodeB)

    encodeC = RB(poolB, "encodeC", filters * 4)
    poolC   = K.layers.MaxPooling3D(pool_size=(2, 2, 2), name="poolC")(encodeC)

    encodeD = RB(poolC, "encodeD", filters * 8)
    poolD   = K.layers.MaxPooling3D(pool_size=(2, 2, 2), name="poolD")(encodeD)

    encodeE = RB(poolD, "encodeE", filters * 16)

    # ── Decoder ──────────────────────────────────────────────────────────
    up = (K.layers.UpSampling3D(size=(2, 2, 2), name="upE")(encodeE)
          if use_upsampling else
          K.layers.Conv3DTranspose(filters * 8, **params_trans,
                                   name="transconvE")(encodeE))
    concatD = K.layers.concatenate([up, encodeD], axis=-1, name="concatD")
    decodeC = RB(concatD, "decodeC", filters * 8)

    up = (K.layers.UpSampling3D(size=(2, 2, 2), name="upC")(decodeC)
          if use_upsampling else
          K.layers.Conv3DTranspose(filters * 4, **params_trans,
                                   name="transconvC")(decodeC))
    concatC = K.layers.concatenate([up, encodeC], axis=-1, name="concatC")
    decodeB = RB(concatC, "decodeB", filters * 4)

    up = (K.layers.UpSampling3D(size=(2, 2, 2), name="upB")(decodeB)
          if use_upsampling else
          K.layers.Conv3DTranspose(filters * 2, **params_trans,
                                   name="transconvB")(decodeB))
    concatB = K.layers.concatenate([up, encodeB], axis=-1, name="concatB")
    decodeA = RB(concatB, "decodeA", filters * 2)

    up = (K.layers.UpSampling3D(size=(2, 2, 2), name="upA")(decodeA)
          if use_upsampling else
          K.layers.Conv3DTranspose(filters, **params_trans,
                                   name="transconvA")(decodeA))
    concatA = K.layers.concatenate([up, encodeA], axis=-1, name="concatA")
    convOut = RB(concatA, "convOut", filters)

    prediction = K.layers.Conv3D(
        filters=number_output_classes,
        kernel_size=(1, 1, 1),
        activation="sigmoid",
        name="PredictionMask")(convOut)

    model = K.models.Model(inputs=[inputs], outputs=[prediction],
                           name=model_name)
    return model