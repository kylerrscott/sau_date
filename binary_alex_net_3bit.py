from typing import Optional, Sequence, Tuple

import larq as lq
import tensorflow as tf
from zookeeper import Field, factory

from larq_zoo.core import utils
from larq_zoo.core.model_factory import ModelFactory



# Constrains to [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
def bit3_act_zero(input):
  thresh0 = 0
  thresh1 = 0.25
  thresh2 = 0.5
  thresh3 = 0.75
  thresh4 = -0.25
  thresh5 = -0.5
  thresh6 = -0.75

  thresh = [-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875]
  quantizations = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
  masks = []

  masks.append(tf.math.less(input, thresh[0]))
  for i in range(1, len(thresh)):
    masks.append(tf.math.greater_equal(input, thresh[i-1]) & tf.math.less(input, thresh[i]))
  masks.append(tf.math.greater_equal(input, thresh[len(thresh)-1]))

  output = input
  for i in range(0, len(masks)):
    output = tf.where(masks[i], tf.ones_like(input) * quantizations[i], output)

  return output



# Constrains to [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1]
def bit3_act(input):
  thresh0 = 0
  thresh1 = 0.25
  thresh2 = 0.5
  thresh3 = 0.75
  thresh4 = -0.25
  thresh5 = -0.5
  thresh6 = -0.75

  thresh = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
  quantizations = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1]
  masks = []

  masks.append(tf.math.less(input, thresh[0]))
  for i in range(1, len(thresh)):
    masks.append(tf.math.greater_equal(input, thresh[i-1]) & tf.math.less(input, thresh[i]))
  masks.append(tf.math.greater_equal(input, thresh[len(thresh)-1]))

  output = input
  for i in range(0, len(masks)):
    output = tf.where(masks[i], tf.ones_like(input) * quantizations[i], output)

  return output



def bit2_act(input):
  thresh0 = 0
  thresh1 = 0.5
  thresh2 = -0.5

  mask1 = tf.math.greater_equal(input, thresh0) & tf.math.less(input, thresh1)
  mask2 = tf.math.greater_equal(input, thresh1)
  mask3 = tf.math.greater_equal(input, thresh2) & tf.math.less(input, thresh0)
  mask4 = tf.math.less(input, thresh2)
  
  output = tf.where(mask1, tf.ones_like(input) * 0.5, input)
  output = tf.where(mask2, tf.ones_like(input), output)
  output = tf.where(mask3, tf.ones_like(input) * -0.5, output)
  output = tf.where(mask4, tf.ones_like(input) * -1, output)

  return output



@factory
class BinaryAlexNet4BitFactory(ModelFactory):

    inflation_ratio: int = Field(1)

    input_quantizer = "ste_sign"
    kernel_quantizer = bit3_act
    kernel_constraint = "weight_clip"

    def conv_block(
        self,
        x: tf.Tensor,
        features: int,
        kernel_size: Tuple[int, int],
        strides: int = 1,
        pool: bool = False,
        first_layer: bool = False,
        no_inflation: bool = False,
    ):
        x = lq.layers.QuantConv2D(
            features * (1 if no_inflation else self.inflation_ratio),
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            input_quantizer=None if first_layer else self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_constraint=self.kernel_constraint,
            use_bias=False,
        )(x)
        if pool:
            x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
        return tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)(x)

    def dense_block(self, x: tf.Tensor, units: int) -> tf.Tensor:
        x = lq.layers.QuantConv2D(
            units,
            kernel_size=1,
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_constraint=self.kernel_constraint,
            use_bias=False,
        )(x)
        return tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)(x)

    def build(self) -> tf.keras.models.Model:
        # Feature extractor
        out = self.conv_block(
            self.image_input,
            features=64,
            kernel_size=11,
            strides=4,
            pool=True,
            first_layer=True,
        )
        out = self.conv_block(out, features=192, kernel_size=5, pool=True)
        out = self.conv_block(out, features=384, kernel_size=3)
        out = self.conv_block(out, features=384, kernel_size=3)
        out = self.conv_block(
            out, features=256, kernel_size=3, pool=True, no_inflation=True
        )

        # Classifier
        if self.include_top:
            try:
                channels = out.shape[-1] * out.shape[-2] * out.shape[-3]
            except TypeError:
                channels = -1
            out = tf.keras.layers.Reshape((1, 1, channels))(out)
            out = self.dense_block(out, units=4096)
            out = self.dense_block(out, units=4096)
            out = self.dense_block(out, self.num_classes)
            out = tf.keras.layers.Flatten()(out)
            out = tf.keras.layers.Activation("softmax", dtype="float32")(out)

        model = tf.keras.models.Model(
            inputs=self.image_input, outputs=out, name="binary_alexnet"
        )

        # Load weights.
        if self.weights == "imagenet":
            # Download appropriate file
            if self.include_top:
                weights_path = utils.download_pretrained_model(
                    model="binary_alexnet",
                    version="v0.3.0",
                    file="binary_alexnet_weights.h5",
                    file_hash="7fc065c47c5c1d92389e0bb988ce6df6a4fa09d803b866e2ba648069d6652d63",
                )
            else:
                weights_path = utils.download_pretrained_model(
                    model="binary_alexnet",
                    version="v0.2.1",
                    file="binary_alexnet_weights_notop.h5",
                    file_hash="1d41b33ff39cd28d13679392641bf7711174a96d182417f91df45d0548f5bb47",
                )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)

        return model


def BinaryAlexNet4Bit(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[utils.TensorType] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
  
    return BinaryAlexNet4BitFactory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()
