# Copyright 2025 CoDeX authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language governing
# permissions and limitations under the License.
# ========================================================================================
"""Computes features for perceptual representations."""

import jax
import jax.numpy as jnp
from codex.loss import wasserstein

try:
    # Soft dependency on `flaxmodels`. If the package is not installed, CoDeX should still
    # be importable, but without the VGG Wasserstein loss.
    import flaxmodels as fm
except ImportError:
    fm = None

Array = jax.Array
ArrayLike = jax.typing.ArrayLike

_MODELS = {}


def load_vgg16_model(mock: bool = False) -> None:
    """Loads VGG16 model with pretrained ImageNet weights.

    Parameters
    ----------
    mock
        Do not load actual weights, just initialize. This is for unit testing.
    """
    if fm is None:
        raise RuntimeError("Please install `flaxmodels` package for VGG features.")
    vgg = fm.VGG16(
        output="activations",
        pretrained=None if mock else "imagenet",  # pyright: ignore
        normalize=False,
        pooling="avg_pool",
        include_head=False,
    )
    dummy_input = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
    params = vgg.init(jax.random.key(0), dummy_input)
    _MODELS["vgg16"] = (vgg, params)


def compute_vgg16_features(image: ArrayLike, num_scales: int = 3) -> list[Array]:
    """Extracts VGG features from an image.

    Parameters
    ----------
    image
        Image in format ``(3, height, width)``. The pixel format is assumed to be: sRGB
        colorspace, with pixel values linearly scaled to floats in the range [0, 1].
    num_scales
        The number of scales of the image the features should be computed on. The image
        will be downsampled ``num_scales - 1`` times and VGG features computed on the
        original image plus the downsampled versions. The concatenated list of all
        features will be returned.

    Returns
    -------
    list
        A list of feature arrays of format ``(channel, height, width)``.
    """
    try:
        vgg, params = _MODELS["vgg16"]
    except KeyError:
        raise RuntimeError("Please call `load_vgg16_model` first.")
    layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
    mean = jnp.array([0.485, 0.456, 0.406]).reshape((-1, 1, 1)).astype(image.dtype)
    std = jnp.array([0.229, 0.224, 0.225]).reshape((-1, 1, 1)).astype(image.dtype)
    image = (image - mean) / std
    features: list[Array] = [image]  # type:ignore
    for _ in range(num_scales):
        activations = vgg.apply(params, image.transpose((1, 2, 0))[None], train=False)
        features.extend(activations[l].squeeze(0).transpose((2, 0, 1)) for l in layers)
        image = wasserstein.lowpass(image, 2)
    return features
