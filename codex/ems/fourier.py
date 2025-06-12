# Copyright 2024 CoDeX authors.
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
"""Fourier basis entropy models."""

from typing import override
import jax
from jax import nn
import jax.numpy as jnp
from codex.ems import continuous
from codex.ops import quantization

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


def autocorrelate(sequence: Array, precision=None) -> Array:
    """Computes batched complex-valued autocorrelation.

    Parameters
    ----------
    sequence
        Array of shape `(batch, length)`, where `batch` is the number of independent
        sequences to correlate with themselves, and `length` is the length of each
        sequence.
    precision
        See `jax.lax.conv_general_dilated`.

    Returns
    -------
    Array
        Shape `(batch, length)`. The right half of each autocorrelation sequence. The left
        half is redundant due to symmetry (even for the real part, odd for the imaginary
        part).
    """
    batch, length = sequence.shape
    return jax.lax.conv_general_dilated(
        sequence[None, :, :],
        sequence[:, None, :].conj(),
        window_strides=(1,),
        padding=[(0, length - 1)],
        feature_group_count=batch,
        precision=precision,
    )[0]


def periodic_prob(coef: Array, x: Array, y: Array = None) -> Array:
    """Evaluates PDF or difference of CDFs of a periodic Fourier basis density.

    This function assumes the model is periodic with period 2.

    Parameters
    ----------
    coef
        Coefficients of Fourier series.
    x
        Locations to evaluate the PDF, or lower location of CDF if `y` is provided.
    y
        If provided, upper location of CDF.

    Returns
    -------
    Array
        If `y` is not provided: `p(x)`, where `p` is the PDF. If `y` is provided: `P(y) -
        P(x)`, where `P` is the CDF.
    """
    _, num_freqs = coef.shape

    # First, autocorrelate coefficients to ensure a non-negative density.
    coef = autocorrelate(coef)

    # The DC coefficient is special: it is the normalizer of the density.
    dc = coef[:, 0].real
    ac = coef[:, 1:]
    pi_n = jnp.pi * jnp.arange(1, num_freqs)

    # Note: we can take the real part below because the coefficient sequence is assumed to
    # have Hermitian symmetry, so the Fourier series is always real.
    if y is None:
        pi_n_x = pi_n * x[..., None]
        return (
            (ac.real * jnp.cos(pi_n_x)).sum(axis=-1) -
            (ac.imag * jnp.sin(pi_n_x)).sum(axis=-1)
        ) / dc + 0.5
    else:
        pi_n_x = pi_n * x[..., None]
        pi_n_y = pi_n * y[..., None]
        cos_diff = jnp.cos(pi_n_y) - jnp.cos(pi_n_x)
        sin_diff = jnp.sin(pi_n_y) - jnp.sin(pi_n_x)
        return (
            ((ac.real / pi_n) * sin_diff).sum(axis=-1) +
            ((ac.imag / pi_n) * cos_diff).sum(axis=-1)
        ) / dc + (y - x) / 2


class PeriodicFourierEntropyModelBase(continuous.ContinuousEntropyModel):
    """Fourier basis entropy model for periodic distributions."""

    real: Array
    imag: Array
    period: float

    def prob(self, x: Array) -> Array:
        # Get and transform model parameters.
        coef = jax.lax.complex(self.real, self.imag)

        # Change of variables. Here, g^{-1} is simply a rescaling to the period.
        dg_inv_dx = 2 / self.period
        g_inv = dg_inv_dx * x

        return periodic_prob(coef, g_inv) * dg_inv_dx

    def neg_log_prob(self, x: Array, eps: float = 1e-20) -> Array:
        p = self.prob(x)
        p = jnp.maximum(p, eps)
        return -jnp.log(p)

    @override
    def bin_prob(self, center: ArrayLike, temperature: ArrayLike = None) -> Array:
        center, temperature = self._maybe_upcast((center, temperature))

        # Get and transform model parameters.
        coef = jax.lax.complex(self.real, self.imag)

        # Transformation for soft rounding.
        upper = quantization.soft_round_inverse(center + 0.5, temperature)
        # soft_round is periodic with period 1, so we don't need to call it again.
        lower = upper - 1

        # Change of variables. Here, g^{-1} is simply a rescaling to the period.
        dg_inv_dx = 2.0 / self.period
        lower *= dg_inv_dx
        upper *= dg_inv_dx

        return periodic_prob(coef, lower, upper)

    @override
    def bin_bits(
        self, center: ArrayLike, temperature: ArrayLike = None, eps: float = 1e-20
    ) -> Array:
        p = self.bin_prob(center, temperature)
        p = jnp.maximum(p, eps)
        return jnp.log(p) / -jnp.log(2.0)


class RealMappedFourierEntropyModelBase(continuous.ContinuousEntropyModel):
    """Fourier basis entropy model mapped to the real line."""

    real: Array
    imag: Array
    offset: Array
    scale: Array

    def prob(self, x: ArrayLike) -> Array:
        # Get and transform model parameters.
        coef = jax.lax.complex(self.real, self.imag)
        scale = nn.softplus(self.scale)
        offset = self.offset

        # Change of variables using scaled and shifted hyperbolic tangent.
        g_inv = jnp.tanh((x - offset) / scale)
        dg_inv_dx = (1 - jnp.square(g_inv)) / scale

        return periodic_prob(coef, g_inv) * dg_inv_dx

    def neg_log_prob(self, x: ArrayLike, eps: float = 1e-20) -> Array:
        p = self.prob(x)
        p = jnp.maximum(p, eps)
        return -jnp.log(p)

    @override
    def bin_prob(self, center: ArrayLike, temperature: ArrayLike = None) -> Array:
        center, temperature = self._maybe_upcast((center, temperature))

        # Get and transform model parameters.
        coef = jax.lax.complex(self.real, self.imag)
        scale = nn.softplus(self.scale)
        offset = self.offset

        # Transformation for soft rounding.
        upper = quantization.soft_round_inverse(center + 0.5, temperature)
        # soft_round is periodic with period 1, so we don't need to call it again.
        lower = upper - 1

        # Change of variables using scaled and shifted hyperbolic tangent.
        upper = jnp.tanh((upper - offset) / scale)
        lower = jnp.tanh((lower - offset) / scale)

        return periodic_prob(coef, lower, upper)

    @override
    def bin_bits(
        self, center: ArrayLike, temperature: ArrayLike = None, eps: float = 1e-20
    ) -> Array:
        p = self.bin_prob(center, temperature)
        p = jnp.maximum(p, eps)
        return jnp.log(p) / -jnp.log(2.0)
