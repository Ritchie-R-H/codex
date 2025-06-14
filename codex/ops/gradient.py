# Copyright 2022 CoDeX authors.
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
"""Special gradient operations."""

from collections.abc import Callable
import jax
import jax.numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


@jax.custom_vjp
def upper_limit(inputs: ArrayLike, limit: ArrayLike) -> Array:
    """Limits an array from above, with faked gradients.

    In contrast to `jnp.minimum`, this function never returns a gradient for `limit`, and
    for values in `inputs` which exceed the limit, it returns a gradient if and only if
    that gradient is positive.

    Parameters
    ----------
    inputs
        The input array.
    limit
        Upper limit for `inputs`.

    Returns
    -------
    Array
      ``jnp.minimum(inputs, limit)``.
    """
    return jnp.minimum(inputs, limit)


@jax.custom_vjp
def lower_limit(inputs: ArrayLike, limit: ArrayLike) -> Array:
    """Limits an array from below, with faked gradients.

    In contrast to `jnp.maximum`, this function never returns a gradient for `limit`, and
    for values in `inputs` which exceed the limit, it returns a gradient if and only if
    that gradient is negative.

    Parameters
    ----------
    inputs
        The input array.
    limit
        Lower limit for `inputs`.

    Returns
    -------
    Array
        ``jnp.maximum(inputs, limit)``.
    """
    return jnp.maximum(inputs, limit)


def upper_limit_fwd(inputs, limit):
    return upper_limit(inputs, limit), (inputs <= limit,)


def lower_limit_fwd(inputs, limit):
    return lower_limit(inputs, limit), (inputs >= limit,)


def upper_limit_bwd(res, grad):
    (limit_inactive,) = res
    pass_through_if = jnp.logical_or(limit_inactive, grad > 0.0)
    inputs_grad = jnp.where(pass_through_if, grad, 0.0)
    return inputs_grad, None


def lower_limit_bwd(res, grad):
    (limit_inactive,) = res
    pass_through_if = jnp.logical_or(limit_inactive, grad < 0.0)
    inputs_grad = jnp.where(pass_through_if, grad, 0.0)
    return inputs_grad, None


upper_limit.defvjp(upper_limit_fwd, upper_limit_bwd)
lower_limit.defvjp(lower_limit_fwd, lower_limit_bwd)


def perturb_and_apply(f: Callable, x: ArrayLike, u: ArrayLike, *args) -> Array:
    """Perturbs the inputs of a pointwise function using JAX.

    This function adds uniform noise in the range -0.5 to 0.5 to the first argument of the
    given function. It further replaces derivatives of the function with (analytically
    computed) expected derivatives w.r.t. the noise.

    Parameters
    ----------
    f
        JAX transformable pointwise function.
    x
        The inputs.
    u
        The noise realization to perturb `x` with. Must be a sample from a uniform
        distribution.
    *args
        Optional, additional arguments of `f`.

    Returns
    -------
    Array
        ``y=f(x+u, *args)``. The gradient of `y` w.r.t. `x` takes the expectation over
        the derivatives w.r.t. the distribution of `u` in closed form.

    Notes
    -----
    This function is further described in Sec. 4.2. of [1]_. Please cite the paper if you
    use this code for scientific or research work.

    .. [1] E. Agustsson, L. Theis: "Universally Quantized Neural Compression," Adv. in
       Neural Information Processing Systems, vol. 33, 2020.
       https://arxiv.org/abs/2006.09952
    """
    # This is the correct output of the function, and allows automatically computing the
    # gradient wrt. all arguments and closures of f, except x.
    output = f(jax.lax.stop_gradient(x) + u, *args)

    # Capture all closures as extra arguments of a new function. Then disable gradient
    # propagation to all arguments except x. Note: closure_convert fails if f is a module
    # since it tries to hash it, and Flax disallows hashing of modules with variables. So
    # we have to wrap it in a lambda function.
    new_f, extra_args = jax.closure_convert(
        lambda *a: f(*a), x, *args  # pylint: disable=unnecessary-lambda
    )
    new_args = jax.lax.stop_gradient(args + tuple(extra_args))

    # Define a function that returns zeros in the forward pass, but defines the
    # closed-form derivative of f with respect to x.
    @jax.custom_jvp
    def zeros_with_df_dx(x, _):
        return jnp.zeros_like(x)

    @zeros_with_df_dx.defjvp
    def zeros_with_df_dx_jvp(primals, tangents):  # type: ignore
        x, a = primals
        x_dot, _ = tangents
        f_dot = (new_f(x + 0.5, *a) - new_f(x - 0.5, *a)) * x_dot
        return jnp.zeros_like(x), f_dot

    # Add the custom function (zeros) to the output, so that gradients wrt. x flow through
    # the custom function, and all others flow through f itself.
    return output + zeros_with_df_dx(x, new_args)
