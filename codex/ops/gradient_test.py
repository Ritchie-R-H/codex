# Copyright 2022 CoDeX authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for special gradient operations."""

from typing import override
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from codex.ops import gradient


def test_upper_limit_has_correct_outputs_and_gradients():
  x = jnp.array([-1, 1], dtype=jnp.float32)
  def fun(x):
    return gradient.upper_limit(x, 0)
  y, vjp = jax.vjp(fun, x)
  pos_dydx, = vjp(jnp.ones_like(x))
  neg_dydx, = vjp(-jnp.ones_like(x))
  assert (y == jnp.array([-1, 0])).all()
  assert (pos_dydx == jnp.array([1, 1])).all()
  assert (neg_dydx == jnp.array([-1, 0])).all()


def test_lower_limit_has_correct_outputs_and_gradients():
  x = jnp.array([-1, 1], dtype=jnp.float32)
  def fun(x):
    return gradient.lower_limit(x, 0)
  y, vjp = jax.vjp(fun, x)
  pos_dydx, = vjp(jnp.ones_like(x))
  neg_dydx, = vjp(-jnp.ones_like(x))
  assert (y == jnp.array([0, 1])).all()
  assert (pos_dydx == jnp.array([0, 1])).all()
  assert (neg_dydx == jnp.array([-1, -1])).all()


def test_perturb_and_apply_returns_function_output():
  def f(a, b):
    return (a ** 3) * (b ** 2)
  rng1, rng2, rng3 = jax.random.split(jax.random.key(0), 3)
  x = jax.random.normal(rng1, (10,))
  y = jax.random.normal(rng2, x.shape)
  u = jax.random.uniform(rng3, x.shape, minval=-.5, maxval=.5)
  chex.assert_trees_all_close(
      gradient.perturb_and_apply(f, x, u, y),
      f(x + u, y))


def test_perturb_and_apply_computes_x_gradient_correctly():
  def f(a, b):
    return (a ** 3) * (b ** 2)
  rng1, rng2, rng3 = jax.random.split(jax.random.key(1), 3)
  x = jax.random.normal(rng1, (10,))
  y = jax.random.normal(rng2, x.shape)
  u = jax.random.uniform(rng3, x.shape, minval=-.5, maxval=.5)
  chex.assert_trees_all_close(
      jax.grad(lambda x: gradient.perturb_and_apply(f, x, u, y).sum())(x),
      f(x + .5, y) - f(x - .5, y))


def test_perturb_and_apply_computes_args_gradient_correctly():
  def f(a, b):
    return (a ** 3) * (b ** 2)
  rng1, rng2, rng3 = jax.random.split(jax.random.key(2), 3)
  x = jax.random.normal(rng1, (10,))
  y = jax.random.normal(rng2, x.shape)
  u = jax.random.uniform(rng3, x.shape, minval=-.5, maxval=.5)
  chex.assert_trees_all_close(
      jax.grad(lambda y: gradient.perturb_and_apply(f, x, u, y).sum())(y),
      jax.jvp(f, (x + u, y), (jnp.zeros_like(x), jnp.ones_like(y)))[1])


def test_perturb_and_apply_does_not_fail_in_flax_modules():
  class InnerWithCall(nn.Module):
    @nn.compact
    @override
    def __call__(self, x):
      bias = self.param("bias", nn.initializers.constant(.25), x.shape)
      return x ** 2 + bias ** 3

  class InnerWithMethod(nn.Module):
    @nn.compact
    def method(self, x):
      bias = self.param("bias", nn.initializers.constant(.25), x.shape)
      return x ** 2 + bias ** 3

  class OuterWithCall(nn.Module):
    @nn.compact
    @override
    def __call__(self, x, u):
      inner = InnerWithCall()
      return gradient.perturb_and_apply(inner, x, u)

  class OuterWithMethod(nn.Module):
    @nn.compact
    @override
    def __call__(self, x, u):
      inner = InnerWithMethod()
      return gradient.perturb_and_apply(inner.method, x, u)

  class OuterWithLambda(nn.Module):
    @nn.compact
    @override
    def __call__(self, x, u):
      inner = InnerWithCall()
      return gradient.perturb_and_apply(lambda x: 1. * inner(x), x, u)

  x = jnp.full((), .75)
  u = jnp.full((), 1.)
  p_call = OuterWithCall().init(jax.random.key(0), x, u)
  p_method = OuterWithMethod().init(jax.random.key(0), x, u)
  p_lambda = OuterWithLambda().init(jax.random.key(0), x, u)

  x_dot_call = jax.grad(lambda x: OuterWithCall().apply(p_call, x, u))(x)
  x_dot_method = jax.grad(lambda x: OuterWithMethod().apply(p_method, x, u))(x)
  x_dot_lambda = jax.grad(lambda x: OuterWithLambda().apply(p_lambda, x, u))(x)
  chex.assert_trees_all_equal(x_dot_call, 1.5)
  chex.assert_trees_all_equal(x_dot_method, 1.5)
  chex.assert_trees_all_equal(x_dot_lambda, 1.5)

  u_dot_call = jax.grad(lambda u: OuterWithCall().apply(p_call, x, u))(u)
  u_dot_method = jax.grad(lambda u: OuterWithMethod().apply(p_method, x, u))(u)
  u_dot_lambda = jax.grad(lambda u: OuterWithLambda().apply(p_lambda, x, u))(u)
  chex.assert_trees_all_equal(u_dot_call, 3.5)
  chex.assert_trees_all_equal(u_dot_method, 3.5)
  chex.assert_trees_all_equal(u_dot_lambda, 3.5)

  p_dot_call = jax.grad(lambda p: OuterWithCall().apply(p, x, u))(p_call)
  p_dot_call, = jax.tree_util.tree_leaves(p_dot_call)
  p_dot_method = jax.grad(lambda p: OuterWithMethod().apply(p, x, u))(p_method)
  p_dot_method, = jax.tree_util.tree_leaves(p_dot_method)
  p_dot_lambda = jax.grad(lambda p: OuterWithLambda().apply(p, x, u))(p_lambda)
  p_dot_lambda, = jax.tree_util.tree_leaves(p_dot_lambda)
  chex.assert_trees_all_equal(p_dot_call, .1875)
  chex.assert_trees_all_equal(p_dot_method, .1875)
  chex.assert_trees_all_equal(p_dot_lambda, .1875)
