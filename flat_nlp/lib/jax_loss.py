# Copyright 2022 Flat NLP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for Flat NLP loss jax implementation.

Losses are defined tensor-wise (as opposed to distance which work at point
level).

Losses expect the signals to be energy normalized.
"""

import jax
import jax.numpy as jnp


@jax.jit
def convolve_tensor(t1: jnp.array, t2: jnp.array) -> jnp.array:
  """Computes the discrete correlation of 2 Nd tensors."""
  t2 = jnp.flip(t2, axis=-1)
  t_out = jnp.fft.ifft(jnp.fft.fft(t1) * jnp.fft.fft(t2)).real
  return t_out


@jax.jit
def convolve_hc_tensor(t1: jnp.array, t2: jnp.array) -> jnp.array:
  raise NotImplementedError


@jax.jit
def flat_loss(t1: jnp.array, t2: jnp.array, epsilon=1e-6) -> jnp.array:
  """Computes flat loss between 2 Nd tensors, tensors are expected to be energy normalized."""
  convolved_tensors = convolve_tensor(t1, t2)
  peak_correlations = jnp.max(jnp.average(convolved_tensors, axis=-2), axis=-1)
  distances = jnp.sqrt(1. - jnp.power(peak_correlations, 2) + epsilon)
  return distances


@jax.jit
def hc_flat_loss(t1: jnp.array, t2: jnp.array) -> jnp.array:
  """Computes flat distance between 2 Nd tensors."""
  raise NotImplementedError
