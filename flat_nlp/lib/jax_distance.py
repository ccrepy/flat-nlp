# Copyright 2023 Flat NLP Authors.
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

"""Library for Flat NLP distance jax implementation.

Distances are defined point-wise.
"""

import jax
import jax.numpy as jnp

from flat_nlp.lib import jax_signal_util


@jax.jit
def convolve_signal(s1: jnp.ndarray, s2: jnp.ndarray) -> jnp.ndarray:
  """Computes the discrete correlation of 2 Nd signals."""
  s2 = jnp.flip(s2, axis=-1)
  return jnp.fft.ifft(jnp.fft.fft(s1) * jnp.fft.fft(s2)).real


@jax.jit
def convolve_rfft_signal(
    rfft_s1: jnp.ndarray, rfft_s2: jnp.ndarray
) -> jnp.ndarray:
  """Computes the discrete correlation of 2 Nd signals in rfft format."""
  s_out = jnp.fft.irfft(rfft_s1 * rfft_s2.conjugate(), axis=-1)
  s_out = jnp.roll(s_out, -1, axis=-1)
  return jnp.fft.rfft(s_out)


@jax.jit
def convolve_hc_signal(hc_s1: jnp.ndarray, hc_s2: jnp.ndarray) -> jnp.ndarray:
  """Computes the discrete correlation of 2 Nd signals in halfcomplex format."""
  hc_s2 = jax_signal_util.hc_conjugate(hc_s2)

  s_out = jax_signal_util.ihc(jax_signal_util.hc_itemwise_pdt(hc_s1, hc_s2))
  s_out = jnp.roll(s_out, -1, axis=-1)
  return jax_signal_util.hc(s_out)


@jax.jit
def flat_distance(
    s1: jnp.ndarray, s2: jnp.ndarray, epsilon: float = 1e-6
) -> jnp.float32:
  """Computes flat distance between 2 Nd signals."""
  convolved_signal = convolve_signal(s1, s2)

  signal_energy = jnp.sqrt(
      jnp.power(s1, 2).sum(axis=-1) * jnp.power(s2, 2).sum(axis=-1)
  )
  convolved_signal = convolved_signal / signal_energy[:, jnp.newaxis]

  peak_correlation = jnp.average(convolved_signal, axis=-2).max()
  return jnp.sqrt(1.0 - jnp.power(peak_correlation, 2) + epsilon)


@jax.jit
def rfft_flat_distance(
    rfft_s1: jnp.ndarray, rfft_s2: jnp.ndarray, epsilon: float = 1e-6
) -> jnp.float32:
  """Computes flat distance between 2 Nd signals in rfft format."""
  convolved_signal = jnp.fft.irfft(convolve_rfft_signal(rfft_s1, rfft_s2))

  signal_energy = jnp.sqrt(
      jax_signal_util.rfft_energy(rfft_s1)
      * jax_signal_util.rfft_energy(rfft_s2)
  )
  convolved_signal = convolved_signal / signal_energy[:, jnp.newaxis]

  peak_correlation = jnp.average(convolved_signal, axis=0).max()
  return jnp.sqrt(1.0 - jnp.power(peak_correlation, 2) + epsilon)


@jax.jit
def hc_flat_distance(
    hc_s1: jnp.ndarray, hc_s2: jnp.ndarray, epsilon: float = 1e-6
) -> jnp.float32:
  """Computes flat distance between 2 Nd signals in halfcomplex format."""
  convolved_signal = jax_signal_util.ihc(convolve_hc_signal(hc_s1, hc_s2))

  signal_energy = jnp.sqrt(
      jax_signal_util.hc_energy(hc_s1) * jax_signal_util.hc_energy(hc_s2)
  )
  convolved_signal = convolved_signal / signal_energy[:, jnp.newaxis]

  peak_correlation = jnp.average(convolved_signal, axis=-2).max()
  return jnp.sqrt(1.0 - jnp.power(peak_correlation, 2) + epsilon)
