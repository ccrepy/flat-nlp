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

"""Utility library for signal processing with Jax.

This file is the Jax counterpart of np_signal_util.py

Unlike `np_signal_util` the jax implementation are point-wise operations (as
opposed to batches).
"""

import functools

from typing import Sequence

import jax
import jax.numpy as jnp

# Type aliases here are solely for documentation, `_NdSignal` is
# truly a jnp.ndarray tensor.
# Hence we disable type hinting for .shape and __get_item__([slice, slice, ...])

# pytype: disable=unsupported-operands
# pytype: disable=attribute-error

_1dSignal = jnp.ndarray  # pylint: disable=invalid-name
_NdSignal = jnp.ndarray  # pylint: disable=invalid-name


@jax.jit
def stride_signal(s: _NdSignal) -> _1dSignal:
  """Strides a nd signals into a flatten representation.

  The matrix is a contigus in memory, np.float64 is equivalent to double in cpp.

  Arguments:
    s: Nd signal to flatten

  Returns:
    The flatten Nd signal.
  """
  signal_dim, signal_len = s.shape
  return s.reshape(signal_dim * signal_len)


@functools.partial(jax.jit, static_argnums=(1, 2))
def unstride_signal(s_strided: _1dSignal, signal_dim: int,
                    signal_len: int) -> _NdSignal:
  """Unstrides a flatten signal into a nd signals."""
  return s_strided.reshape(signal_dim, signal_len)


@jax.jit
def rfft_to_hc(rfft_s: _NdSignal) -> _NdSignal:
  """Converts a signal from rfft to hc format."""
  rev_imag = jnp.flip(jnp.imag(rfft_s)[:, 1:-1], axis=-1)
  return jnp.concatenate([jnp.real(rfft_s), rev_imag], axis=-1)


@jax.jit
def hc_to_rfft(hc_s):
  """Converts a signal from hc to rfft format."""
  _, signal_len = hc_s.shape
  half_len = signal_len // 2

  rev_imag = jnp.flip(hc_s[:, half_len + 1:], axis=-1)

  return jax.lax.complex(hc_s[:, :half_len + 1],
                         jnp.pad(rev_imag, [[0, 0], [1, 1]]))


@jax.jit
def hc(s: _NdSignal) -> _NdSignal:
  """Transforms a time domain signal into half complex."""
  _, signal_len = s.shape
  assert signal_len % 2 == 0, 'signal_len must be even'
  return rfft_to_hc(jnp.fft.rfft(s))


@jax.jit
def ihc(hc_s: _NdSignal) -> _NdSignal:
  """Transforms a half complex signal into the time domain."""
  return jnp.fft.irfft(hc_to_rfft(hc_s))


@jax.jit
def hc_conjugate(hc_s):
  """Computes the conjugate of a hc signal."""
  _, signal_len = hc_s.shape
  half_len = signal_len // 2

  return jnp.concatenate([
      hc_s[:, :half_len + 1],
      -hc_s[:, half_len + 1:],
  ],
                         axis=-1)


@jax.jit
def hc_itemwise_pdt(hc_s1: _NdSignal, hc_s2: _NdSignal) -> _NdSignal:
  """Multiplies 2 hc signals itemwise to perform a convolution in time domain."""
  _, signal_len = hc_s1.shape
  half_len = signal_len // 2

  # Identify real and imag part of the complex harmonics (IE: 1 <= i < half_len)
  h_i_hc_s1_real = hc_s1[:, 1:half_len]
  h_i_hc_s1_imag = jnp.flip(hc_s1[:, half_len + 1:], axis=-1)

  h_i_hc_s2_real = hc_s2[:, 1:half_len]
  h_i_hc_s2_imag = jnp.flip(hc_s2[:, half_len + 1:], axis=-1)

  # Rebuild the signal by parts using hc convention:
  #   [
  #     h0,
  #     real part of (a + ib) * (a' + ib'): a * a' - b * b',
  #     hn,
  #     reversed [ imag part of (a + ib) * (a' + ib'): a * b' + a' * b],
  #   ]
  return jnp.concatenate([
      hc_s1[:, :1] * hc_s2[:, :1],
      h_i_hc_s1_real * h_i_hc_s2_real - h_i_hc_s1_imag * h_i_hc_s2_imag,
      hc_s1[:, half_len:half_len + 1] * hc_s2[:, half_len:half_len + 1],
      jnp.flip(
          h_i_hc_s1_real * h_i_hc_s2_imag + h_i_hc_s1_imag * h_i_hc_s2_real,
          axis=-1),
  ],
                         axis=-1)


@jax.jit
def energy(s: _NdSignal) -> jnp.ndarray:
  """Computes the energies of a signal."""
  return jnp.power(s, 2).sum(axis=-1)


@jax.jit
def rfft_energy(rfft_s: _NdSignal) -> jnp.ndarray:
  """Computes the energies of a rfft signal."""
  _, signal_len = rfft_s.shape
  true_len = (signal_len - 1) * 2

  # Continuous component
  h_0 = rfft_s[:, 0] * jnp.conjugate(rfft_s[:, 0])

  # Middle frequencies, they need to be counted twice in the case of rfft
  #   detailed explanation here: https://dsp.stackexchange.com/a/67110
  h_i = 2 * jnp.sum(
      rfft_s[:, 1:-1] * jnp.conjugate(rfft_s[:, 1:-1]), axis=-1)

  # Nyquist frequency
  h_n = rfft_s[:, -1] * jnp.conjugate(rfft_s[:, -1])
  return jnp.real(h_0 + h_i + h_n) / true_len


@jax.jit
def hc_energy(hc_s: _NdSignal) -> jnp.ndarray:
  """Computes the energies of a hc signal."""
  _, signal_len = hc_s.shape
  half_len = signal_len // 2

  # Continuous component
  h_0 = jnp.power(hc_s[:, 0], 2)

  # Middle frequencies, they need to be counted twice in the case of rfft
  #   detailed explanation here: https://dsp.stackexchange.com/a/67110
  h_i_real = hc_s[:, 1:half_len]
  h_i_imag = hc_s[:, half_len + 1:]  # no need to use jnp.flip for energy
  h_i = 2 * jnp.sum(jnp.power(h_i_real, 2) + jnp.power(h_i_imag, 2), axis=-1)

  # Nyquist frequency
  h_n = jnp.power(hc_s[:, half_len], 2)
  return jnp.real(h_0 + h_i + h_n) / signal_len


@functools.partial(jax.jit, static_argnums=(1,))
def hc_low_pass_mask(hc_s, cutoff_harmonic: int):
  """Computes the low pass mask for a hc signla."""
  signal_dim, signal_len = hc_s.shape
  half_len = signal_len // 2

  harmonics = jnp.concatenate([
      jnp.arange(0, half_len + 1),
      jnp.arange(half_len - 1, 0, -1)
  ])

  return jnp.tile(
      harmonics <= cutoff_harmonic,
      (signal_dim, 1),
  )
