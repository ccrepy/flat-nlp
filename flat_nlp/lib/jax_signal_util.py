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

"""Utility library for signal processing with Jax.

This file is the Jax counterpart of np_signal_util.py
"""
import functools

from typing import Sequence

import jax
import jax.numpy as jnp

# Type aliases here are solely for documentation, `Sequence[_NdSignal]` is
# truly a jnp.ndarray tensor.
# Hence we disable type hinting for .shape and __get_item__([slice, slice, ...])

# pytype: disable=unsupported-operands
# pytype: disable=attribute-error

_1dSignal = jnp.ndarray  # pylint: disable=invalid-name
_NdSignal = Sequence[_1dSignal]


@jax.jit
def stride_signal(signal_list: Sequence[_NdSignal]) -> Sequence[_1dSignal]:
  """Strides a sequence of nd signals into a 2d matrix.

  The matrix is a contigus in memory, np.float64 is equivalent to double in cpp.

  Arguments:
    signal_list: array of Nd signals

  Returns:
    The 2d matrix representation of the Nd signals.
  """
  n_signal, signal_dim, signal_len = signal_list.shape
  return signal_list.reshape(n_signal, signal_dim * signal_len)


@functools.partial(jax.jit, static_argnums=(1, 2))
def unstride_signal(strided_signal_list: Sequence[_1dSignal], signal_dim: int,
                    signal_len: int) -> Sequence[_NdSignal]:
  """Unstrides a 2d matrix into a sequence of nd signals."""
  return strided_signal_list.reshape(-1, signal_dim, signal_len)


@jax.jit
def rfft_to_hc(rfft_t: Sequence[_NdSignal]) -> Sequence[_NdSignal]:
  """Converts a tensor from rfft to hc format."""
  rev_imag = jnp.flip(jnp.imag(rfft_t)[:, :, 1:-1], axis=-1)
  return jnp.concatenate([jnp.real(rfft_t), rev_imag], axis=-1)


@jax.jit
def hc_to_rfft(hc_t):
  """Converts a tensor from hc to rfft format."""
  _, _, signal_len = hc_t.shape
  half_len = signal_len // 2

  rev_imag = jnp.flip(hc_t[:, :, half_len + 1:], axis=2)

  return jax.lax.complex(hc_t[:, :, :half_len + 1],
                         jnp.pad(rev_imag, [[0, 0], [0, 0], [1, 1]]))


@jax.jit
def hc(t: Sequence[_NdSignal]) -> Sequence[_NdSignal]:
  """Transforms a time domain signal into half complex."""
  _, _, signal_len = t.shape
  assert signal_len % 2 == 0, 'signal_len must be even'
  return rfft_to_hc(jnp.fft.rfft(t))


@jax.jit
def ihc(hc_t: Sequence[_NdSignal]) -> Sequence[_NdSignal]:
  """Transforms a half complex signal into the time domain."""
  return jnp.fft.irfft(hc_to_rfft(hc_t))


@jax.jit
def hc_conjugate(hc_t):
  """Computes the conjugate of a hc tensor."""
  _, _, signal_len = hc_t.shape
  half_len = signal_len // 2

  return jnp.concatenate([
      hc_t[:, :, :half_len + 1],
      -hc_t[:, :, half_len + 1:],
  ],
                         axis=2)


@jax.jit
def hc_itemwise_pdt(hc_t1, hc_t2):
  """Multiplies 2 hc tensors itemwise to perform a convolution in time domain."""
  _, _, signal_len = hc_t1.shape
  half_len = signal_len // 2

  # Identify real and imag part of the complex harmonics (IE: 1 <= i < half_len)
  h_i_hc_t1_real = hc_t1[:, :, 1:half_len]
  h_i_hc_t1_imag = jnp.flip(hc_t1[:, :, half_len + 1:], axis=2)

  h_i_hc_t2_real = hc_t2[:, :, 1:half_len]
  h_i_hc_t2_imag = jnp.flip(hc_t2[:, :, half_len + 1:], axis=2)

  # Rebuild the signal by parts using hc convention:
  #   [
  #     h0,
  #     real part of (a + ib) * (a' + ib'): a * a' - b * b',
  #     hn,
  #     reversed [ imag part of (a + ib) * (a' + ib'): a * b' + a' * b],
  #   ]
  return jnp.concatenate([
      hc_t1[:, :, :1] * hc_t2[:, :, :1],
      h_i_hc_t1_real * h_i_hc_t2_real - h_i_hc_t1_imag * h_i_hc_t2_imag,
      hc_t1[:, :, half_len:half_len + 1] * hc_t2[:, :, half_len:half_len + 1],
      jnp.flip(
          h_i_hc_t1_real * h_i_hc_t2_imag + h_i_hc_t1_imag * h_i_hc_t2_real,
          axis=2),
  ],
                         axis=2)


@jax.jit
def energy(t: Sequence[_NdSignal]) -> Sequence[Sequence[float]]:
  """Computes the energies of a tensor."""
  return jnp.power(t, 2).sum(axis=-1)


@jax.jit
def rfft_energy(rfft_t: Sequence[_NdSignal]) -> Sequence[Sequence[float]]:
  """Computes the energies of a rfft tensor."""
  _, _, signal_len = rfft_t.shape
  true_len = (signal_len - 1) * 2

  # Continuous component
  h_0 = rfft_t[:, :, 0] * jnp.conjugate(rfft_t[:, :, 0])

  # Middle frequencies, they need to be counted twice in the case of rfft
  #   detailed explanation here: https://dsp.stackexchange.com/a/67110
  h_i = 2 * jnp.sum(
      rfft_t[:, :, 1:-1] * jnp.conjugate(rfft_t[:, :, 1:-1]), axis=2)

  # Nyquist frequency
  h_n = rfft_t[:, :, -1] * jnp.conjugate(rfft_t[:, :, -1])
  return jnp.real(h_0 + h_i + h_n) / true_len


@jax.jit
def hc_energy(hc_t: Sequence[_NdSignal]) -> Sequence[Sequence[float]]:
  """Computes the energies of a hc tensor."""
  _, _, signal_len = hc_t.shape
  half_len = signal_len // 2

  # Continuous component
  h_0 = jnp.power(hc_t[:, :, 0], 2)

  # Middle frequencies, they need to be counted twice in the case of rfft
  #   detailed explanation here: https://dsp.stackexchange.com/a/67110
  h_i_real = hc_t[:, :, 1:half_len]
  h_i_imag = hc_t[:, :, half_len + 1:]  # no need to use jnp.flip
  h_i = 2 * jnp.sum(jnp.power(h_i_real, 2) + jnp.power(h_i_imag, 2), axis=2)

  # Nyquist frequency
  h_n = jnp.power(hc_t[:, :, half_len], 2)
  return jnp.real(h_0 + h_i + h_n) / signal_len

