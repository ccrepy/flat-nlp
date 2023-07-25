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

"""Utility library for signal processing with numpy.

We will use the following abbreviations:

  * rfft: real fourier transform format as defined in numpy
            https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html

  * hc: half complex format as defined in fftw lib
          http://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html

In order to leverage vectorized operations of numpy, we will make the following
assumptions:

  * inputs and outputs are tensors like this:
      array[n_signal, nd_signal, signal_len]

  * signal_len is always a power of 2.
      This beneficiate from DFT optimizations

"""

# pytype: disable=unsupported-operands
# pytype: disable=attribute-error
# pylint: disable=line-too-long

from typing import Sequence

import numpy as np

# Type aliases here are solely for documentation, `Sequence[_NdSignal]` is
# truly a np.ndarray tensor.
# Hence we disable type hinting for .shape and __get_item__([slice, slice, ...])

_1dSignal = np.ndarray  # pylint: disable=invalid-name
_NdSignal = Sequence[_1dSignal]


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


def unstride_signal(strided_signal_list: Sequence[_1dSignal], signal_dim: int,
                    signal_len: int) -> Sequence[_NdSignal]:
  """Unstrides a 2d matrix into a sequence of nd signals."""
  return strided_signal_list.reshape(-1, signal_dim, signal_len)


def rfft_to_hc(rfft_t: Sequence[_NdSignal]) -> Sequence[_NdSignal]:
  """Converts a tensor from rfft to hc format."""
  rev_imag = np.flip(np.imag(rfft_t)[:, :, 1:-1], axis=2)
  return np.concatenate([np.real(rfft_t), rev_imag], axis=2)


def hc_to_rfft(hc_t: Sequence[_NdSignal]) -> Sequence[_NdSignal]:
  """Converts a tensor from hc to rfft format."""
  _, _, signal_len = hc_t.shape
  half_len = signal_len // 2

  rfft = np.array(hc_t[:, :, :half_len + 1], dtype=np.complex64)
  rfft[:, :, 1:-1] += 1j * np.flip(hc_t[:, :, half_len + 1:], axis=2)
  return rfft


def hc(t: Sequence[_NdSignal]) -> Sequence[_NdSignal]:
  """Transforms a time domain signal into half complex."""
  _, _, signal_len = t.shape
  assert signal_len % 2 == 0, 'signal_len must be even'
  return rfft_to_hc(np.fft.rfft(t))


def ihc(hc_t: Sequence[_NdSignal]) -> Sequence[_NdSignal]:
  """Transforms a half complex signal into the time domain."""
  return np.fft.irfft(hc_to_rfft(hc_t))


def hc_conjugate(hc_t: Sequence[_NdSignal]) -> Sequence[_NdSignal]:
  """Computes the conjugate of a hc tensor."""
  _, _, signal_len = hc_t.shape
  half_len = signal_len // 2

  conj_hc_t = np.array(hc_t)
  conj_hc_t[:, :, half_len + 1:] = -conj_hc_t[:, :, half_len + 1:]
  return conj_hc_t


def hc_itemwise_pdt(hc_t1: Sequence[_NdSignal],
                    hc_t2: Sequence[_NdSignal]) -> Sequence[_NdSignal]:
  """Multiplies 2 hc tensors itemwise to perform a convolution in time domain."""
  _, _, signal_len = hc_t1.shape
  half_len = signal_len // 2

  hc_pdt = np.zeros_like(hc_t1)

  # Handle first and last values in hc format  (IE: i == 0 or i == half_len)
  hc_pdt[:, :, 0] = hc_t1[:, :, 0] * hc_t2[:, :, 0]
  hc_pdt[:, :, half_len] = hc_t1[:, :, half_len] * hc_t2[:, :, half_len]

  # Handle remaining values in hc format (IE: 1 <= i < half_len)
  h_i_hc_t1_real = hc_t1[:, :, 1:half_len]
  h_i_hc_t1_imag = np.flip(hc_t1[:, :, half_len + 1:], axis=2)

  h_i_hc_t2_real = hc_t2[:, :, 1:half_len]
  h_i_hc_t2_imag = np.flip(hc_t2[:, :, half_len + 1:], axis=2)

  # real part of (a + ib) * (a' + ib'): a * a' - b * b'
  h_i_hc_real = h_i_hc_t1_real * h_i_hc_t2_real - h_i_hc_t1_imag * h_i_hc_t2_imag
  hc_pdt[:, :, 1:half_len] = h_i_hc_real

  # imag part of (a + ib) * (a' + ib'): a * b' + a' * b
  h_i_hc_imag = h_i_hc_t1_real * h_i_hc_t2_imag + h_i_hc_t1_imag * h_i_hc_t2_real
  hc_pdt[:, :, half_len + 1:] = np.flip(h_i_hc_imag, axis=2)

  return hc_pdt


def energy(t: Sequence[_NdSignal]) -> Sequence[Sequence[float]]:
  """Computes the energies of a tensor."""
  return np.power(t, 2).sum(axis=-1)


def rfft_energy(rfft_t: Sequence[_NdSignal]) -> Sequence[Sequence[float]]:
  """Computes the energies of a rfft tensor."""
  _, _, signal_len = rfft_t.shape
  true_len = (signal_len - 1) * 2

  # Continuous component
  h_0 = rfft_t[:, :, 0] * np.conjugate(rfft_t[:, :, 0])

  # Middle frequencies, they need to be counted twice in the case of rfft
  #   detailed explanation here: https://dsp.stackexchange.com/a/67110
  h_i = 2 * np.sum(
      rfft_t[:, :, 1:-1] * np.conjugate(rfft_t[:, :, 1:-1]), axis=2)

  # Nyquist frequency
  h_n = rfft_t[:, :, -1] * np.conjugate(rfft_t[:, :, -1])
  return np.real(h_0 + h_i + h_n) / true_len


def hc_energy(hc_t: Sequence[_NdSignal]) -> Sequence[Sequence[float]]:
  """Computes the energies of a hc tensor."""
  _, _, signal_len = hc_t.shape
  half_len = signal_len // 2

  # Continuous component
  h_0 = np.power(hc_t[:, :, 0], 2)

  # Middle frequencies, they need to be counted twice in the case of rfft
  #   detailed explanation here: https://dsp.stackexchange.com/a/67110
  h_i_real = hc_t[:, :, 1:half_len]
  h_i_imag = hc_t[:, :, half_len + 1:]  # no need to use np.flip for energy
  h_i = 2 * np.sum(np.power(h_i_real, 2) + np.power(h_i_imag, 2), axis=2)

  # Nyquist frequency
  h_n = np.power(hc_t[:, :, half_len], 2)
  return np.real(h_0 + h_i + h_n) / signal_len

