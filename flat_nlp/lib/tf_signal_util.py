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

"""Utility library for signal processing with tensorflow.

This file is the tensorflow counterpart of np_signal_util.py
"""

import tensorflow as tf


@tf.function
def stride_signal(signal_t):
  """Strides a sequence of nd signals into a 2d matrix."""
  n_signal, signal_dim, signal_len = signal_t.shape
  return tf.reshape(signal_t, (n_signal, signal_dim * signal_len))


@tf.function
def unstride_signal(strided_signal_t, signal_dim: int, signal_len: int):
  """Unstrides a 2d matrix into a sequence of nd signals."""
  return tf.reshape(strided_signal_t, (-1, signal_dim, signal_len))


@tf.function
def rfft_to_hc(rfft_t):
  """Converts a tensor from rfft to hc format."""
  return tf.concat(
      [tf.math.real(rfft_t),
       tf.reverse(tf.math.imag(rfft_t[:, :, 1:-1]), [2])],
      axis=2)


@tf.function
def hc_to_rfft(hc_t):
  """Converts a tensor from hc to rfft format."""
  _, _, signal_len = hc_t.shape
  half_len = signal_len // 2

  rev_imag = tf.reverse(hc_t[:, :, half_len + 1:], [2])

  return tf.complex(hc_t[:, :, :half_len + 1],
                    tf.pad(rev_imag, [[0, 0], [0, 0], [1, 1]], 'CONSTANT'))


@tf.function
def hc(t):
  """Transforms a time domain signal into half complex."""
  _, _, signal_len = t.shape
  assert signal_len % 2 == 0, 'signal_len must be even'
  return rfft_to_hc(tf.signal.rfft(t))


@tf.function
def ihc(hc_t):
  """Transforms a half complex signal into the time domain."""
  return tf.signal.irfft(hc_to_rfft(hc_t))


@tf.function
def hc_conjugate(hc_t):
  """Computes the conjugate of a hc tensor."""
  _, _, signal_len = hc_t.shape
  half_len = signal_len // 2

  return tf.concat([
      hc_t[:, :, :half_len + 1],
      -hc_t[:, :, half_len + 1:],
  ],
                   axis=2)


@tf.function
def hc_itemwise_pdt(hc_t1, hc_t2):
  """Multiplies 2 hc tensors itemwise to perform a convolution in time domain."""
  _, _, signal_len = hc_t1.shape
  half_len = signal_len // 2

  # Identify real and imag part of the complex harmonics (IE: 1 <= i < half_len)
  h_i_hc_t1_real = hc_t1[:, :, 1:half_len]
  h_i_hc_t1_imag = tf.reverse(hc_t1[:, :, half_len + 1:], [2])

  h_i_hc_t2_real = hc_t2[:, :, 1:half_len]
  h_i_hc_t2_imag = tf.reverse(hc_t2[:, :, half_len + 1:], [2])

  # Rebuild the signal by parts using hc convention:
  #   [
  #     h0,
  #     real part of (a + ib) * (a' + ib'): a * a' - b * b',
  #     hn,
  #     reversed [ imag part of (a + ib) * (a' + ib'): a * b' + a' * b],
  #   ]
  return tf.concat([
      hc_t1[:, :, :1] * hc_t2[:, :, :1],
      h_i_hc_t1_real * h_i_hc_t2_real - h_i_hc_t1_imag * h_i_hc_t2_imag,
      hc_t1[:, :, half_len:half_len + 1] * hc_t2[:, :, half_len:half_len + 1],
      tf.reverse(
          h_i_hc_t1_real * h_i_hc_t2_imag + h_i_hc_t1_imag * h_i_hc_t2_real,
          [2]),
  ],
                   axis=2)


@tf.function
def rfft_energy(rfft_t):
  """Computes the energies of a rfft tensor."""
  _, _, signal_len = rfft_t.shape
  true_len = (signal_len - 1) * 2

  # Continuous component
  h_0 = rfft_t[:, :, 0] * tf.math.conj(rfft_t[:, :, 0])

  # Middle frequencies, they need to be counted twice in the case of rfft
  #   detailed explanation here: https://dsp.stackexchange.com/a/67110
  h_i = 2 * tf.math.reduce_sum(
      rfft_t[:, :, 1:-1] * tf.math.conj(rfft_t[:, :, 1:-1]), [2])

  # Nyquist frequency
  h_n = rfft_t[:, :, -1] * tf.math.conj(rfft_t[:, :, -1])
  return tf.math.real(h_0 + h_i + h_n) / true_len


@tf.function
def hc_energy(hc_t):
  """Computes the energies of a hc tensor."""
  _, _, signal_len = hc_t.shape
  half_len = signal_len // 2

  # Continuous component
  h_0 = tf.math.pow(hc_t[:, :, 0], 2)

  # Middle frequencies, they need to be counted twice in the case of rfft
  #   detailed explanation here: https://dsp.stackexchange.com/a/67110
  h_i_real = hc_t[:, :, 1:half_len]
  h_i_imag = hc_t[:, :, half_len + 1:]  # no need to use tf.reverse
  h_i = 2 * tf.math.reduce_sum(
      tf.math.pow(h_i_real, 2) + tf.math.pow(h_i_imag, 2), [2])

  # Nyquist frequency
  h_n = tf.math.pow(hc_t[:, :, half_len], 2)
  return tf.math.real(h_0 + h_i + h_n) / signal_len
