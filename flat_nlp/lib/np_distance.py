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

"""Library for Flat NLP distance numpy implementation.

Distances are defined point-wise.
"""

import dataclasses

import numpy as np

from flat_nlp.lib import np_signal_util


def convolve_signal(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
  """Computes the discrete correlation of 2 Nd signals.

    The signals are not energy normalized.

    The second signal is flipped, so we perform a circular correlation and not a
    convolution.

  Arguments:
    s1: first nd signal to correlate, in time domain
    s2: second nd signal to correlate, in time domain

  Returns:
    The correlated signal in time domain
  """
  s2 = np.flip(s2, axis=-1)
  return np.fft.ifft(np.fft.fft(s1) * np.fft.fft(s2)).real


def convolve_hc_signal(
    hc_s1: np.ndarray, hc_s2: np.ndarray
) -> np_signal_util._NdSignal:
  """Computes the discrete correlation of 2 Nd signals in halfcomplex format.

    The signals are not energy normalized.

    The second signal is flipped, so we perform a circular correlation and not a
    convolution.

  Arguments:
    hc_s1: first nd signal to correlate, in halfcomplex format
    hc_s2: second nd signal to correlate, in halfcomplex format

  Returns:
    The correlated signal in halfcomplex format.
  """

  hc_t1, hc_t2, = np.array([hc_s1]), np.array([hc_s2])
  hc_t2 = np_signal_util.hc_conjugate(hc_t2)

  t_out = np_signal_util.ihc(np_signal_util.hc_itemwise_pdt(hc_t1, hc_t2))
  t_out = np.roll(t_out, -1, axis=-1)
  (hc_t_out,) = np_signal_util.hc(t_out)
  return hc_t_out


def flat_distance(s1: np.ndarray, s2: np.ndarray) -> float:
  """Computes flat distance between 2 Nd signal."""
  convolved_signal = convolve_signal(s1, s2)

  signal_energy = np.sqrt(
      np.power(s1, 2).sum(axis=-1) * np.power(s2, 2).sum(axis=-1)
  )
  convolved_signal = convolved_signal / signal_energy[:, np.newaxis]

  peak_correlation = np.average(convolved_signal, axis=0).max()
  return np.sqrt(1. - np.power(min(1., peak_correlation), 2))


def hc_flat_distance(hc_s1: np.ndarray, hc_s2: np.ndarray) -> float:
  """Computes flat distance between 2 Nd signal in halfcomplex format."""
  convolved_hc_signal = convolve_hc_signal(hc_s1, hc_s2)
  (convolved_signal,) = np_signal_util.ihc(np.array([convolved_hc_signal]))

  (hc_s1_energy,) = np.array(np_signal_util.hc_energy(np.array([hc_s1])))
  (hc_s2_energy,) = np.array(np_signal_util.hc_energy(np.array([hc_s2])))

  signal_energy = np.sqrt(hc_s1_energy * hc_s2_energy)
  convolved_signal = convolved_signal / signal_energy[:, np.newaxis]

  peak_correlation = np.average(convolved_signal, axis=0).max()
  return np.sqrt(1. - np.power(min(1., peak_correlation), 2))


@dataclasses.dataclass(frozen=True)
class StridedFlatDistanceFn:
  """Callable class wrapping flat_distance for strided inputs."""
  signal_dim_: int
  signal_len_: int

  def __call__(self, s1: np.ndarray, s2: np.ndarray) -> float:
    return flat_distance(
        s1.reshape(self.signal_dim_, self.signal_len_),
        s2.reshape(self.signal_dim_, self.signal_len_))


@dataclasses.dataclass(frozen=True)
class StridedHalfComplexFlatDistanceFn:
  """Callable class wrapping hc_flat_distance for strided inputs."""
  signal_dim_: int
  signal_len_: int

  def __call__(self, hc_s1: np.ndarray, hc_s2: np.ndarray) -> float:
    return hc_flat_distance(
        hc_s1.reshape(self.signal_dim_, self.signal_len_),
        hc_s2.reshape(self.signal_dim_, self.signal_len_))
