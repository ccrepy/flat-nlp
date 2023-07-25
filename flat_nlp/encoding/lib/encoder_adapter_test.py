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

"""Tests for encoder_adapter."""

import numpy as np

from flat_nlp.encoding.lib import encoder_adapter
from flat_nlp.encoding.lib import test_encoding_util
from flat_nlp.lib import np_signal_util

from absl.testing import absltest


class EncoderAdapterTest(absltest.TestCase):

  def test_base_encoder_adapter(self):
    encoder = encoder_adapter.BaseEncoderAdapter(
        test_encoding_util.SIMPLE_TEXT8_ENCODER)

    with self.assertRaises(NotImplementedError):
      _ = encoder.call(['one sentence'])

  def test_hc_encoder_adapter(self):
    encoder = encoder_adapter.HalfComplexEncoderAdapter(
        test_encoding_util.SIMPLE_TEXT8_ENCODER)

    self.assertTrue(encoder.is_output_hc)
    self.assertEqual(encoder.call(['one sentence']).shape, (1, 5, 8))

    np.testing.assert_allclose(
        test_encoding_util.SIMPLE_TEXT8_ENCODER.call(['one sentence']).numpy(),
        np_signal_util.ihc(encoder.call(['one sentence']).numpy()),
        atol=0.0001)

  def test_strided_output_encoder_adapter(self):
    encoder = encoder_adapter.StridedOutputEncoderAdapter(
        test_encoding_util.SIMPLE_TEXT8_ENCODER)

    self.assertTrue(encoder.is_output_strided)
    self.assertEqual(encoder.call(['one sentence']).shape, (1, 40))

    np.testing.assert_allclose(
        test_encoding_util.SIMPLE_TEXT8_ENCODER.call(['one sentence']).numpy(),
        np_signal_util.unstride_signal(
            encoder.call(['one sentence']).numpy(), encoder.vector_size,
            encoder.n_vector))


if __name__ == '__main__':
  absltest.main()
