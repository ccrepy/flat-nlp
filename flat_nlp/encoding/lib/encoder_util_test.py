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

"""Tests for encoder_util."""

import numpy as np

from flat_nlp.encoding.lib import encoder_adapter
from flat_nlp.encoding.lib import encoder_util
from flat_nlp.encoding.lib import test_encoding_util

from absl.testing import absltest
from absl.testing import parameterized


class EncoderUtilTest(parameterized.TestCase):

  @parameterized.parameters(
      (False, False),
      (True, False),
      (False, True),
      (True, True),
  )
  def test_build_distance_fn(self, halfcomplex, strided):
    encoder = test_encoding_util.SIMPLE_TEXT8_ENCODER
    if halfcomplex:
      encoder = encoder_adapter.HalfComplexEncoderAdapter(encoder)
    if strided:
      encoder = encoder_adapter.StridedOutputEncoderAdapter(encoder)

    distance_fn = encoder_util.build_distance_fn(encoder)
    a, b = np.array(encoder.call(['the cat is here', 'a dog is there']))

    self.assertAlmostEqual(distance_fn(a, b), 0.4782338)

  def test_encoder_numpy_dtype(self):
    encoder = test_encoding_util.SIMPLE_TEXT8_ENCODER
    encoded_string = encoder.call(['dummy query']).numpy()

    # test dtype is float64 (double in cpp)
    self.assertEqual(encoded_string.dtype, np.float64)

    # test contiguous memory
    self.assertTrue(encoded_string.flags['C_CONTIGUOUS'])


if __name__ == '__main__':
  absltest.main()
